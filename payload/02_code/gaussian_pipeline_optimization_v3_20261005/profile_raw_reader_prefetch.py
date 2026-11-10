"""Reuse the original RAW128 scorer screen with a common accepted quality workspace."""

import argparse
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/raw_reader_scorer_dag_001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import data_identity
import profile_native_quality_workspace as original
from raw_reader_prefetch import raw_reader_prefetch_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generic", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OWN)
    generic = original.load(args.generic / "RESULT.json")
    if generic["sources"] != [data_identity(path) for path in
            (CODE / "check_raw_reader_prefetch.py", CODE / "raw_reader_prefetch.py",
             original.V1 / "quality_engine.py", original.V1 / "run_initial_pipeline.py")]:
        raise ValueError("A frozen current RAW reader source differs from its generic")
    old_generic = original.load(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/native_quality_workspace_001/g0/RESULT.json")
    if data_identity(CODE / "native_quality_workspace.py") != old_generic["sources"][1]:
        raise ValueError("The common accepted quality workspace source changed")
    source = inspect.getsource(original.main)
    changes = {
        'gaussian_pipeline_optimization_v3_20261005/native_quality_workspace_001':
            'gaussian_pipeline_optimization_v3_20261005/raw_reader_scorer_dag_001',
        'generic_passed_pilot_and_complete_stage_pending': 'generic_original_scorer_passed_actual_RAW_pilot_pending',
        'baseline.TOOLS / "mcap_audit.py", baseline.TOOLS / "camera_motion.py", baseline.TOOLS / "color_contract.py", exclusion]':
            'baseline.TOOLS / "mcap_audit.py", baseline.TOOLS / "camera_motion.py", baseline.TOOLS / "color_contract.py", exclusion, _reader_source, _original_profile_source]',
        'if sources[1] != generic["sources"][1] or sources[2] != generic["sources"][2] or sources[3] != generic["sources"][3]:':
            'if sources[-2] != generic["sources"][1] or sources[3] != generic["sources"][2]:',
        'with native_quality_workspace_execution(mode == "workspace") as workspace_trace:':
            'with raw_reader_prefetch_execution(mode == "workspace") as reader_trace, native_quality_workspace_execution(True) as workspace_trace:',
        '"histogram": histogram_trace, "peak_process_RSS_bytes":':
            '"histogram": histogram_trace, "reader": reader_trace, "peak_process_RSS_bytes":',
        'if workspace is not None and (not workspace["closed"] or workspace["owned_plane_bytes_after_close"] != 0 or workspace["calls"] != 384):':
            'if not reader_trace["binding_restored"] or (mode == "workspace" and (len(reader_trace["calls"]) != 1 or reader_trace["calls"][0]["delivered"] != 128 or not reader_trace["calls"][0]["closed"] or reader_trace["calls"][0]["reader_alive_after_join"])) or (workspace is not None and (not workspace["closed"] or workspace["owned_plane_bytes_after_close"] != 0 or workspace["calls"] != 384)):',
        'accepted_count_contribution=0)':
            'accepted_count_contribution=0, common_R141_workspace_both_sides=True, legacy_mode_workspace_denotes_only_new_RAW_prefetch=True)',
    }
    for before, after in changes.items():
        if source.count(before) != 1:
            raise ValueError("The unchanged complete RAW pilot source anchor changed: " + before)
        source = source.replace(before, after)
    namespace = dict(vars(original))
    namespace.update(__file__=str(Path(__file__)), raw_reader_prefetch_execution=raw_reader_prefetch_execution,
        _reader_source=CODE / "raw_reader_prefetch.py", _original_profile_source=Path(original.__file__))
    exec(compile(source, "<original_RAW128_common_workspace_reader_DAG>", "exec"), namespace)
    try:
        namespace["main"]()
    finally:
        if args.output.is_dir():
            (args.output / "COMPOSED_PILOT_ENTRY.py").write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
