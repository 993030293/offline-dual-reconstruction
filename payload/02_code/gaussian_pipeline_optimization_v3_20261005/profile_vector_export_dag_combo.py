"""Run the actual original R140 DAG with a guarded sampling-only adapter."""

import argparse
import difflib
from pathlib import Path
import sys

from profile_vector_full_export import BASE, LIBRARY
from vector_artifact_execution import vector_artifact_execution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling", choices=("native", "vector"), required=True)
    args, remaining = parser.parse_known_args()
    original = Path(__file__).with_name("profile_export_qa_dag.py")
    before = original.read_text(encoding="utf-8")
    source = before
    anchor = 'BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/export_qa_dag_001"'
    if source.count(anchor) != 1: raise ValueError("The exact original profile output scope changed")
    source = source.replace(anchor, 'BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/ud001/v5_simd_001"')
    source = source.replace('sources = [Path(__file__), code / "export_qa_dag.py",',
        'sources = [Path(__file__), code / "profile_export_qa_dag.py", code / "profile_vector_full_export.py", code / "vector_artifact_execution.py", '
        'code / "vector_export_qa_artifact_worker.py", code / "dataset_sampling_shared_carrier.py", '
        'code / "dataset_sampling_carrier.py", code / "complete_carrier_configuration.py", code / "vector_camera_sampling.py", '
        'code / "native_vector_camera_sampling.cpp", code / "native_camera_sampling.py", code / "camera_pixel_remap.py", '
        '_r123_library, code / "export_qa_dag.py",')
    # The original snapshot loop copies source texts; the DLL remains identity guarded.
    source = source.replace('        shutil.copyfile(path, copies / (f"{index:03d}_" + path.name))',
        '        if path.suffix != ".dll":\n            shutil.copyfile(path, copies / (f"{index:03d}_" + path.name))')
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "retained_original_DAG_profile", "_r123_library": LIBRARY}
    exec(compile(source, "<retained_original_R140_DAG_sampling_combo>", "exec"), namespace)
    previous = sys.argv
    try:
        sys.argv = [str(original), *remaining]
        output_index = remaining.index("--output") + 1
        output = Path(remaining[output_index]).resolve()
        proof = output.with_name(output.name + "_adapter")
        with vector_artifact_execution(args.sampling == "vector", proof):
            (proof / "COMPLETE_EXECUTED_DAG_PROFILE.py").write_text(source, encoding="utf-8")
            (proof / "DAG_PROFILE_ADAPTER.diff").write_text("".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True),
                fromfile=original.name, tofile="sampling_only_DAG_combo_profile")), encoding="utf-8")
            namespace["main"]()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
