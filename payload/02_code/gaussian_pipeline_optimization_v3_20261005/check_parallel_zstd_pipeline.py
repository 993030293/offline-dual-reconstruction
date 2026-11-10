"""Check lazy worker admission and actual ordered native archive fixtures."""

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from check_generic_checkpoint_transport import tensor_schema
from run_codec_campaign import exact
import parallel_zstd_checkpoint as pipeline
import zstd_checkpoint as native


def pipeline_trace_valid(trace, require_overlap=True):
    if (trace["workers"] != 2 or trace["capacity"] != 4
            or not 1 <= trace["worker_count_observed"] <= 2
            or not 1 <= trace["peak_pending"] <= 4 or not trace["all_owned_workers_joined"]
            or not trace["single_ordered_writer"]):
        return False
    intervals, writes = trace["compression_intervals"], trace["write_intervals"]
    if not intervals or len(intervals) != len(writes) or len({r["thread"] for r in intervals}) != trace["worker_count_observed"]:
        return False
    if not all(math.isfinite(r[k]) for r in intervals + writes for k in ["started", "ended"]):
        return False
    if not all(r["ended"] >= r["started"] for r in intervals + writes):
        return False
    overlap = sum(a["started"] < b["ended"] and b["started"] < a["ended"] for a in intervals for b in writes)
    return overlap == trace["actual_compression_write_overlaps"] and (overlap > 0 or not require_overlap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scope = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/parallel_zstd_transport_001"
    args.output.resolve().relative_to(scope.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [data_identity(p) for p in [Path(__file__), Path(pipeline.__file__), Path(native.__file__)]]
    for i, identity in enumerate(sources):
        shutil.copyfile(identity["path"], args.output / f"SOURCE_{i:02d}_{Path(identity['path']).name}")
    first = json.loads((scope / "p0/RESULT.json").read_text())
    checks = {}
    for r in first["rows"]:
        checks[f"actual_input{r['input']}_lazy_worker_budget_parallel_write_join"] = pipeline_trace_valid(r["sides"]["pipeline"]["encode"]["ordered_pipeline"])
    real = first["rows"][0]["sides"]["pipeline"]["encode"]["ordered_pipeline"]
    for name, key, value in [("zero_workers", "worker_count_observed", 0), ("excess_workers", "worker_count_observed", 3),
            ("excess_queue", "peak_pending", 5), ("missing_join", "all_owned_workers_joined", False),
            ("wrong_capacity", "capacity", 2), ("fake_overlap", "actual_compression_write_overlaps", 0)]:
        malformed = copy.deepcopy(real)
        malformed[key] = value
        checks[name + "_rejected"] = not pipeline_trace_valid(malformed)
    import torch
    g = torch.Generator().manual_seed(20261006)
    shared = torch.randn((37, 11), generator=g)
    fixtures = {
        "mixed_alias_empty": dict(base=shared, view=shared.T, offset=shared[2:19:2, 1:8],
            empty=torch.empty((0, 7)), ids=torch.arange(19, dtype=torch.int64),
            boolean=torch.arange(19) % 3 == 0, half=shared.half(), bfloat=shared.bfloat16(),
            doubles=shared.double(), complex=torch.complex(shared, shared * .25),
            queue=[3, 0, 2], CPU_RNG=g.get_state()),
        "multiblock_entropy": dict(auxiliary=torch.randint(0, 256, (9 * 1024**2 + 31,), dtype=torch.uint8, generator=g)),
    }
    for name, value in fixtures.items():
        folder = args.output / name
        folder.mkdir()
        source = folder / "source.pth"
        torch.save(value, source)
        a = native.encode_checkpoint(source, folder / "serial.bin")
        b = pipeline.encode_checkpoint(source, folder / "pipeline.bin")
        expanded = folder / "expanded.pth"
        native.verified_expand(folder / "pipeline.bin", expanded)
        original, restored = [torch.load(p, map_location="cpu", weights_only=False) for p in [source, expanded]]
        checks[name + "_native_archive_body_exact"] = all(a["archive"][k] == b["archive"][k] for k in ["sha256", "bytes"])
        checks[name + "_all_native_state_values_alias_layout"] = exact(original, restored) and tensor_schema(original) == tensor_schema(restored)
        checks[name + "_lazy_worker_budget_and_join"] = pipeline_trace_valid(b["ordered_pipeline"], require_overlap=False)
    checks["all_original_non_worker_screen_guards_passed"] = all(v for k, v in first["checks"].items() if "actual_bounded_parallel_write_and_join" not in k)
    checks["current_three_generic_sources_unchanged"] = sources == [data_identity(Path(r["path"])) for r in sources]
    checks["CPU_only_without_CUDA_initialization"] = not torch.cuda.is_initialized()
    result = {
        "status": "generic_and_lazy_worker_screen_correction_passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "sources": sources, "new_optimization_count": 0,
        "original_p0_RESULT_unchanged": True,
        "correction": "ThreadPoolExecutor max_workers bounds lazy workers; original screen demanded exactly two for each file",
        "original_actual_17point569251percent_encoding_3point307712percent_whole_screen_not_formal": True,
        "remaining_full_faults_formal_actual_native_recovery": "pending",
    }
    atomic_json(args.output / "RESULT.json", result)
    if not all(checks.values()):
        raise AssertionError("A lazy worker, malformed trace or actual native fixture guard failed")
    print(json.dumps({k: result[k] for k in ["status", "checks"]}), flush=True)


if __name__ == "__main__":
    main()
