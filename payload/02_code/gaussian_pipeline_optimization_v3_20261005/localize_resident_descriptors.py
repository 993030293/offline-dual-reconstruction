"""Exercise private CUDA descriptors in the full R90/R92/R94/R102 pose route."""

import time
_import_started = time.perf_counter()
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
import localize_prepared_queries
from resident_sift_matching import resident_sift_transport
_import_seconds = time.perf_counter()-_import_started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport-mode", choices=("incumbent", "resident"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.output.exists():
        raise FileExistsError("The actual residency pose output must be unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), *(Path(__file__).with_name(name) for name in ("descriptor_residency_plan.py",
        "resident_sift_worker.py", "resident_sift_matching.py", "integer_sift_worker.py", "integer_sift_matching.py",
        "generic_sift_worker.py", "generic_sift_matching.py", "query_preparation_pipeline.py", "localize_prepared_queries.py"))]
    sources = [data_identity(path) for path in paths]
    previous = sys.argv
    try:
        with resident_sift_transport(args.transport_mode == "resident") as adapter:
            sys.argv = [str(Path(localize_prepared_queries.__file__)), "--preparation-mode", "parallel",
                "--output", str(args.output), *remaining]
            localize_prepared_queries.main()
        found = list(args.output.rglob("INTEGER_SIFT_TRACE.json"))
        if len(found) != 1:
            raise ValueError("The actual original matching worker trace must be unambiguous")
        worker = json.loads(found[0].read_text(encoding="utf-8"))["trace"]
        stage = _import_seconds+adapter["adapter_setup_seconds"]+adapter["adapter_restore_seconds"]
        stage += worker["startup_seconds"]+worker["shutdown_seconds"]
        stage += sum(row["total_prepare_IPC_kernel_seconds"] for row in worker["calls"])
        if sources != [data_identity(path) for path in paths] or not adapter["original_worker_factory_restored"]:
            raise ValueError("An actual descriptor residency source or worker adapter changed")
        atomic_json(args.output/"RESIDENCY.json", {"mode": args.transport_mode, "sources": sources, "adapter": adapter,
            "common_adapter_and_dependency_import_seconds": _import_seconds, "complete_matching_worker_stage_seconds": stage,
            "complete_stage_includes_startup_all_encoding_content_identity_IPC_allocation_GPU_cache_clear_close_and_imports": True,
            "worker": worker, "same_original_current_R90_R92_R94_R102_all24_48": True, "counted": False})
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
