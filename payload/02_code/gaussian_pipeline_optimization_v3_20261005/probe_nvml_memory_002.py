"""Check current native driver memory fields against the existing CLI interface."""

import argparse
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from nvml_memory_monitor_002 import NativeMemory, LIBRARY, MIB, cli_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("nvml_memory_monitor_002.py"), LIBRARY]
    sources = [data_identity(p) for p in paths]
    native, rows = NativeMemory(), []
    try:
        for _ in range(3):
            before = {v: native.read(v) for v in ("v1", "v2")}
            cli = cli_memory()
            after = {v: native.read(v) for v in ("v1", "v2")}
            rows.append({"before": before, "CLI_used_free_MiB": cli, "after": after,
                "compatible": {v: ((before[v]["used"]+MIB-1)//MIB, (before[v]["free"]+MIB-1)//MIB) == cli
                    == ((after[v]["used"]+MIB-1)//MIB, (after[v]["free"]+MIB-1)//MIB) for v in ("v1", "v2")}})
    finally:
        native.close()
    checks = {"owned_library_shutdown": native.closed and not native.initialized,
        "all_sources_unchanged": sources == [data_identity(p) for p in paths],
        "some_memory_ABI_exact_all3_cli_brackets": any(all(row["compatible"][v] for row in rows) for v in ("v1", "v2"))}
    result = {"status": "interface_confirmed_cost_not_run" if all(checks.values()) else "unconfirmed_interface",
        "checks": checks, "rows": rows, "metadata": native.metadata, "sources": sources,
        "compatible_routes": [v for v in ("v1", "v2") if all(row["compatible"][v] for row in rows)],
        "whole_seconds": time.perf_counter()-STARTED, "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
