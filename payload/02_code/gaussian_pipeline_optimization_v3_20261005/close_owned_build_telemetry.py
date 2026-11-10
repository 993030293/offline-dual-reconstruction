"""Close only the exact MSVC telemetry instance saved by this private build guard."""

import argparse
from pathlib import Path
import time

import psutil

import profile_quality_corner_cost as reference

OUT = reference.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"


def matches(row, saved, expected):
    return (Path(row["executable"]) == expected and any(row["pid"] == item["pid"] and
        row["creation_time"] == item["creation_time"] for item in saved["owned_process_identities"])
        and any(row["parent_pid"] == item["pid"] for item in saved["owned_process_identities"]))


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    base = OUT / "vector_gray_stats_001"
    output.relative_to(base)
    output.mkdir(parents=True, exist_ok=False)
    saved = reference.load(base / "GUARD_B0.json")
    ownership = reference.load(base / "BUILD_TELEMETRY_OWNERSHIP_001.json")
    expected = Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/vctip.exe")
    checks = {"valid_record_matches": matches(ownership, saved, expected),
        "different_birth_rejected": not matches(dict(ownership, creation_time=ownership["creation_time"] + 1), saved, expected),
        "different_executable_rejected": not matches(dict(ownership, executable="C:/unrelated.exe"), saved, expected),
        "different_parent_rejected": not matches(dict(ownership, parent_pid=-1), saved, expected),
        "different_PID_rejected": not matches(dict(ownership, pid=-1), saved, expected)}
    if not all(checks.values()):
        raise ValueError("The strict saved build-process ownership regression failed")
    process = psutil.Process(ownership["pid"])
    live = {"pid": process.pid, "creation_time": process.create_time(), "executable": process.exe(), "parent_pid": process.ppid()}
    if not matches(live, saved, expected) or live != {key: ownership[key] for key in live}:
        raise ValueError("The exact build telemetry instance changed before close")
    library = base / "b0/native_vector_gray_stats.dll"
    before = reference.data_identity(library)
    process.terminate()
    returncode = process.wait(timeout=10)
    try:
        same_still_alive = psutil.Process(live["pid"]).create_time() == live["creation_time"]
    except psutil.NoSuchProcess:
        same_still_alive = False
    checks["exact_owned_telemetry_instance_exited"] = not same_still_alive
    checks["original_private_DLL_bytes_unchanged"] = before == reference.data_identity(library)
    if not all(checks.values()):
        raise ValueError("The original owned telemetry close contract failed")
    reference.atomic_json(output / "RESULT.json", {"status": "exact_owned_build_telemetry_closed",
        "checks": checks, "closed_instance": live, "returncode": returncode, "source": reference.data_identity(Path(__file__)),
        "source_snapshot": Path(__file__).read_text(encoding="utf-8"), "seconds": time.perf_counter() - started,
        "new_optimization_count": 0, "original_build_full_pairs_and_negative_review_not_rerun": True})
    print({"status": "exact_owned_build_telemetry_closed", "checks": checks}, flush=True)


if __name__ == "__main__":
    main()
