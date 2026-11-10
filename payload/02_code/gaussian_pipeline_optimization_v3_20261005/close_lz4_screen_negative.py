"""Close an actual negative screen without treating it as a formal campaign."""

import time

STARTED = time.perf_counter()

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/lz4_transport_001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def windows_path(path):
    if not path.startswith("/mnt/c/"):
        raise ValueError("An identity is outside the registered C drive")
    return Path("C:/" + path[len("/mnt/c/"):])


def gone(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    if (OUT / "NEGATIVE_CLOSURE_001.json").exists():
        raise FileExistsError("Never overwrite a screen closure")
    result = read(OUT / "p0/RESULT.json")
    lease = read(OUT / "SCREEN_LEASE_001.json")
    win = read(OUT / "SCREEN_WINDOWS_GUARD_001.json")
    lin = read(OUT / "SCREEN_LINUX_GUARD_001.json")
    identities = result["sources"] + result["inputs"]
    current = [all(data_identity(windows_path(r["path"]))[k] == r[k] for k in ["bytes", "sha256"]) for r in identities]
    copies = []
    for i, r in enumerate(result["sources"]):
        path = OUT / "p0" / f"SOURCE_{i:02d}_{Path(r['path']).name}"
        raw = path.read_bytes()
        copies.append(len(raw) == r["bytes"] and hashlib.sha256(raw).hexdigest() == r["sha256"])
    rows = win["owned_process_identities"] + [dict(pid=lease[k + "_pid"], creation_time=lease[k + "_creation_time"]) for k in ["guard", "child"]]
    windows = [dict(r, gone=gone(r)) for r in rows]
    code = "import json,psutil; rows=json.loads(" + repr(json.dumps(lin["owned_process_identities"])) + "); out=[]\n"
    code += "for r in rows:\n alive=False\n try:\n  p=psutil.Process(r['pid']); alive=p.create_time()==r['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE\n except psutil.NoSuchProcess: pass\n out.append(dict(r,alive=alive))\nprint(json.dumps(out))"
    linux = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=30))
    old = read(OUT.parent / "codec_algorithms_001/PROTOCOL.json")["sources_inputs"][-3:]
    checks = {
        "registered_diagnostic_speed_gate_failed": result["status"] == "screen_negative_no_promotion"
            and not result["checks"]["complete_three_source_diagnostic_stage_at_least5percent"],
        "other_complete_payload_layout_source_storage_and_CPU_guards_passed": all(v for k, v in result["checks"].items() if k != "complete_three_source_diagnostic_stage_at_least5percent"),
        "all_eight_current_source_input_identities": len(current) == 8 and all(current),
        "all_five_recorded_source_copies": len(copies) == 5 and all(copies),
        "all_three_original_R84_input_revisions_retained": result["inputs"] == old,
        "actual_both_guards_and_lease_terminal_released": win["status"] == lin["status"] == "completed"
            and win["returncode"] == lin["returncode"] == lease["returncode"] == 0 and bool(lease.get("lease_released_utc")),
        "all_recorded_six_Windows_instances_gone": len(windows) == 6 and all(r["gone"] for r in windows),
        "recorded_Linux_child_instance_gone": all(not r["alive"] for r in linux),
        "no_formal_generic_native_recovery_promotion": not (OUT / "ACCEPTANCE.json").exists()
            and result["new_optimization_count"] == 0,
    }
    seconds = time.perf_counter() - STARTED
    payload = {
        "status": "closed_negative_screen_count0" if all(checks.values()) else "failed_close",
        "checks": checks, "new_optimization_count": 0, "Windows": windows, "Linux": linux,
        "complete_three_source_diagnostic_seconds": result["complete_three_source_stage_seconds"],
        "complete_three_source_diagnostic_reduction": result["complete_three_source_stage_reduction"],
        "archive_ratios_to_zstd": result["archive_ratios_to_zstd"],
        "actual_screen_lease_child_seconds": lease["child_elapsed_seconds"],
        "negative_closure_before_publication_seconds": seconds,
        "all_measured_nonoverlapping_screen_and_closure_seconds": lease["child_elapsed_seconds"] + seconds,
        "single_pass_screen_not_three_pair_formal_medians": True,
        "formal_generic_faults_actual_training_recovery_external_GT_preview_production": "not_run",
        "no_subagent_root_single_chat_self_review": True,
        "closing_script_own_pid": psutil.Process().pid,
        "closing_script_own_creation_time": psutil.Process().create_time(),
        "own_closing_script_exit_requires_outer_check": True,
        "small_text_publication_after_close_not_claimed_included": True,
        "closing_source": data_identity(Path(__file__)),
    }
    atomic_json(OUT / "NEGATIVE_CLOSURE_001.json", payload)
    if not all(checks.values()):
        raise AssertionError("An actual screen source, output, cost or closure gate failed")
    print(json.dumps({k: payload[k] for k in ["status", "checks", "all_measured_nonoverlapping_screen_and_closure_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
