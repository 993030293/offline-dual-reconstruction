"""Release a stale GPU owner only after exact Windows and WSL drainage checks."""

import argparse
import json
import msvcrt
from pathlib import Path
import subprocess
import time

import psutil

from atomic_json_windows import atomic_json, read_json


def read(path):
    return read_json(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--failed-side", required=True, type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    side = args.failed_side.resolve()
    side.relative_to(campaign)
    owner_path = campaign / "gpu_coordination/OWNER.json"
    owner = read(owner_path)
    stage = read(side / "training_process.json")
    controller = read(side / "PIPELINE_STATE.json")
    if owner["status"] != "acquired" or stage["status"] != "failed_controller":
        raise ValueError("This repair requires the explicitly failed training monitor and stale acquired lease")
    Path(owner["receipt"]).resolve().relative_to(side)
    records = stage["owned_process_identities"] + [
        {"pid": owner["pid"], "creation_time": owner["process_birth"]},
        {"pid": controller["pid"], "creation_time": controller["process_birth"]}]
    checked = []
    with (campaign / "gpu_coordination/GPU.lock").open("r+b", buffering=0) as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            if read(owner_path) != owner:
                raise RuntimeError("The GPU owner changed while acquiring its lock")
            for row in records:
                try:
                    item = psutil.Process(row["pid"])
                    birth = item.create_time()
                    if birth == row["creation_time"]:
                        raise RuntimeError("An exact failed GPU process or controller remains alive")
                    checked.append({**row, "status": "PID_reused", "current_birth": birth})
                except psutil.NoSuchProcess:
                    checked.append({**row, "status": "no_such_process"})
            prefix = "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_full_raw_ab_20261008/"
            probe = "import psutil,json; prefix=" + repr(prefix) + "; rows=[]; denied=[]\n"
            probe += "for p in psutil.process_iter():\n"
            probe += " try:\n  c=p.cmdline()\n  if any(a.startswith(prefix) for a in c): rows.append({'pid':p.pid,'birth':p.create_time(),'argv':c})\n"
            probe += " except psutil.NoSuchProcess: pass\n except psutil.AccessDenied: denied.append(p.pid)\n"
            probe += "print(json.dumps({'owned_path_processes':rows,'access_denied':denied}))"
            output = subprocess.check_output(["wsl.exe", "-d", "E26-C2-LIV-20260907-134802", "--",
                "/opt/e26_thin_surface_v1/venv/bin/python", "-c", probe], text=True, timeout=30)
            wsl = json.loads(output)
            if wsl["owned_path_processes"] or wsl["access_denied"]:
                raise RuntimeError("WSL drainage has not been established")
            if read(owner_path) != owner:
                raise RuntimeError("The GPU owner changed during drainage inspection")
            archive = campaign / ("gpu_owner_reclaim_" + str(time.time_ns()))
            archive.mkdir(exist_ok=False)
            atomic_json(archive / "ORIGINAL_OWNER.json", owner)
            evidence = {"status": "released_after_verified_dead_owner", "checked_unix": time.time(),
                "windows_exact_identities": checked, "WSL": wsl, "OS_byte_lock_acquired": True,
                "original_owner": owner, "failed_stage_receipt": str(side / "training_process.json"),
                "no_process_termination_performed": True}
            atomic_json(archive / "RECLAIM_RECEIPT.json", evidence)
            atomic_json(owner_path, {**owner, "status": "released", "verified_reclaim": str(archive),
                                     "released_unix": time.time()})
            print(json.dumps({"status": evidence["status"], "evidence": str(archive)}))
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    main()
