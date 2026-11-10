"""Read only recorded R115 lease and process identities after every node closed."""

import json
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/origin_dependency_001"
target = OUTPUT / "POST_CLOSE_PID_RECEIPT.json"
if target.exists():
    raise FileExistsError("The post-close process receipt must be fresh")
receipts = []
processes = []
for path in sorted(OUTPUT.glob("LEASE_*.json")):
    row = json.loads(path.read_text(encoding="utf-8"))
    if row["status"] != "child_exited" or not row.get("lease_released_utc"):
        raise AssertionError("An owned historical node still has no release receipt")
    receipts.append({"path": str(path), "returncode": row["returncode"],
                     "released_utc": row["lease_released_utc"]})
    for prefix in ("guard", "child"):
        pid, creation = row[prefix + "_pid"], row[prefix + "_creation_time"]
        try:
            current = psutil.Process(pid).create_time()
            same = abs(current - creation) < .001
        except psutil.NoSuchProcess:
            current, same = None, False
        processes.append({"pid": pid, "recorded_creation_time": creation,
                          "current_creation_time": current, "same_owned_process_alive": same})
if any(row["same_owned_process_alive"] for row in processes):
    raise AssertionError("An owned controller or child remains alive")
payload = {"status": "all_owned_historical_controllers_and_children_closed",
           "lease_count": len(receipts), "process_identity_count": len(processes),
           "receipts": receipts, "processes": processes,
           "new_GPU_or_performance_experiment": False, "counted": False}
target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps({key: payload[key] for key in ("status", "lease_count", "process_identity_count", "counted")}))
