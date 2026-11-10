"""Keep all six fresh full pairs in one lease and retain unchanged science gates."""

import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup019/R171"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from run_sift_process_campaign_005 import compare_pair


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    started = time.perf_counter()
    campaign = OUT / "c0"
    campaign.mkdir(parents=True, exist_ok=False)
    cpu = read(OUT.parent / "cpu0/RESULT.json")
    public = read(OUT / "g0/RESULT.json")
    for component in (cpu, public):
        if not all(component["checks"].values()) or any(data_identity(Path(row["path"])) != row for row in component["sources"]):
            raise ValueError("All complete component sources and checks must remain intact")
    paths = [Path(__file__), *[CODE / name for name in ("localize_strict_encoding.py", "strict_sift_encoding.py",
        "native_sift_encoding.cpp", "strict_encoding_node.py", "strict_encoding_owned_json_write.py",
        "codec_localization_node.py", "sift_process_node_003.py", "localize_geometry_sift_fusion.py",
        "run_sift_process_campaign_005.py", "run_bounded_journal_task.py")],
        ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005/compare_actual_localization.py",
        OUT.parent / "build0/native_sift_encoding.dll"]
    sources = [data_identity(path) for path in paths]
    for number, path in enumerate(paths):
        shutil.copyfile(path, campaign / (f"SOURCE_{number:02d}_" + path.name))
    cpu_bytes = sum(row[mode]["trace"]["actual_recorded_successful_output_bytes"]
                    for row in cpu["trials"] for mode in ("raw", "native"))
    cpu_bytes += sum(trace["actual_recorded_successful_output_bytes"] for row in cpu["generic"] for trace in row["traces"])
    fee_bytes = cpu_bytes + public["extra_actual_array_output_bytes"] + 18*96233216 + 18*(4*96233216+31983305)+2*1024**3
    state = {"status": "running", "sources": sources, "orders": [["raw", "native"], ["native", "raw"], ["raw", "native"]],
             "pairs": [], "runs": [], "new_optimization_count": 0,
             "known_current_extra_array_output_bytes_plus_registered2GiB": fee_bytes,
             "maximum_each_complete_whole_ratio": 1.1, "maximum_whole_median_ratio": 1.03,
             "minimum_net_array_output_reduction": .1, "no_speed_production_or_hardware_memory_claim": True}
    atomic_json(campaign / "PROTOCOL.json", state)
    try:
        for trial, order in enumerate(state["orders"]):
            folder = campaign / ("t" + str(trial))
            folder.mkdir()
            rows = {}
            for mode in order:
                if time.perf_counter()-started > 850 or sources != [data_identity(path) for path in paths]:
                    raise RuntimeError("The bounded complete frozen campaign changed")
                tag = "b" if mode == "raw" else "c"
                output, node = folder / tag, folder / ("n" + tag)
                command = [sys.executable, "-B", str(CODE / "strict_encoding_node.py"), "--kind", "localization",
                    "--mode", "process", "--node", str(node), "--output", str(output), "--", sys.executable, "-B",
                    str(CODE / "localize_strict_encoding.py"), "--encoding-mode", mode, "--output", str(output)]
                state.update(current_trial=trial, current_mode=mode)
                atomic_json(campaign / "STATE.json", state)
                before = time.perf_counter()
                with (folder / (mode + "_controller.log")).open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    actor = {"pid": process.pid, "creation_time": psutil.Process(process.pid).create_time()}
                    code = process.wait(timeout=900)
                elapsed = time.perf_counter()-before
                node_result = read(node / "NODE.json")
                result = read(output / "RESULT.json") if (output / "RESULT.json").exists() else None
                row = {"mode": mode, "owner": actor, "returncode": code, "output": str(output), "node": str(node),
                       "complete_parent_window_seconds": elapsed, "command": command}
                state["runs"].append(row)
                atomic_json(campaign / "STATE.json", state)
                if code or node_result["status"] != "completed" or node_result["live_Windows_after_wait"] or any(
                        item["alive"] for item in node_result["actual_Linux_after_wait"]) or not result or not all(result["checks"].values()):
                    raise RuntimeError("An actual original full24/48 strict-encoding node failed")
                row["actual_array_output_bytes"] = result["actual_array_output_bytes"]
                rows[mode] = row
            science = compare_pair(folder / "b", folder / "c")
            atomic_json(folder / "INVARIANTS.json", science)
            b, c = rows["raw"], rows["native"]
            net_upper = (b["actual_array_output_bytes"]-c["actual_array_output_bytes"])/(b["actual_array_output_bytes"]+fee_bytes/6)
            checks = {"all_original_full_science_extra_numeric_exact": science["all_protected"],
                      "each_whole_within1_1": c["complete_parent_window_seconds"] <= 1.1*b["complete_parent_window_seconds"],
                      "current_known_net_upper_at_least10percent": net_upper >= .1}
            pair = {"trial": trial, "order": order, "checks": checks,
                    "whole_seconds": {mode: row["complete_parent_window_seconds"] for mode, row in rows.items()},
                    "actual_array_output_bytes": {mode: row["actual_array_output_bytes"] for mode, row in rows.items()},
                    "current_net_upper_not_final_acceptance": net_upper, "compare_seconds": science["complete_compare_seconds"]}
            state["pairs"].append(pair)
            atomic_json(campaign / "STATE.json", state)
            print(json.dumps(pair), flush=True)
            if not all(checks.values()):
                raise ValueError("A frozen full science, each-whole or known-work upper guard failed")
        medians = {mode: statistics.median(pair["whole_seconds"][mode] for pair in state["pairs"]) for mode in ("raw", "native")}
        state.update(status="three_complete_pairs_pending_actual_faults_recovery_and_root",
                     raw_whole_medians_seconds=medians, raw_whole_median_within1_03=medians["native"] <= 1.03*medians["raw"],
                     complete_campaign_seconds=time.perf_counter()-started)
        if not state["raw_whole_median_within1_03"]:
            state["status"] = "negative0_complete_whole_median_guard"
        atomic_json(campaign / "RESULT.json", state)
    except BaseException as error:
        state.update(status="failed_verification0", error=repr(error), complete_campaign_seconds=time.perf_counter()-started)
        atomic_json(campaign / "STATE.json", state)
        raise
    finally:
        atomic_json(campaign / "STATE.json", state)


if __name__ == "__main__":
    main()
