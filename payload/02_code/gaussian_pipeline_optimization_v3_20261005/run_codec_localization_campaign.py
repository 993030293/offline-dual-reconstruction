"""Measure fresh full24/48 pairs without altering the incumbent science gates."""

import argparse
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
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup018/R170"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from run_sift_process_campaign_005 import compare_pair


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_node(folder, mode):
    tag = "b" if mode == "raw" else "c"
    node, output = folder / ("n" + tag), folder / tag
    command = [sys.executable, "-B", str(CODE / "codec_localization_node.py"), "--kind", "localization",
        "--mode", "process", "--node", str(node), "--output", str(output), "--", sys.executable, "-B",
        str(CODE / "localize_codec_requests.py"), "--wire-mode", mode, "--output", str(output)]
    started = time.perf_counter()
    with (folder / (mode + "_controller.log")).open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        identity = {"pid": process.pid, "creation_time": psutil.Process(process.pid).create_time()}
        atomic_json(folder / (mode + "_CONTROLLER.json"), {"command": command, "owner": identity, "status": "running"})
        code = process.wait(timeout=930)
    elapsed = time.perf_counter()-started
    receipt = read(node / "NODE.json")
    result = read(output / "RESULT.json") if (output / "RESULT.json").exists() else None
    row = {"mode": mode, "command": command, "owner": identity, "returncode": code,
           "complete_parent_window_seconds": elapsed, "node": str(node), "output": str(output)}
    atomic_json(folder / (mode + "_CONTROLLER_FINAL.json"), row)
    if (code or receipt["status"] != "completed" or receipt["live_Windows_after_wait"]
            or any(item["alive"] for item in receipt["actual_Linux_after_wait"])
            or not result or not all(result["checks"].values())):
        raise RuntimeError("An actual complete original-science wire-codec node failed")
    row["actual_written_input_bytes"] = result["actual_written_input_bytes"]
    return row


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("first", "remaining"), required=True)
    args = parser.parse_args()
    campaign = OUT / "c0"
    paths = [Path(__file__), *[CODE / name for name in ("localize_codec_requests.py",
        "codec_localization_node.py", "codec_owned_json_write.py", "codec_sift_matching.py",
        "sift_payload_input_worker.py", "bounded_zstd_payload.py", "codec_owned_matching_worker.py",
        "localize_geometry_sift_fusion.py", "run_sift_process_campaign_005.py", "sift_process_node_003.py",
        "run_bounded_journal_task.py", "sift_process_node_config.py", "check_codec_protocol_errors.py",
        "check_codec_sift_protocol.py", "check_codec_sift_protocol_v2.py")],
        ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005/compare_actual_localization.py"]
    sources = [data_identity(path) for path in paths]
    for name in ("g1/RESULT.json", "g2/RESULT.json"):
        component = read(OUT / name)
        if not all(component["checks"].values()):
            raise ValueError("All registered actual native protocol checks must pass first")
        if any(data_identity(Path(row["path"])) != row for row in component["sources"]):
            raise ValueError("An actual native component source changed before the full entry")
    if args.stage == "first":
        campaign.mkdir(parents=True, exist_ok=False)
        for number, path in enumerate(paths):
            shutil.copyfile(path, campaign / (f"SOURCE_{number:02d}_" + path.name))
        state = {"status": "running_first", "sources": sources, "pairs": [], "runs": [],
                 "orders": [["raw", "codec"], ["codec", "raw"], ["raw", "codec"]],
                 "origin_pipe_fee_budget_bytes": 96233216+50*4100,
                 "additional_unknown_pipe_fee_budget_bytes": 128*1024**2,
                 "maximum_each_whole_ratio": 1.1, "maximum_raw_and_charged_whole_median_ratio": 1.03,
                 "minimum_net_complete_pipe_reduction": .1, "no_speed_or_production_claim": True,
                 "new_optimization_count": 0}
        atomic_json(campaign / "PROTOCOL.json", state)
        indices = [0]
    else:
        state = read(campaign / "STATE.json")
        if state["sources"] != sources or len(state["pairs"]) != 1 or not all(state["pairs"][0]["checks"].values()):
            raise ValueError("The first full pair and all frozen sources must be intact")
        indices = [1, 2]
    try:
        for trial in indices:
            if time.perf_counter()-started > 850 or sources != [data_identity(path) for path in paths]:
                raise RuntimeError("The bounded frozen campaign changed")
            folder = campaign / ("t" + str(trial))
            folder.mkdir()
            runs = {}
            for mode in state["orders"][trial]:
                state.update(status="running", current_trial=trial, current_mode=mode)
                atomic_json(campaign / "STATE.json", state)
                row = run_node(folder, mode)
                runs[mode] = row
                state["runs"].append(row)
                atomic_json(campaign / "STATE.json", state)
            science = compare_pair(folder / "b", folder / "c")
            atomic_json(folder / "INVARIANTS.json", science)
            b, c = runs["raw"], runs["codec"]
            fee_bytes = (state["origin_pipe_fee_budget_bytes"]+state["additional_unknown_pipe_fee_budget_bytes"]
                + sum(read(OUT / name)["extra_actual_written_input_bytes"] for name in ("g1/RESULT.json", "g2/RESULT.json")))
            net_upper = (b["actual_written_input_bytes"]-c["actual_written_input_bytes"])/(b["actual_written_input_bytes"]+fee_bytes/6)
            checks = {"all_original_protected_science_including_extra_exact_poses": science["all_protected"],
                      "each_actual_complete_parent_whole_within1_1": c["complete_parent_window_seconds"] <= 1.1*b["complete_parent_window_seconds"],
                      "all_complete50_messages_actual_pipe_bytes_reduced": c["actual_written_input_bytes"] < b["actual_written_input_bytes"],
                      "known_current_net_upper_still_at_least10percent": net_upper >= .1}
            pair = {"trial": trial, "order": state["orders"][trial], "checks": checks,
                    "whole_seconds": {mode: row["complete_parent_window_seconds"] for mode, row in runs.items()},
                    "actual_pipe_bytes": {mode: row["actual_written_input_bytes"] for mode, row in runs.items()},
                    "known_current_extra_pipe_fee_bytes": fee_bytes, "current_net_upper_not_final": net_upper,
                    "complete_compare_seconds": science["complete_compare_seconds"]}
            state["pairs"].append(pair)
            atomic_json(campaign / "STATE.json", state)
            print(json.dumps(pair), flush=True)
            if not all(checks.values()):
                raise ValueError("An unchanged science, whole or necessary-work cost guard failed")
        state.update(status="first_complete_pair_passed_pending_rest" if len(state["pairs"]) == 1
                     else "three_complete_pairs_passed_pending_faults_recovery_and_review",
                     raw_whole_medians_seconds={mode: statistics.median(row["whole_seconds"][mode] for row in state["pairs"])
                                              for mode in ("raw", "codec")},
                     actual_pipe_medians_bytes={mode: statistics.median(row["actual_pipe_bytes"][mode] for row in state["pairs"])
                                                for mode in ("raw", "codec")},
                     complete_stage_seconds=time.perf_counter()-started)
        if len(state["pairs"]) == 3:
            medians = state["raw_whole_medians_seconds"]
            state["raw_whole_median_within_V7_1_03"] = medians["codec"] <= 1.03*medians["raw"]
            if not state["raw_whole_median_within_V7_1_03"]:
                state["status"] = "negative0_full_whole_median_guard"
        atomic_json(campaign / "RESULT.json", state)
    except BaseException as error:
        state.update(status="failed_verification0", error=repr(error), complete_stage_seconds=time.perf_counter()-started)
        atomic_json(campaign / "STATE.json", state)
        raise
    finally:
        atomic_json(campaign / "STATE.json", state)


if __name__ == "__main__":
    main()
