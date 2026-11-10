"""Run isolated candidate checks sequentially after a verified native prerequisite ends."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "actions", "prerequisite-process", "prerequisite-result"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    actions = read(args.actions)
    if not isinstance(actions, list) or not actions or len({r["name"] for r in actions}) != len(actions):
        raise ValueError("A nonempty unique candidate stage sequence is required")
    frozen_actions = data_identity(args.actions)
    state = {"status": "waiting_for_prerequisite", "pid": os.getpid(), "source": data_identity(Path(__file__)),
        "actions": actions, "action_source": frozen_actions, "runs": [], "counted": False,
        "maximum_group_seconds": 3800, "prerequisite_wait_cap_seconds": 900,
        "prerequisite_process": str(args.prerequisite_process), "prerequisite_result": str(args.prerequisite_result),
        "sequential_heavy_work": True, "no_automatic_model_training_or_new_effective_count": True}
    atomic_json(args.output/"PROTOCOL.json", state)
    atomic_json(args.output/"STATE.json", state)
    started = time.monotonic()
    try:
        while True:
            if time.monotonic()-started > 900:
                raise TimeoutError("The native prerequisite wait cap was reached; no candidate started")
            if args.prerequisite_process.exists():
                receipt = read(args.prerequisite_process)
                if receipt["status"] == "completed":
                    break
                if receipt["status"] != "running":
                    raise RuntimeError("The native prerequisite terminated without its complete receipt")
            time.sleep(5)
        prerequisite = read(args.prerequisite_result)
        if (prerequisite["status"] != "three_recovery_pairs_passed_metric_training_quality_pending" or
                len(prerequisite["pairs"]) != 3 or not all(r["all_geometry_guards"] for r in prerequisite["pairs"])):
            raise ValueError("The prerequisite did not preserve its complete recovery guards")
        state.update(status="running_sequential_candidate_stages", prerequisite_status=prerequisite["status"])
        atomic_json(args.output/"STATE.json", state)
        for action in actions:
            if time.monotonic()-started > 3800 or frozen_actions != data_identity(args.actions):
                raise ValueError("The candidate queue exceeded its group cap or its frozen sequence changed")
            output_name = action["name"]
            if not output_name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in output_name):
                raise ValueError("A stage receipt name is outside the independent output namespace")
            command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(args.output/(output_name+"_process.json")),
                "--log", str(args.output/(output_name+".log")), "--maximum-seconds", str(action["maximum_seconds"]),
                "--maximum-rss-gib", str(action["maximum_RSS_GiB"]), "--", sys.executable, *action["arguments"]]
            row = {"name": output_name, "status": "running", "command": command}
            state["runs"].append(row)
            with (args.output/(output_name+"_controller.log")).open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                row["pid"] = process.pid
                atomic_json(args.output/"STATE.json", state)
                code = process.wait()
            row.update(status="completed" if code == 0 else "failed", returncode=code)
            atomic_json(args.output/"STATE.json", state)
            if code:
                raise RuntimeError("A candidate stage failed; later stages were not started: "+output_name)
            if action.get("result_path"):
                result = read(ROOT/action["result_path"])
                row["actual_result_status"] = result["status"]
                if action.get("required_status") and result["status"] != action["required_status"]:
                    raise ValueError("A candidate stage did not satisfy its declared prerequisite: "+output_name)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps({"stage": output_name, "status": row["status"], "result": row.get("actual_result_status")}), flush=True)
        state.update(status="candidate_stages_completed_individual_acceptance_and_combination_pending",
            elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
