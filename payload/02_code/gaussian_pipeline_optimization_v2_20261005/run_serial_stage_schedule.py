"""Run an explicit isolated schedule only after an owned predecessor is terminal."""

import argparse
import json
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
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read(args.configuration)
    args.output.mkdir(parents=True, exist_ok=False)
    identity = data_identity(args.configuration)
    state = {"status": "waiting_for_owned_predecessor", "configuration": identity, "runs": [], "counted": False}
    atomic_json(args.output/"STATE.json", state)
    started = time.monotonic()
    try:
        while True:
            source = read(ROOT/config["predecessor"])
            if source["status"] == config["required_status"]:
                if len(source["seed_checks"]) != 3 or not all(p["all24_quality_guards_passed"] for p in source["seed_checks"]):
                    raise ValueError("The predecessor is missing an original quality denominator")
                break
            if source["status"].startswith("failed") or time.monotonic()-started > config["maximum_wait_seconds"]:
                raise RuntimeError("The owned predecessor failed or exceeded the declared schedule wait")
            time.sleep(5)
        state["predecessor_receipt"] = data_identity(ROOT/config["predecessor"])
        for action in config["actions"]:
            if identity != data_identity(args.configuration):
                raise ValueError("The explicit serial stage configuration changed")
            label = action["name"]
            command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(args.output/(label+"_process.json")),
                "--log", str(args.output/(label+".log")), "--maximum-seconds", str(action["maximum_seconds"]),
                "--maximum-rss-gib", str(action["maximum_RSS_GiB"]), "--", *action["command"]]
            with (args.output/(label+"_controller.log")).open("x", encoding="utf-8") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
                row = {"name": label, "pid": process.pid, "status": "running", "command": command}
                state["runs"].append(row)
                state["status"] = "serial_actions_running"
                atomic_json(args.output/"STATE.json", state)
                code = process.wait()
            row.update(returncode=code, status="completed" if code == 0 else "failed")
            atomic_json(args.output/"STATE.json", state)
            if code:
                raise RuntimeError("An explicit serial stage action failed")
        state.update(status="serial_actions_terminal_acceptance_separate", elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
