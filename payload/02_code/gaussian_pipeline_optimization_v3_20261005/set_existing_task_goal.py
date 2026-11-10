"""Set one user-authorized task goal without resuming a desktop-owned writer."""

import argparse
import json
import time
from pathlib import Path

from inspect_existing_threads import ReadClient, ROOT, THREADS


class GoalClient(ReadClient):
    def set_goal(self, thread_id, objective):
        if thread_id not in THREADS or not 0 < len(objective) <= 4000:
            raise ValueError("Unauthorized task goal")
        self.sequence += 1
        request_id = self.sequence
        self.send({"id": request_id, "method": "thread/goal/set", "params": {
            "threadId": thread_id, "objective": objective, "status": "active",
        }})
        deadline = time.monotonic() + 30
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Goal-set outcome is unknown; do not repeat automatically")
            message = self.messages.get(timeout=remaining)
            if message is None:
                raise RuntimeError("Official goal server closed before the response")
            if isinstance(message, Exception):
                raise message
            if message.get("id") == request_id:
                return message
            if "id" in message and "method" in message:
                raise RuntimeError("Unexpected host request during goal-only update")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-inspection", required=True, type=Path)
    parser.add_argument("--branch", required=True, choices=("a", "b"))
    parser.add_argument("--codex-binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    inspection = json.loads(args.expected_inspection.read_text(encoding="utf-8"))
    assignment = next(x for x in manifest["assignments"] if x["task_id"] == "batch_" + args.branch + "_001")
    thread_id = assignment["thread_id"]
    expected = next(x for x in inspection["threads"] if x["thread_id"] == thread_id)["goal_get"]["result"]["goal"]
    output = args.output.resolve()
    output.relative_to(ROOT / "04_outputs")
    output.mkdir(parents=True, exist_ok=False)
    result = {"task_id": assignment["task_id"], "thread_id": thread_id,
              "turn_start_requests": 0, "thread_resume_requests": 0,
              "parent_goal_changed": False, "expected_previous_goal": expected}
    client = GoalClient(codex_binary=args.codex_binary)
    try:
        initialized = client.request("initialize", {
            "clientInfo": {"name": "gaussian_optimization_coordinator",
                           "title": "Authorized task goal update", "version": "0.1.0"},
            "capabilities": {"experimentalApi": True},
        })
        if "error" in initialized:
            raise RuntimeError("Official initialization was rejected")
        client.send({"method": "initialized", "params": {}})
        before = client.request("thread/goal/get", {"threadId": thread_id})
        if "error" in before:
            raise RuntimeError("Official goal read was rejected")
        current = before["result"]["goal"]
        result["before"] = current
        if (current or {}).get("objective") == assignment["objective"]:
            result["status"] = "already_expected_task_goal"
        else:
            if (current or {}).get("objective") != (expected or {}).get("objective") or (current or {}).get("status") != (expected or {}).get("status"):
                raise RuntimeError("Goal changed since inspection; no new objective was sent")
            response = client.set_goal(thread_id, assignment["objective"])
            result["goal_set_response"] = response
            if "error" in response:
                result["status"] = "official_goal_set_rejected"
            else:
                after = client.request("thread/goal/get", {"threadId": thread_id})
                result["after"] = after
                goal = after.get("result", {}).get("goal") or {}
                exact = goal.get("status") == "active" and goal.get("objective") == assignment["objective"]
                result["status"] = "task_goal_active_verified" if exact else "goal_set_readback_failed"
    finally:
        client.close()
        result["owned_rpc_server_closed"] = client.process.poll() is not None
        (output / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] not in ("already_expected_task_goal", "task_goal_active_verified"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
