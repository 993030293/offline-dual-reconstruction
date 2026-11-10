"""Read the reviewer child explicitly recorded by the authorized B chat."""

import argparse
import json
from pathlib import Path
import time

from inspect_existing_threads import ReadClient, ROOT

B_THREAD = "01a10c61-c330-7502-b21e-2e0c1799e03a"


class ChildReadClient(ReadClient):
    def __init__(self, child_thread_id, **kwargs):
        self.child_thread_id = child_thread_id
        super().__init__(**kwargs)

    def request(self, method, params, timeout=20):
        if method == "initialize":
            return super().request(method, params, timeout)
        if method not in ("thread/read", "thread/turns/list") or params.get("threadId") != self.child_thread_id:
            raise ValueError("Only the snapshot-proven B reviewer child can be read")
        self.sequence += 1
        request_id = self.sequence
        self.send({"id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Reviewer provenance read timed out")
            message = self.messages.get(timeout=remaining)
            if message is None:
                raise RuntimeError("Owned reviewer read server closed before response")
            if isinstance(message, Exception):
                raise message
            if message.get("id") == request_id:
                return message
            if "id" in message and "method" in message:
                raise RuntimeError("Unexpected host request during child provenance read")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-snapshot", required=True, type=Path)
    parser.add_argument("--codex-binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    snapshot = json.loads(args.parent_snapshot.read_text(encoding="utf-8"))
    record = next(row for row in snapshot["threads"] if row["thread_id"] == B_THREAD)
    turns = record["latest_turns"]["result"]["data"]
    activities = [item for turn in turns for item in turn["items"]
                  if item["type"] == "subAgentActivity" and item.get("agentPath") == "/root/completion_monitor"]
    reviewers = sorted({item["agentThreadId"] for item in activities})
    if len(reviewers) != 1 or not any(item["kind"] == "started" for item in activities):
        raise ValueError("An unambiguous real B-local reviewer was not recorded")
    output = args.output.resolve()
    output.relative_to(ROOT / "04_outputs")
    output.mkdir(parents=True, exist_ok=False)
    reviewer = reviewers[0]
    result = {"read_only": True, "parent_thread_id": B_THREAD, "reviewer_thread_id": reviewer,
              "agent_path_is_thread_scoped": True, "parent_activity": activities,
              "turn_start_requests": 0, "thread_resume_requests": 0, "goal_set_requests": 0}
    client = ChildReadClient(reviewer, codex_binary=args.codex_binary)
    try:
        reply = client.request("initialize", {"clientInfo": {"name": "review_provenance_reader", "version": "0.1.0"},
                                              "capabilities": {"experimentalApi": True}})
        if "error" in reply:
            raise RuntimeError("The official read service rejected initialization")
        client.send({"method": "initialized", "params": {}})
        result["thread_read"] = client.request("thread/read", {"threadId": reviewer, "includeTurns": False})
        result["latest_turns"] = client.request("thread/turns/list", {"threadId": reviewer, "limit": 1,
                                                                      "sortDirection": "desc", "itemsView": "full"})
    finally:
        client.close()
        result["owned_rpc_server_closed"] = client.process.poll() is not None
        (output / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    data = result["latest_turns"].get("result", {}).get("data", [])
    writes = [change["path"] for turn in data for item in turn["items"] if item["type"] == "fileChange"
              for change in item["changes"]]
    print(json.dumps({"output": str(output), "reviewer_thread_id": reviewer,
        "read_error": result["thread_read"].get("error"), "turns_error": result["latest_turns"].get("error"),
        "recorded_file_change_paths": writes, "owned_rpc_server_closed": result["owned_rpc_server_closed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
