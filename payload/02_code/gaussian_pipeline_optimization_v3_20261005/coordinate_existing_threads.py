"""Run two explicitly authorized, isolated tasks through official Codex RPC."""

import argparse
import datetime
import json
import queue
import time
import uuid
from pathlib import Path

import psutil

from inspect_existing_threads import ReadClient, ROOT, THREADS
from run_with_resource_lease import save


DESKTOP_CODEX = Path("C:/Users/99303/AppData/Local/OpenAI/Codex/bin/ca9abb0b4d8ac692/codex.exe")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class ControlClient(ReadClient):
    def __init__(self, output, state):
        self.output = output
        self.state = state
        self.event_file = (output / "EVENTS.jsonl").open("a", encoding="utf-8")
        super().__init__(codex_binary=DESKTOP_CODEX)
        self.state["codex_binary"] = str(DESKTOP_CODEX)
        self.state["codex_version"] = "0.159.0"
        self.state["global_config_modified"] = False
        self.state["controller_server_pid"] = self.process.pid
        self.state["controller_server_creation_time"] = psutil.Process(self.process.pid).create_time()
        self.persist()

    def persist(self):
        self.state["updated_utc"] = utc()
        save(self.output / "STATE.json", self.state)

    def event(self, kind, **values):
        row = {"utc": utc(), "kind": kind, **values}
        self.event_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.event_file.flush()

    def handle(self, message):
        if message is None:
            raise RuntimeError("Owned official RPC server closed")
        if isinstance(message, Exception):
            raise message
        method = message.get("method")
        params = message.get("params", {})
        if "id" in message and method:
            # Unknown approvals and host tools cannot be converted to success.
            self.event("server_request_needs_host_support", method=method,
                       request_id=message["id"], thread_id=params.get("threadId"),
                       tool=params.get("tool"))
            self.state["host_request_attention"] = True
            self.persist()
            if method == "item/tool/call":
                self.send({"id": message["id"], "error": {
                    "code": -32601,
                    "message": "This local coordination client has no dynamic host tool handler",
                }})
            return
        thread_id = params.get("threadId")
        if thread_id not in THREADS:
            return
        record = self.state["threads"][thread_id]
        if method in ("turn/started", "turn/completed"):
            turn = params.get("turn", {})
            record["latest_turn_id"] = turn.get("id")
            record["latest_turn_status"] = turn.get("status")
            if turn.get("error"):
                record["last_turn_error"] = turn["error"]
            self.event(method, thread_id=thread_id, turn_id=turn.get("id"),
                       status=turn.get("status"), error=turn.get("error"))
            self.persist()
        elif method == "thread/goal/updated":
            record["goal"] = params.get("goal")
            self.event(method, thread_id=thread_id, goal=record["goal"])
            self.persist()
        elif method == "thread/status/changed":
            record["runtime_status"] = params.get("status")
            self.event(method, thread_id=thread_id, status=record["runtime_status"])
            self.persist()
        elif method == "item/completed":
            item = params.get("item", {})
            if item.get("type") == "agentMessage":
                self.event("agent_message", thread_id=thread_id,
                           turn_id=params.get("turnId"), phase=item.get("phase"),
                           text=item.get("text", ""))
                record["last_agent_message_utc"] = utc()
                self.persist()

    def request(self, method, params, timeout=90):
        allowed = {"initialize", "thread/read", "thread/turns/list", "thread/goal/get",
                   "thread/resume", "turn/start", "thread/goal/set"}
        if method not in allowed:
            raise ValueError("Unapproved RPC method")
        if method != "initialize" and params.get("threadId") not in THREADS:
            raise ValueError("Thread is outside the explicitly authorized pair")
        self.sequence += 1
        request_id = self.sequence
        self.event("rpc_sent", request_id=request_id, method=method,
                   thread_id=params.get("threadId"))
        self.send({"id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.event("rpc_outcome_unknown_do_not_resend", request_id=request_id, method=method)
                raise TimeoutError("Official RPC response deadline exceeded: " + method)
            try:
                message = self.messages.get(timeout=min(15, remaining))
            except queue.Empty:
                self.persist()
                continue
            if isinstance(message, dict) and message.get("id") == request_id and "method" not in message:
                self.event("rpc_returned", request_id=request_id, method=method,
                           thread_id=params.get("threadId"), error=message.get("error"))
                if "error" in message:
                    raise RuntimeError("Official RPC rejected " + method + ": " +
                                       json.dumps(message["error"], ensure_ascii=False))
                return message["result"]
            self.handle(message)

    def close(self):
        super().close()
        self.event_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prior-inspection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    prior = json.loads(args.prior_inspection.read_text(encoding="utf-8"))
    assignments = manifest["assignments"]
    if {item["thread_id"] for item in assignments} != set(THREADS) or len(assignments) != 2:
        raise ValueError("Manifest must identify the explicitly authorized two existing chats")
    output = args.output.resolve()
    output.relative_to(ROOT / "04_outputs")
    output.mkdir(parents=True, exist_ok=False)
    prior_by_id = {item["thread_id"]: item for item in prior["threads"]}
    state = {
        "status": "preparing", "created_utc": utc(), "controller_pid": psutil.Process().pid,
        "controller_creation_time": psutil.Process().create_time(),
        "native_goals_required": True, "new_threads_created": 0,
        "parent_goal_changed": False, "counted_optimizations": 0, "threads": {},
    }
    for assignment in assignments:
        state["threads"][assignment["thread_id"]] = {
            "task_id": assignment["task_id"], "prompt": assignment["prompt"],
            "objective": assignment["objective"], "phase": "prepared",
        }
    client = ControlClient(output, state)
    running = False
    try:
        client.request("initialize", {
            "clientInfo": {"name": "gaussian_optimization_coordinator",
                           "title": "Authorized Gaussian optimization parallel tasks", "version": "0.1.0"},
            "capabilities": {"experimentalApi": True},
        })
        client.send({"method": "initialized", "params": {}})
        for assignment in assignments:
            thread_id = assignment["thread_id"]
            record = state["threads"][thread_id]
            thread = client.request("thread/read", {"threadId": thread_id, "includeTurns": False})["thread"]
            latest = client.request("thread/turns/list", {
                "threadId": thread_id, "limit": 1, "itemsView": "notLoaded", "sortDirection": "desc",
            })["data"]
            goal = client.request("thread/goal/get", {"threadId": thread_id})["goal"]
            old_thread = prior_by_id[thread_id]["thread_read"]["result"]["thread"]
            old_turns = prior_by_id[thread_id]["latest_turns"]["result"]["data"]
            if Path(thread["cwd"]).resolve() != ROOT:
                raise ValueError("Existing chat workspace changed")
            if goal is not None or not latest or latest[0]["status"] == "inProgress":
                raise RuntimeError("Another goal or active task now owns the existing chat")
            if latest[0]["id"] != old_turns[0]["id"] or thread["updatedAt"] != old_thread["updatedAt"]:
                raise RuntimeError("Existing chat changed since inspection; no task was resent")
            record["pre_dispatch_latest_turn"] = latest[0]
            resumed = client.request("thread/resume", {"threadId": thread_id, "excludeTurns": True})
            if resumed["thread"]["id"] != thread_id or Path(resumed["cwd"]).resolve() != ROOT:
                raise ValueError("Official resume returned a different thread or workspace")
            record["resumed_model"] = resumed.get("model")
            record["resumed_approval_policy"] = resumed.get("approvalPolicy")
            record["phase"] = "resumed_not_started"
            client.persist()
        for assignment in assignments:
            thread_id = assignment["thread_id"]
            record = state["threads"][thread_id]
            prompt_path = (ROOT / assignment["prompt"]).resolve()
            prompt_path.relative_to(ROOT / "research")
            prompt = (
                "主聊天依据用户的明确并行授权派发本批任务，请立即实施并持续完成。"
                "这是已有聊天的任务，不创建新用户聊天。控制器将为此批设置原生goal。"
                "中央50项目标和11已接受计数保持独立；不能把本批候选否定闭环记成50项成功。\n\n"
                + prompt_path.read_text(encoding="utf-8")
            )
            record["phase"] = "turn_start_requested_do_not_duplicate"
            client.persist()
            running = True
            started = client.request("turn/start", {
                "threadId": thread_id,
                "clientUserMessageId": str(uuid.uuid5(uuid.NAMESPACE_URL, assignment["task_id"] + thread_id)),
                "input": [{"type": "text", "text": prompt}],
            })
            record["dispatched_turn_id"] = started["turn"]["id"]
            record["latest_turn_id"] = started["turn"]["id"]
            record["latest_turn_status"] = started["turn"]["status"]
            goal = client.request("thread/goal/set", {
                "threadId": thread_id, "objective": assignment["objective"], "status": "active",
            })["goal"]
            if not goal or goal["status"] != "active" or goal["objective"] != assignment["objective"]:
                raise RuntimeError("Native task goal was not actually activated")
            record["goal"] = goal
            record["phase"] = "dispatched_native_goal_active"
            client.persist()
            print(json.dumps({"thread_id": thread_id, "turn_id": record["dispatched_turn_id"],
                              "goal_status": goal["status"], "task_id": assignment["task_id"]}), flush=True)
        state["status"] = "running_two_native_goals"
        client.persist()
    except Exception as error:
        state["dispatch_error"] = {"type": type(error).__name__, "message": str(error)}
        state["status"] = "dispatch_needs_attention_no_automatic_resend"
        client.persist()
        print(json.dumps({"status": state["status"], "error": state["dispatch_error"]}), flush=True)
        if not running:
            client.close()
            state["owned_rpc_server_closed"] = True
            save(output / "STATE.json", state)
            raise
    last_probe = time.monotonic()
    try:
        while True:
            try:
                client.handle(client.messages.get(timeout=10))
            except queue.Empty:
                client.persist()
            if time.monotonic() - last_probe >= 30:
                for assignment in assignments:
                    thread_id = assignment["thread_id"]
                    record = state["threads"][thread_id]
                    record["goal"] = client.request("thread/goal/get", {"threadId": thread_id})["goal"]
                    thread = client.request("thread/read", {"threadId": thread_id, "includeTurns": False})["thread"]
                    record["runtime_status"] = thread["status"]
                client.persist()
                last_probe = time.monotonic()
                done = all((record.get("goal") or {}).get("status") == "complete" and
                           record.get("runtime_status", {}).get("type") == "idle"
                           for record in state["threads"].values())
                if done:
                    state["status"] = "two_task_goals_complete_review_required"
                    client.persist()
                    break
    finally:
        client.close()
        state["owned_rpc_server_closed"] = True
        state["closed_utc"] = utc()
        save(output / "STATE.json", state)


if __name__ == "__main__":
    main()
