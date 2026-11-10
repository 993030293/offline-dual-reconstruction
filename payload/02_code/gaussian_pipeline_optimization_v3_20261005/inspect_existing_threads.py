"""Read explicitly authorized threads through the installed official RPC API."""

import argparse
import json
import queue
import subprocess
import threading
import time
from pathlib import Path


THREADS = (
    "01a10c61-9180-7b33-b15d-d414c526b640",
    "01a10c61-c330-7502-b21e-2e0c1799e03a",
)
ROOT = Path(__file__).resolve().parents[2]
CODEX = Path(
    "C:/Users/99303/scoop/apps/nodejs/current/bin/node_modules/@openai/codex/"
    "node_modules/@openai/codex-win32-x64/vendor/x86_64-pc-windows-msvc/bin/codex.exe"
)


class ReadClient:
    def __init__(self, codex_binary=CODEX):
        self.messages = queue.Queue()
        self.sequence = 0
        self.process = subprocess.Popen(
            [str(codex_binary), "app-server", "--listen", "stdio://"],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.messages.put(json.loads(line))
        except Exception as error:
            self.messages.put(error)
        finally:
            self.messages.put(None)

    def send(self, message):
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def request(self, method, params, timeout=20):
        allowed = {
            "initialize", "thread/read", "thread/goal/get", "thread/turns/list",
        }
        if method not in allowed:
            raise ValueError("This client permits read-only methods only")
        if method != "initialize" and params.get("threadId") not in THREADS:
            raise ValueError("Thread is outside the explicitly authorized pair")
        self.sequence += 1
        request_id = self.sequence
        self.send({"id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("RPC response deadline exceeded: " + method)
            message = self.messages.get(timeout=remaining)
            if message is None:
                raise RuntimeError("Owned RPC server closed before its response")
            if isinstance(message, Exception):
                raise message
            if message.get("id") == request_id:
                return message
            if "id" in message and "method" in message:
                raise RuntimeError("Unexpected server request in read-only inspection")

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)
        self.reader.join(timeout=5)
        self.process.stdout.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--codex-binary", type=Path, default=CODEX)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "04_outputs")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    client = ReadClient(codex_binary=args.codex_binary)
    result = {"read_only": True, "model_turns_started": 0, "threads": []}
    try:
        initialized = client.request("initialize", {
            "clientInfo": {
                "name": "gaussian_optimization_coordinator",
                "title": "Gaussian optimization read-only coordination",
                "version": "0.1.0",
            },
            "capabilities": {"experimentalApi": True},
        })
        result["initialize"] = initialized
        if "error" in initialized:
            raise RuntimeError("Official RPC initialization rejected")
        client.send({"method": "initialized", "params": {}})
        for thread_id in THREADS:
            record = {"thread_id": thread_id}
            record["thread_read"] = client.request("thread/read", {
                "threadId": thread_id, "includeTurns": False,
            })
            record["goal_get"] = client.request("thread/goal/get", {
                "threadId": thread_id,
            })
            record["latest_turns"] = client.request("thread/turns/list", {
                "threadId": thread_id, "limit": 3,
                "sortDirection": "desc", "itemsView": "full",
            })
            result["threads"].append(record)
    finally:
        client.close()
        result["owned_rpc_server_closed"] = client.process.poll() is not None
        result["elapsed_seconds"] = time.perf_counter() - started
        (output / "RESULT.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    summary = []
    for record in result["threads"]:
        thread = record["thread_read"].get("result", {}).get("thread", {})
        turns = record["latest_turns"].get("result", {}).get("data", [])
        summary.append({
            "thread_id": record["thread_id"], "cwd": thread.get("cwd"),
            "status": thread.get("status"), "name": thread.get("name"),
            "read_error": record["thread_read"].get("error"),
            "goal": record["goal_get"],
            "turns_error": record["latest_turns"].get("error"),
            "turns": [{"id": turn.get("id"), "status": turn.get("status")}
                      for turn in turns],
        })
    print(json.dumps({"output": str(output), "summary": summary},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
