"""Hold bounded CUDA buffers while an owned host compares memory interfaces."""

import argparse
import json
import os
from pathlib import Path
import struct
import sys
import threading
import time

import psutil
import torch

HEADER = struct.Struct("<I")


def save(path, value):
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    os.replace(temporary, path)


def exact(stream, count):
    result = bytearray()
    while len(result) < count:
        data = stream.read(count-len(result))
        if not data:
            raise EOFError("The owned memory-probe request ended early")
        result.extend(data)
    return result


def send(value):
    data = json.dumps(dict(value, payload_bytes=0), separators=(",", ":")).encode()
    sys.stdout.buffer.write(HEADER.pack(len(data))+data)
    sys.stdout.buffer.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource", type=Path, required=True)
    args = parser.parse_args()
    if torch.__version__ != "2.7.1+cu128" or not torch.cuda.is_available():
        raise RuntimeError("The existing CUDA runtime is required")
    torch.set_num_threads(4)
    torch.cuda.init()
    process, stopped = psutil.Process(), threading.Event()
    state = {"scope": "WSL", "pid": process.pid, "creation_time": process.create_time(), "peak_RSS_bytes": 0,
        "closed": False, "counted": False}
    began = time.monotonic()
    def observe():
        while not stopped.wait(.1):
            state["peak_RSS_bytes"] = max(state["peak_RSS_bytes"], process.memory_info().rss)
            if state["peak_RSS_bytes"] > 32*1024**3 or time.monotonic()-began > 180:
                os._exit(70)
    watcher = threading.Thread(target=observe, name="owned-memory-buffer-rss", daemon=True)
    watcher.start()
    save(args.resource, state)
    held = None
    try:
        stream = sys.stdin.buffer
        while header := stream.read(HEADER.size):
            if len(header) != HEADER.size:
                raise EOFError("The owned memory-probe header was truncated")
            length = HEADER.unpack(header)[0]
            if not 0 < length <= 4096:
                raise ValueError("A memory-probe request exceeds its metadata bound")
            request = json.loads(exact(stream, length))
            command = request["command"]
            if command == "info":
                send({"status": "ready", "torch": torch.__version__})
            elif command == "allocate":
                size = request["bytes"]
                if type(size) is not int or not 0 <= size <= 512*1024**2 or held is not None:
                    raise ValueError("Only one bounded private GPU buffer is admitted")
                held = torch.zeros(size, dtype=torch.uint8, device="cuda")
                torch.cuda.synchronize()
                send({"status": "held", "bytes": size, "CUDA_reserved_bytes": torch.cuda.memory_reserved()})
            elif command == "release":
                held = None
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                send({"status": "released"})
            elif command == "stop":
                held = None
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                send({"status": "stopped", "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved()})
                return
            else:
                raise ValueError("An unregistered memory-probe request was received")
    finally:
        stopped.set()
        watcher.join()
        state.update(closed=True, peak_RSS_bytes=max(state["peak_RSS_bytes"], process.memory_info().rss),
            peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved())
        save(args.resource, state)


if __name__ == "__main__":
    main()
