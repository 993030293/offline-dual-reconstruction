"""Serve the versioned preview locally and record process ownership."""

import argparse
import json
import os
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--port", type=int, default=8195)
args = parser.parse_args()
root = args.root.resolve(strict=True)
if not (root / "DISPLAY_EXPORT_RECEIPT.json").is_file():
    raise ValueError("The directory is not a versioned display export")
server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(SimpleHTTPRequestHandler, directory=str(root)))
state = {"status": "serving", "pid": os.getpid(), "root": str(root), "port": args.port,
         "url": f"http://127.0.0.1:{args.port}/", "started_unix": time.time()}
path = root / f"SERVER_STATE_{args.port}.json"
path.write_text(json.dumps(state, indent=2), encoding="utf-8")
print(json.dumps(state), flush=True)
try:
    server.serve_forever()
finally:
    server.server_close()
    state.update(status="stopped", stopped_unix=time.time())
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
