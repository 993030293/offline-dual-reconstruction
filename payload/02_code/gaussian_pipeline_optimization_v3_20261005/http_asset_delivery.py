"""Serve held immutable native assets with explicit gzip negotiation on loopback."""

import argparse
import gzip
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
from urllib.parse import unquote, urlsplit

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from build_http_asset_packages import byte_equal
from locked_input_identity import LockedInputs


def encoding_qualities(value):
    choices = {}
    for part in (value or "").split(","):
        parameters = part.strip().lower().split(";")
        name = parameters[0].strip()
        quality = 1.
        for parameter in parameters[1:]:
            if parameter.strip().startswith("q="):
                try:
                    quality = float(parameter.strip()[2:])
                except ValueError:
                    quality = 0.
        choices[name] = quality if 0. <= quality <= 1. else 0.
    return choices


def accepts_gzip(value):
    choices = encoding_qualities(value)
    return choices.get("gzip", choices.get("*", 0.)) > 0.


def accepts_identity(value):
    choices = encoding_qualities(value)
    return choices.get("identity", 0. if choices.get("*") == 0. else 1.) > 0.


class DeliveryServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True

    def __init__(self, campaign, receipt):
        self.campaign, self.receipt = campaign, receipt
        self.routes, self.index, self.requests = {}, {}, []
        self.request_lock = threading.Lock()
        self.snapshot = LockedInputs([campaign], data_identity)
        try:
            for trial in range(3):
                for mode in ("identity", "gzip"):
                    prefix = f"/trial{trial}/{mode}/"
                    folder = campaign / f"trial{trial}" / mode
                    catalog = json.loads((folder / "FILE_INDEX.json").read_text(encoding="utf-8"))
                    self.routes[prefix] = folder
                    self.index[prefix] = {row["relative"]: row for row in catalog["files"]}
                    for row in catalog["files"]:
                        actual = self.snapshot.identity(folder / row["served_file"])
                        if actual != row["encoded_identity"]:
                            raise ValueError("A served full release identity differs from the completed package")
                        if row["encoding"] == "gzip":
                            original = campaign / f"trial{trial}" / "identity" / row["relative"]
                            with gzip.open(folder / row["served_file"], "rb") as expanded, original.open("rb") as native:
                                if not byte_equal(expanded, native):
                                    raise ValueError("A served compressed asset does not expand to the original complete native bytes")
            super().__init__(("127.0.0.1", 0), DeliveryHandler)
        except BaseException:
            self.snapshot.close()
            raise

    def resolve(self, request):
        path = unquote(urlsplit(request).path)
        prefix = next((p for p in self.routes if path.startswith(p)), None)
        if prefix is None:
            return None
        relative = path[len(prefix):]
        if any(x in (".", "..") for x in relative.split("/")) or "\\" in relative:
            return None
        folder = self.routes[prefix]
        actual = (folder / relative).resolve()
        if not actual.is_relative_to(folder):
            return None
        return prefix, relative, actual

    def server_close(self):
        try:
            super().server_close()
            self.snapshot.verify()
        finally:
            self.snapshot.close()


class DeliveryHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def translate_path(self, path):
        resolved = self.server.resolve(path)
        return str(resolved[2]) if resolved else str(self.server.campaign / "__unsupported_route__")

    def send_head(self):
        resolved = self.server.resolve(self.path)
        if resolved is None:
            self.send_error(404, "Unknown owned route")
            return None
        prefix, relative, original_path = resolved
        row = self.server.index[prefix].get(relative)
        encoded = row and row["encoding"] == "gzip" and accepts_gzip(self.headers.get("Accept-Encoding"))
        if not encoded and not accepts_identity(self.headers.get("Accept-Encoding")):
            self.send_error(406, "No acceptable owned representation")
            return None
        served = self.server.routes[prefix] / row["served_file"] if encoded else original_path
        if row and row["encoding"] == "gzip" and not encoded:
            served = self.server.campaign / prefix.split("/")[1] / "identity" / relative
        if row is None or not served.is_file():
            return super().send_head()
        expected = row["encoded_identity"]
        if row["encoding"] == "gzip" and not encoded:
            expected = self.server.index[prefix.replace("/gzip/", "/identity/")][relative]["encoded_identity"]
        if self.server.snapshot.identity(served) != expected:
            raise ValueError("A held served source changed before consumption")
        stream = served.open("rb")
        try:
            stat = os.fstat(stream.fileno())
            self.send_response(200)
            self.send_header("Content-type", self.guess_type(str(original_path)))
            self.send_header("Content-Length", str(stat.st_size))
            self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
            self.send_header("Vary", "Accept-Encoding")
            if encoded:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            with self.server.request_lock:
                self.server.requests.append({"path": self.path, "method": self.command,
                    "encoding": "gzip" if encoded else "identity", "body_bytes_declared": stat.st_size,
                    "accept_encoding": self.headers.get("Accept-Encoding", ""), "relative": relative})
            return stream
        except BaseException:
            stream.close()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    campaign.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005")
    if args.receipt.exists():
        raise FileExistsError("A fresh server receipt is required")
    started = time.perf_counter()
    server = DeliveryServer(campaign, args.receipt)
    pid = os.getpid()
    record = {"status": "serving", "pid": pid, "creation_time": psutil.Process(pid).create_time(),
        "port": server.server_port, "url": f"http://127.0.0.1:{server.server_port}",
        "startup_full_validation_seconds": time.perf_counter() - started,
        "held_source_snapshot": server.snapshot.receipt(), "requests": [], "counted": False}
    atomic_json(args.receipt, record)
    print(json.dumps(record), flush=True)
    def owned_control():
        if sys.stdin.readline().strip() == "stop":
            server.shutdown()

    control = threading.Thread(target=owned_control, name="owned-http-control", daemon=True)
    control.start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
        record.update(status="stopped", elapsed_seconds=time.perf_counter() - started,
            requests=server.requests, server_closed=True, held_read_handles_closed=True)
        atomic_json(args.receipt, record)


if __name__ == "__main__":
    main()
