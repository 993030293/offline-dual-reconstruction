"""Fetch the official checksum-bound native matcher into an isolated directory."""

import argparse
from pathlib import Path
import time
import urllib.request

from runtime import atomic_json, data_identity


MODEL_URL = "https://github.com/colmap/colmap/releases/download/3.13.0/sift-lightglue.onnx"
MODEL_SHA = "e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = [("sift-lightglue.onnx", MODEL_URL),
        ("COLMAP_LICENSE.txt", "https://raw.githubusercontent.com/colmap/colmap/4.1.1/COPYING.txt"),
        ("LightGlue_LICENSE.txt", "https://raw.githubusercontent.com/cvg/LightGlue/eb42fee2d71449efb0aa5c10549752b5d75384d8/LICENSE")]
    atomic_json(args.output/"CONTRACT.json", {"files": files, "expected_model_sha256": MODEL_SHA,
        "maximum_bytes": 1024**3, "maximum_seconds": 300,
        "source": data_identity(Path(__file__)), "installation": False, "model_execution": False})
    started = time.monotonic()
    total = 0
    receipts = []
    try:
        for name, url in files:
            path = args.output/name
            with urllib.request.urlopen(url, timeout=60) as response, path.open("xb") as handle:
                while True:
                    chunk = response.read(1024**2)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 1024**3 or time.monotonic()-started > 300:
                        raise RuntimeError("Isolated download exceeded its frozen resource budget")
                    handle.write(chunk)
            identity = data_identity(path)
            if name == "sift-lightglue.onnx" and identity["sha256"] != MODEL_SHA:
                raise ValueError("Model differs from the checksum embedded in the installed COLMAP CLI")
            if name.endswith("LICENSE.txt") and path.stat().st_size < 100:
                raise ValueError("License download is incomplete")
            receipts.append(dict(identity, url=url))
        atomic_json(args.output/"RESULT.json", {"status": "official_model_verified_not_executed", "files": receipts,
            "download_seconds": time.monotonic()-started, "bytes": total, "counted": False})
        print("Official native LightGlue model and licenses verified; inference remains untested")
    except Exception as error:
        atomic_json(args.output/"FAILURE.json", {"error": repr(error), "completed": receipts,
            "download_seconds": time.monotonic()-started, "bytes": total, "partial_files_preserved": True})
        raise


if __name__ == "__main__":
    main()
