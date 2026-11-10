"""Serve fresh models and validate complete native packed display artifacts locally."""

import argparse
import hashlib
import html
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import struct
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity

BRANCHES = {case+"-"+mode for case in ("box2", "box1", "c1", "c0_july") for mode in ("baseline", "optimized")}
LIMIT = 256*1024**2
NAMES = ("packedArray", "sh1", "sh2", "sh3")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_asset(data, expected):
    if not 16 <= len(data) <= LIMIT or data[:8] != b"SPKPK001":
        raise ValueError("Invalid complete asset length or magic")
    header_length, start = struct.unpack_from("<II", data, 8)
    if header_length > 16384 or start != 16+(header_length+3)//4*4 or start > len(data):
        raise ValueError("Invalid packed header bounds")
    header = json.loads(data[16:16+header_length].decode("utf-8", errors="strict"))
    if header["schema"] != 1 or header["maxSh"] != 3 or header["numSplats"] != expected["numSplats"]:
        raise ValueError("The population or SH contract changed")
    if header["provenance"] != {name: expected[name] for name in ("spark_sha256", "source_ply_sha256")}:
        raise ValueError("The source or renderer provenance changed")
    encoding = header["splatEncoding"]
    keys = ("rgbMin", "rgbMax", "lnScaleMin", "lnScaleMax", "sh1Max", "sh2Max", "sh3Max")
    if not all(isinstance(encoding[key], (int, float)) and math.isfinite(encoding[key]) for key in keys) or encoding["lodOpacity"] is not False:
        raise ValueError("Unsupported native encoding")
    if len(header["buffers"]) != 4 or header["payloadBytes"] != len(data)-start:
        raise ValueError("A complete native buffer is absent")
    offset = 0
    for name, capacity, row in zip(NAMES, (4, 2, 4, 4), header["buffers"]):
        length = row["length"]
        if type(length) is not int or length < expected["numSplats"]*capacity or row["name"] != name or row["offset"] != offset or row["byteLength"] != length*4:
            raise ValueError("Invalid native buffer layout or complete population capacity")
        end = start+offset+row["byteLength"]
        if end > len(data) or digest(data[start+offset:end]) != row["sha256"]:
            raise ValueError("The native array payload is corrupt or truncated")
        offset += row["byteLength"]
    if offset != header["payloadBytes"]:
        raise ValueError("Unexpected trailing native payload")
    return header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8211)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(exist_ok=True)
    publish_lock = threading.Lock()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, directory=str(root), **kwargs)

        def json_reply(self, value, code=200):
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def available(self):
            return [branch for branch in sorted(BRANCHES) if (root / branch / "PACKAGING_RESULT.json").is_file()]

        def do_GET(self):
            if self.path.split("?", 1)[0] == "/available.json":
                return self.json_reply({"branches": self.available()})
            if self.path.split("?", 1)[0] == "/":
                links = []
                for branch in self.available():
                    metadata = read(root / branch / "cases/packed/display_manifest.json")
                    links.append('<li><a href="/'+branch+'/cases/packed/index.html">'+html.escape(metadata["label"])+"</a> · "+str(metadata["full_gaussian_count"])+" 个高斯</li>")
                body = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>四组原始录制对比</title><h1>四组原始录制 · 高斯模型对比</h1><p>使用左键旋转、右键平移、滚轮缩放。切换自由飞行后使用 W/A/S/D、Q/E，Shift 加速。每个模型均由本轮原始录制重跑。</p><ul>'+"".join(links)+'</ul><p>本页仅列出资产准备完成的模型。自动聚焦范围是通用显示裁剪，不代表物体真值分割；远离采集路径的画质需要单独检查。</p></html>').encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return self.wfile.write(body)
            return super().do_GET()

        def do_POST(self):
            try:
                parts = self.path.split("/")
                if len(parts) not in (4, 5) or parts[1] != "api" or parts[3] not in BRANCHES:
                    raise ValueError("Unknown publication route")
                branch = parts[3]
                directory = root / branch
                sources = read(directory / "ASSET_SOURCES.json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= LIMIT:
                    raise ValueError("Publication content length exceeds the fixed budget")
                data = self.rfile.read(length)
                if len(data) != length:
                    raise ValueError("Publication upload is incomplete")
                with publish_lock:
                    if parts[2] == "packed" and len(parts) == 5 and parts[4] in ("scene", "box"):
                        label = parts[4]
                        expected = sources[label]
                        if data_identity(directory / ("cases/packed/"+label+".ply"))["sha256"] != expected["source_ply_sha256"] or data_identity(root / "shared-build/vendor/spark.module.js")["sha256"] != expected["spark_sha256"]:
                            raise ValueError("The current complete source identity changed")
                        header = validate_asset(data, expected)
                        target = directory / ("cases/packed/"+label+".spk")
                        if target.exists() and target.read_bytes() != data:
                            raise ValueError("An existing packed artifact differs; a new version is required")
                        if not target.exists():
                            temporary = target.with_suffix(".spk.partial")
                            temporary.write_bytes(data)
                            temporary.replace(target)
                        result = {**expected, "asset_sha256": digest(data), "asset_bytes": len(data)}
                        atomic_json(directory / (label+"_PACK_VALIDATION.json"), {"status": "complete_native_payload_validated", "header": header, "asset": result})
                        manifest_path = directory / "ASSET_MANIFEST.json"
                        manifest = read(manifest_path) if manifest_path.exists() else {}
                        manifest[label] = result
                        atomic_json(manifest_path, manifest)
                        return self.json_reply(result)
                    if parts[2] == "packing-receipt" and len(parts) == 4 and length < 65536:
                        result = json.loads(data.decode("utf-8"))
                        manifest = read(directory / "ASSET_MANIFEST.json")
                        if result["status"] != "both_native_assets_published" or set(manifest) != {"scene", "box"} or len(result["records"]) != 2:
                            raise ValueError("The native packaging campaign is incomplete")
                        for label, row in zip(("scene", "box"), result["records"]):
                            if row["label"] != label or row["complete_native_arrays_equal"] is not True or row["numSplats"] != manifest[label]["numSplats"] or row["publication"] != manifest[label]:
                                raise ValueError("The complete client native array oracle changed")
                        atomic_json(directory / "PACKAGING_RESULT.json", {**result, "accepted_unix": time.time(), "complete_payload_validation": True,
                            "client_oracle_source": data_identity(directory / "cases/packed/pack_export.js"),
                            "server_source": data_identity(Path(__file__)), "browser_free_flight_verification_pending": True})
                        return self.json_reply({"status": "native_array_and_complete_asset_oracles_recorded"})
                    raise ValueError("Unknown publication action")
            except (ValueError, KeyError, OSError, UnicodeError, TypeError) as error:
                self.json_reply({"error": str(error)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(json.dumps({"status": "listening", "port": args.port, "root": str(root)}, ensure_ascii=False), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
