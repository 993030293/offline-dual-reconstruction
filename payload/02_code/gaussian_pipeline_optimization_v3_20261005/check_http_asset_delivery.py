"""Exercise actual transport, held bytes, negotiation, and bad release admission."""

import argparse
import gzip
from http.client import HTTPConnection
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from build_http_asset_packages import build
from http_asset_delivery import DeliveryServer


def request(server, method, path, encoding):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    try:
        connection.request(method, path, headers={"Accept-Encoding": encoding})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005")
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "opaque_source"
    source.mkdir()
    values = {"empty.spk": b"", "odd.spk": bytes(range(251)) * 11,
        "large.spk": bytes(range(256)) * 32771,
        "nonasset.js": b"export const unchanged = true;\n"}
    for name, value in values.items():
        (source / name).write_bytes(value)
    source_rows = {n: data_identity(source / n) for n in values}
    campaign = args.output / "campaign"
    for trial in range(3):
        for mode in ("identity", "gzip"):
            build(source, campaign / f"trial{trial}" / mode, mode == "gzip", source_rows)
    rows = []
    server = DeliveryServer(campaign, args.output / "SERVER.json")
    worker = threading.Thread(target=server.serve_forever, name="owned-http-oracle")
    worker.start()
    try:
        encodings = [("identity", "identity"), ("gzip", "gzip"), ("br, gzip;q=0", "identity"),
            ("*;q=.4", "gzip"), ("gzip;q=0, *;q=1", "identity"),
            ("gzip;q=NaN", "identity"), ("GZip ; q=1", "gzip"), ("", "identity"),
            ("gzip;q=2", "identity"), ("br", "identity")]
        for mode in ("identity", "gzip"):
            for name, original in values.items():
                for encoding, preferred in encodings:
                    status, headers, body = request(server, "GET", f"/trial0/{mode}/{name}", encoding)
                    expected = preferred if mode == "gzip" and name.endswith(".spk") else "identity"
                    decoded = gzip.decompress(body) if expected == "gzip" else body
                    checks = {"status200": status == 200, "all_decoded_bytes_exact": decoded == original,
                        "encoding_exact": headers.get("Content-Encoding", "identity") == expected,
                        "complete_wire_length_exact": int(headers["Content-Length"]) == len(body),
                        "vary_negotiation": headers.get("Vary") == "Accept-Encoding"}
                    rows.append({"name": name, "mode": mode, "accept_encoding": encoding, "checks": checks})
                    if not all(checks.values()):
                        raise AssertionError(str(rows[-1]))
        for method in ("HEAD", "GET"):
            for encoding in ("gzip;q=0, identity;q=0", "*;q=0"):
                status, _, _ = request(server, method, "/trial0/gzip/odd.spk", encoding)
                rows.append({"name": "explicit_no_representation_406", "method": method,
                    "encoding": encoding, "checks": {"406": status == 406}})
        for mode in ("identity", "gzip"):
            for encoding in ("gzip", "gzip;q=0"):
                status, headers, body = request(server, "HEAD", f"/trial0/{mode}/odd.spk", encoding)
                encoded = mode == "gzip" and encoding == "gzip"
                target = campaign / "trial0" / ("gzip/odd.spk.gz" if encoded else "identity/odd.spk")
                rows.append({"name": "HEAD_has_no_body_and_original_GET_length", "checks": {
                    "status": status == 200, "body_empty": body == b"",
                    "length": int(headers["Content-Length"]) == target.stat().st_size}})
        for route in ("/outside", "/trial0/gzip/missing.spk", "/trial0/gzip/%2e%2e/odd.spk"):
            status, _, _ = request(server, "GET", route, "gzip")
            rows.append({"name": "missing_or_escaped_route_404", "route": route, "checks": {"404": status == 404}})
        for name in ("identity/odd.spk", "gzip/odd.spk.gz"):
            refused = False
            try:
                with (campaign / "trial0" / name).open("r+b"):
                    raise AssertionError("A held transport artifact admitted mutation")
            except PermissionError:
                refused = True
            rows.append({"name": "held_actual_transport_mutation_refused", "file": name, "checks": {"refused": refused}})
    finally:
        server.shutdown()
        worker.join(timeout=10)
        server.server_close()
    if worker.is_alive() or server.snapshot.rows:
        raise RuntimeError("The actual server or held handles did not close")
    rows.append({"name": "complete_server_and_handles_closed", "checks": {"closed": True}})
    catalog_path = campaign / "trial0/gzip/FILE_INDEX.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    row = next(r for r in catalog["files"] if r["relative"] == "odd.spk")
    encoded_file = campaign / "trial0/gzip/odd.spk.gz"
    original_bytes = encoded_file.read_bytes()
    for defect in ("truncated_gzip", "corrupt_gzip_crc", "manifest_identity_mismatch"):
        fixture_catalog = json.loads(json.dumps(catalog))
        if defect == "truncated_gzip":
            encoded_file.write_bytes(original_bytes[:-3])
        elif defect == "corrupt_gzip_crc":
            changed = bytearray(original_bytes)
            changed[-5] ^= 1
            encoded_file.write_bytes(changed)
        else:
            encoded_file.write_bytes(original_bytes)
        changed_row = next(r for r in fixture_catalog["files"] if r["relative"] == "odd.spk")
        changed_row["encoded_identity"] = data_identity(encoded_file)
        if defect == "manifest_identity_mismatch":
            changed_row["encoded_identity"]["sha256"] = "0" * 64
        atomic_json(catalog_path, fixture_catalog)
        error = None
        try:
            unexpected = DeliveryServer(campaign, args.output / "unused.json")
        except (ValueError, EOFError, gzip.BadGzipFile) as caught:
            error = {"type": type(caught).__name__, "message": str(caught)}
        else:
            unexpected.server_close()
            raise AssertionError("A bad complete package was admitted")
        rows.append({"name": defect, "checks": {"rejected_before_listen": bool(error)}, "error": error})
    encoded_file.write_bytes(original_bytes)
    atomic_json(catalog_path, catalog)
    with encoded_file.open("r+b") as probe:
        probe.seek(0, 2)
    rows.append({"name": "constructor_failures_release_own_handles", "checks": {"original_writable_after_close": True}})
    if not all(all(row["checks"].values()) for row in rows):
        raise AssertionError("An original full transport or failure contract did not pass")
    atomic_json(args.output / "RESULT.json", {"status": "all_real_transport_oracles_passed",
        "checks": rows, "check_count": len(rows), "source_files": source_rows,
        "opaque_byte_fixtures_are_not_models_or_quality_evidence": True,
        "actual_loopback_server_started_and_closed": True, "counted": False})
    print(json.dumps({"status": "all_real_transport_oracles_passed", "check_count": len(rows)}))


if __name__ == "__main__":
    main()
