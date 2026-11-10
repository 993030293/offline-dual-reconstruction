"""Write or verify the package file manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXCLUDED = {
    "MANIFEST.json",
    "INSTALL_RECEIPT.json",
    "INSTALL_VERIFY.json",
    "config/local_runtime.json",
}
PACKAGE_ID = "e26_video_to_target_m3_nollm_v1r23_20260927"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in EXCLUDED or "__pycache__" in path.parts:
            continue
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.write == args.verify:
        raise ValueError("select exactly one of --write or --verify")
    root = args.root.resolve(strict=True)
    manifest_path = root / "MANIFEST.json"
    current = inventory(root)
    if args.write:
        document = {
            "schema_version": "e26_video_to_target_m3_manifest_v1",
            "package_id": PACKAGE_ID,
            "file_count": len(current),
            "total_bytes": sum(int(row["bytes"]) for row in current),
            "files": current,
        }
        manifest_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {manifest_path} with {len(current)} files")
        return
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected.get("schema_version") != "e26_video_to_target_m3_manifest_v1":
        raise RuntimeError("Package manifest schema is invalid")
    if expected.get("package_id") != PACKAGE_ID:
        raise RuntimeError("Package manifest ID is invalid")
    if expected.get("file_count") != len(current):
        raise RuntimeError("Package manifest file count is invalid")
    if expected.get("total_bytes") != sum(int(row["bytes"]) for row in current):
        raise RuntimeError("Package manifest byte count is invalid")
    if expected.get("files") != current:
        expected_map = {row["path"]: row for row in expected.get("files", [])}
        current_map = {row["path"]: row for row in current}
        missing = sorted(set(expected_map) - set(current_map))
        added = sorted(set(current_map) - set(expected_map))
        changed = sorted(
            path for path in set(expected_map) & set(current_map) if expected_map[path] != current_map[path]
        )
        print(json.dumps({"status": "failed", "missing": missing, "added": added, "changed": changed}, indent=2))
        raise SystemExit(2)
    print(json.dumps({"status": "verified", "file_count": len(current)}, indent=2))


if __name__ == "__main__":
    main()
