"""Content-addressed, read-only input freezing for E26."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .trajectory_io import sha256_file


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def directory_signature(path: Path) -> tuple[str, list[dict[str, object]]]:
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"directory input is not a directory: {root}")
    rows: list[dict[str, object]] = []
    for item in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if not item.is_file():
            continue
        rows.append(
            {
                "relative_path": item.relative_to(root).as_posix(),
                "size_bytes": int(item.stat().st_size),
                "sha256": sha256_file(item),
            }
        )
    if not rows:
        raise ValueError(f"directory input contains no files: {root}")
    return hashlib.sha256(canonical_json(rows)).hexdigest(), rows


@dataclass(frozen=True)
class InputAsset:
    asset_id: str
    path: Path
    asset_type: str
    expected_sha256: str | None = None


def inspect_asset(asset: InputAsset) -> dict[str, object]:
    path = asset.path.resolve(strict=True)
    if not asset.asset_id or not asset.asset_type:
        raise ValueError("asset_id and asset_type are required")
    if path.is_file():
        digest = sha256_file(path)
        record: dict[str, object] = {
            "asset_id": asset.asset_id,
            "asset_type": asset.asset_type,
            "kind": "file",
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": digest,
        }
    elif path.is_dir():
        digest, rows = directory_signature(path)
        record = {
            "asset_id": asset.asset_id,
            "asset_type": asset.asset_type,
            "kind": "directory",
            "path": str(path),
            "file_count": len(rows),
            "sha256": digest,
            "files": rows,
        }
    else:
        raise ValueError(f"unsupported input asset: {path}")
    if asset.expected_sha256 is not None:
        expected = asset.expected_sha256.lower()
        if len(expected) != 64:
            raise ValueError(f"invalid expected SHA256 for {asset.asset_id}")
        if digest != expected:
            raise ValueError(
                f"SHA256 mismatch for {asset.asset_id}: {digest} != {expected}"
            )
        record["expected_sha256"] = expected
        record["expected_sha256_matches"] = True
    return record


def build_input_manifest(
    assets: list[InputAsset],
    *,
    configuration_sha256: str,
    implementation_sha256: str,
) -> dict[str, object]:
    identifiers = [asset.asset_id for asset in assets]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("input asset IDs must be unique")
    records = [inspect_asset(asset) for asset in assets]
    manifest = {
        "schema_version": "e26_input_manifest_v1",
        "status": "frozen_read_only",
        "coordinate_contract": {
            "transform_convention": "T_target_source",
            "translation_unit": "m",
            "quaternion_order": "xyzw",
            "timestamp_unit": "int64_nanoseconds",
        },
        "configuration_sha256": configuration_sha256,
        "implementation_sha256": implementation_sha256,
        "assets": records,
    }
    manifest["manifest_content_sha256"] = hashlib.sha256(
        canonical_json(manifest)
    ).hexdigest()
    return manifest


def write_input_freeze(
    output_directory: Path,
    manifest: dict[str, object],
) -> tuple[Path, Path]:
    output = output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "input_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sha_path = output / "sha256_manifest.txt"
    lines: list[str] = []
    for asset in manifest["assets"]:
        lines.append(f"{asset['sha256']}  {asset['asset_id']}  {asset['path']}")
    lines.append(f"{sha256_file(manifest_path)}  input_manifest.json")
    sha_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path, sha_path
