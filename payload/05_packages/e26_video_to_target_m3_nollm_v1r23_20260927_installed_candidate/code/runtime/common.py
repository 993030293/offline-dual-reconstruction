"""Shared helpers for the S00-S08 preview package."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def require_new_directory(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists():
        raise FileExistsError(f"output already exists: {resolved}")
    resolved.mkdir(parents=True)
    return resolved


def image_rows(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, list):
        source_rows = document
    elif isinstance(document, dict):
        source_rows = document.get("records", document.get("rows"))
    else:
        source_rows = None
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError(
            "image manifest must be a non-empty list or contain a non-empty records or rows list"
        )

    rows = []
    for position, source_row in enumerate(source_rows):
        if not isinstance(source_row, dict):
            raise ValueError(f"image manifest row {position} must be an object")
        row = dict(source_row)
        row.setdefault("image_index", position)
        if "image_file" not in row and "image_name" in row:
            row["image_file"] = row["image_name"]
        if not isinstance(row.get("image_file"), str) or not row["image_file"]:
            raise ValueError(f"image manifest row {position} is missing image_file or image_name")
        rows.append(row)

    indices = [int(row["image_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("image indices must be consecutive and start at zero")
    return rows
