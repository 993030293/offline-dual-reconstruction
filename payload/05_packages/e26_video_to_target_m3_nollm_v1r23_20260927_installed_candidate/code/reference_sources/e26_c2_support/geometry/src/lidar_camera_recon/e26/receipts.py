"""Immutable receipt helpers for E26 stage entry points."""

from __future__ import annotations

import json
from pathlib import Path


def new_run_receipt(*, scientific_boundary: str) -> dict[str, object]:
    return {
        "schema_version": "e26_joint_pose_masked_mvs_run_receipt_v1",
        "configured": True,
        "started": False,
        "completed": False,
        "validated": False,
        "scientifically_accepted": False,
        "failure_stage": None,
        "scientific_boundary": scientific_boundary,
    }


def write_json_new(path: Path, payload: object) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
