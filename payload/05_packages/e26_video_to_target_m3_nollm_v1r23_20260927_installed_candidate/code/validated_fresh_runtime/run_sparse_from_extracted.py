"""Run a fresh COLMAP sparse stage after the machine-only S02 gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_frontend(path: Path):
    spec = importlib.util.spec_from_file_location("e26_fresh_frontend", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load frontend: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--frontend", required=True, type=Path)
    parser.add_argument("--colmap", required=True, type=Path)
    parser.add_argument("--s02-receipt", required=True, type=Path)
    args = parser.parse_args()

    workspace = args.workspace.resolve(strict=True)
    s02 = read_json(args.s02_receipt.resolve(strict=True))
    if s02.get("status") != "complete_machine_only_input_integrity":
        raise RuntimeError("Machine-only S02 gate did not pass")
    if s02.get("language_model_used") is not False:
        raise RuntimeError("S02 receipt does not prove language_model_used=false")

    frontend = load_frontend(args.frontend.resolve(strict=True))
    manifest = read_json(workspace / "image_manifest.json")
    calibration = read_json(workspace / "camera_calibration.json")
    sparse = frontend.prepare_sparse(
        workspace,
        args.colmap.resolve(strict=True),
        calibration,
        manifest,
    )
    receipt_path = workspace / "workspace_receipt.json"
    receipt = read_json(receipt_path)
    receipt["sparse"] = sparse
    receipt["status"] = "sparse_and_undistorted_workspace_complete"
    receipt["s02_machine_receipt"] = str(args.s02_receipt.resolve(strict=True))
    receipt["language_model_used"] = False
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(receipt_path)
    print(json.dumps(sparse, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
