#!/usr/bin/env python
"""Validate the D1 one-million-point offline diagnostic viewer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
from pathlib import Path


def load_module(path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("pointcloud_viewer_v3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the viewer module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer-script", type=Path, required=True)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--viewer-html", type=Path, required=True)
    parser.add_argument("--viewer-receipt", type=Path, required=True)
    parser.add_argument("--node-executable", type=Path, required=True)
    parser.add_argument("--output-validation", type=Path, required=True)
    args = parser.parse_args()
    if args.output_validation.exists():
        raise FileExistsError(args.output_validation)
    viewer = load_module(args.viewer_script)
    receipt = json.loads(args.viewer_receipt.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "e26_pointcloud_interactive_viewer_receipt_v3":
        raise ValueError("Unexpected viewer receipt schema")
    if viewer.sha256(args.viewer_script) != receipt.get("generator_sha256"):
        raise ValueError("Viewer generator hash drifted")
    if viewer.sha256(args.input_ply) != receipt.get("input_ply_sha256"):
        raise ValueError("Input PLY hash drifted")
    if viewer.sha256(args.viewer_html) != receipt.get("output_html_sha256"):
        raise ValueError("Viewer HTML hash drifted")
    if receipt.get("display_point_counts") != {
        "performance": 150000,
        "balanced": 350000,
        "detail": 650000,
        "ultra": 1000000,
    }:
        raise ValueError("Viewer quality counts drifted")
    document = args.viewer_html.read_text(encoding="utf-8")
    external_scripts = re.findall(r"<script\b[^>]*\bsrc\s*=", document, re.IGNORECASE)
    scripts = re.findall(
        r"<script(?:\s[^>]*)?>(.*?)</script>", document, re.IGNORECASE | re.DOTALL
    )
    control_script = next(
        (script for script in reversed(scripts) if "const meta =" in script), None
    )
    if control_script is None:
        raise ValueError("Viewer control script is absent")
    javascript = subprocess.run(
        [str(args.node_executable), "--check", "-"],
        input=control_script,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if javascript.returncode != 0:
        raise ValueError(f"Viewer control JavaScript is invalid: {javascript.stderr}")
    required_controls = {
        "ultra_quality": 'data-quality="ultra"',
        "point_size": 'id="point-size"',
        "automatic_point_size": 'id="point-size-auto"',
        "projection_toggle": 'id="projection-toggle"',
        "subject_face_a": 'data-camera="subjectFaceA"',
        "subject_face_b": 'data-camera="subjectFaceB"',
        "right_grazing_minus": 'data-camera="rightGrazingMinus"',
        "right_grazing_plus": 'data-camera="rightGrazingPlus"',
        "s09_stop_boundary": "S09 连通组件过滤及其后续阶段均未运行",
    }
    control_checks = {name: token in document for name, token in required_controls.items()}
    if not all(control_checks.values()) or external_scripts:
        raise ValueError("Viewer controls or offline dependency boundary failed")
    points, _ = viewer.read_xyzrgb_ply(args.input_ply)
    indices = viewer.progressive_spatial_indices(points, 1_000_000)
    coverage = viewer.spatial_coverage(
        points,
        indices,
        receipt["display_point_counts"],
    )
    if coverage != receipt.get("spatial_coverage"):
        raise ValueError("Spatial coverage receipt drifted")
    for level in ("64", "128", "256"):
        values = [coverage[label][level] for label in viewer.QUALITY_LABELS]
        if any(left > right for left, right in zip(values, values[1:])):
            raise ValueError("Higher viewer quality reduced spatial coverage")
    validation = {
        "schema_version": "e26_pointcloud_interactive_viewer_validation_v2",
        "status": "pass",
        "stage": "D1",
        "viewer_script": str(args.viewer_script.resolve()),
        "viewer_script_sha256": viewer.sha256(args.viewer_script),
        "viewer_html": str(args.viewer_html.resolve()),
        "viewer_html_sha256": viewer.sha256(args.viewer_html),
        "viewer_receipt": str(args.viewer_receipt.resolve()),
        "viewer_receipt_sha256": viewer.sha256(args.viewer_receipt),
        "input_ply": str(args.input_ply.resolve()),
        "input_ply_sha256": viewer.sha256(args.input_ply),
        "input_point_count": int(points.shape[0]),
        "display_point_counts": receipt["display_point_counts"],
        "control_checks": control_checks,
        "javascript_syntax_valid": True,
        "external_script_count": 0,
        "offline_self_contained": True,
        "spatial_coverage": coverage,
        "source_ply_modified": False,
        "crop_modified": False,
        "synthetic_points_added": 0,
        "surface_completion_count": 0,
        "mesh_generated": False,
        "s09_executed": False,
        "claim_boundary": "S08 engineering preview validation only; S09 and later stages are excluded.",
    }
    args.output_validation.parent.mkdir(parents=True, exist_ok=True)
    with args.output_validation.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(validation, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
