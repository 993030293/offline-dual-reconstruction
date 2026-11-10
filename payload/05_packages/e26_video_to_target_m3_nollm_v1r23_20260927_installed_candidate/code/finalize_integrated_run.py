"""Validate an integrated S00-S08 run and record its observed mesh artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_no_language_model(name: str, value: dict) -> None:
    if value.get("language_model_used") is not False:
        raise RuntimeError(f"{name} does not prove language_model_used=false")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=("ReliableLidar", "LegacyPgsr"))
    parser.add_argument("--tsdf-stage", required=True)
    parser.add_argument("--tsdf-receipt-name", required=True)
    parser.add_argument("--preview-stage", required=True)
    parser.add_argument("--output-name", default="INTEGRATED_FINAL_STATUS.json")
    args = parser.parse_args()

    run = args.run_root.resolve(strict=True)
    output = run / args.output_name
    if output.exists():
        raise FileExistsError(output)

    contract = read_json(run / "00_contract" / "run_contract.json")
    metric_input = read_json(run / "06_pgsr_inlier" / "metric_input_receipt.json")
    training = read_json(run / "06_pgsr_inlier" / "training_receipt.json")
    tsdf_root = run / args.tsdf_stage
    preview_root = run / args.preview_stage
    tsdf = read_json(tsdf_root / args.tsdf_receipt_name)
    ground = read_json(tsdf_root / "ground_filter_receipt.json")
    preview = read_json(preview_root / "s08_preview_derivative_receipt.json")
    viewer = read_json(preview_root / "s08_offline_viewer_receipt.json")
    validation = read_json(preview_root / "s08_offline_viewer_validation.json")

    for name, value in (
        ("run contract", contract),
        ("metric input receipt", metric_input),
        ("training receipt", training),
        ("TSDF receipt", tsdf),
        ("ground receipt", ground),
        ("S08 receipt", preview),
    ):
        require_no_language_model(name, value)
    if metric_input.get("schema_version") != "e26_mask_free_metric_pgsr_inlier_input_v3":
        raise RuntimeError("Metric PGSR input lacks the repaired lineage and geometry admission contract")
    if metric_input.get("status") != "admitted_for_mask_free_pgsr":
        raise RuntimeError("Metric PGSR input was not admitted")
    if metric_input.get("independent_geometry_admitted_camera_count") != training.get(
        "registered_camera_count"
    ):
        raise RuntimeError("Training camera count differs from independent geometry admission")
    if preview.get("connected_component_filter_applied") is not False:
        raise RuntimeError("S09 component filtering was applied")
    if preview.get("mesh_generated") is not True:
        raise RuntimeError("S08 observed-surface mesh was not materialized")
    if validation.get("status") != "pass" or validation.get("s09_executed") is not False:
        raise RuntimeError("Offline viewer validation did not pass the S08 terminal gate")

    mesh_info = preview.get("observed_surface_mesh")
    if not isinstance(mesh_info, dict):
        raise RuntimeError("S08 receipt lacks observed_surface_mesh")
    mesh_path = Path(mesh_info["path"]).resolve(strict=True)
    if sha256_file(mesh_path) != mesh_info["sha256"]:
        raise RuntimeError("S08 observed-surface mesh identity changed")
    html = Path(viewer["output_html"]).resolve(strict=True)
    if sha256_file(html) != viewer["output_html_sha256"]:
        raise RuntimeError("Offline HTML identity changed")

    result = {
        "schema_version": "e26_integrated_s00_s08_final_status_v1",
        "status": "complete_s08_observed_surface",
        "profile": args.profile,
        "run_root": str(run),
        "language_model_used": False,
        "visual_language_model_used": False,
        "manual_visual_decision_used": False,
        "semantic_model_used": False,
        "object_mask_used": False,
        "s09_executed": False,
        "surface_completion_used": False,
        "metric_alignment": {
            "policy_sha256": metric_input["policy_sha256"],
            "sim3_sha256": metric_input["source_sim3_sha256"],
            "sim3_inlier_cameras": metric_input["metric_ransac_inlier_prior_count"],
            "independent_geometry_admitted_cameras": metric_input[
                "independent_geometry_admitted_camera_count"
            ],
            "sim3_only_camera_admission": False,
        },
        "unknown_surface_policy": preview["unknown_surface_policy"],
        "observed_surface_mesh": mesh_info,
        "preview": {
            "points": preview["preview_surface_points"],
            "point_cloud": preview["output_ply"],
            "html": str(html),
            "viewer_validation": validation["status"],
        },
        "scientific_boundary": (
            "The run materializes the observed S08 triangle surface and an offline preview. "
            "Open or unobserved surfaces remain unknown; this status does not claim watertightness, "
            "dimension accuracy, physical completeness, or final manufacturing readiness."
        ),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
