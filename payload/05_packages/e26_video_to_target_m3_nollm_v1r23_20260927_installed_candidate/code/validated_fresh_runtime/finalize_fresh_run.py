"""Validate the terminal S08 artifacts and write a compact run summary."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--tsdf-stage", default="07_roi_tsdf")
    parser.add_argument("--preview-stage", default="08_s08_preview")
    parser.add_argument("--output-name", default="FINAL_STATUS.json")
    args = parser.parse_args()
    run = args.run_root.resolve(strict=True)
    output = run / args.output_name
    if output.exists():
        raise FileExistsError(output)
    contract = read_json(run / "00_contract" / "run_contract.json")
    workspace = read_json(run / "01_visual_workspace" / "workspace_receipt.json")
    machine = read_json(run / "02_machine" / "s02_machine_frame_inventory.json")
    support = read_json(run / "03_metric_context" / "metric_time_support_receipt.json")
    binding = read_json(run / "04_trajectory_binding" / "interpolated_camera_poses.json")
    sim3 = read_json(run / "05_sim3" / "sim3.json")
    metric_input = read_json(run / "06_pgsr_inlier" / "metric_input_receipt.json")
    training = read_json(run / "06_pgsr_inlier" / "training_receipt.json")
    tsdf = read_json(run / args.tsdf_stage / "roi_tsdf_receipt.json")
    ground = read_json(run / args.tsdf_stage / "ground_filter_receipt.json")
    preview = read_json(run / args.preview_stage / "s08_preview_derivative_receipt.json")
    viewer = read_json(run / args.preview_stage / "s08_offline_viewer_receipt.json")
    validation = read_json(run / args.preview_stage / "s08_offline_viewer_validation.json")
    if any(
        value.get("language_model_used") is not False
        for value in (contract, machine, metric_input, training, tsdf, ground, preview)
    ):
        raise RuntimeError("At least one required receipt lacks language_model_used=false")
    if metric_input.get("schema_version") != "e26_mask_free_metric_pgsr_inlier_input_v3":
        raise RuntimeError("Metric PGSR input lacks repaired lineage and geometry admission")
    if metric_input.get("independent_geometry_admitted_camera_count") != training.get(
        "registered_camera_count"
    ):
        raise RuntimeError("Training camera count differs from independent geometry admission")
    if preview.get("connected_component_filter_applied") is not False:
        raise RuntimeError("S09 component filtering was applied")
    if validation.get("status") != "pass" or validation.get("s09_executed") is not False:
        raise RuntimeError("Offline viewer validation did not pass the S08 terminal gate")
    html = Path(viewer["output_html"]).resolve(strict=True)
    if sha256_file(html) != viewer["output_html_sha256"]:
        raise RuntimeError("Offline HTML identity changed after validation")
    raw_mcap = contract.get("raw_mcap") or {
        "path": contract["mcap_path"],
        "bytes": contract["mcap_bytes"],
        "sha256": contract["mcap_sha256"],
    }
    status = {
        "schema_version": "e26_video_to_s08_nollm_final_status_v2",
        "status": "complete_s00_through_s08",
        "run_root": str(run),
        "tsdf_stage": args.tsdf_stage,
        "preview_stage": args.preview_stage,
        "raw_mcap": raw_mcap,
        "language_model_used": False,
        "semantic_model_used": False,
        "sam2_used": False,
        "object_mask_used": False,
        "registered_images": workspace["sparse"]["registered_image_count"],
        "metric_supported_images": support["metric_bound_count"],
        "bound_images": binding["bound_count"],
        "sim3_correspondences": sim3["correspondence_count"],
        "sim3_inliers": sim3["inlier_count"],
        "sim3_only_camera_admission": False,
        "independent_geometry_admitted_cameras": metric_input[
            "independent_geometry_admitted_camera_count"
        ],
        "metric_alignment_policy_sha256": metric_input["policy_sha256"],
        "trained_cameras": training["registered_camera_count"],
        "gaussian_count": training["gaussian_count"],
        "raw_tsdf_vertices": tsdf["mesh"]["vertices"],
        "raw_tsdf_triangles": tsdf["mesh"]["triangles"],
        "step08_vertices": preview["step08_vertices"],
        "step08_triangles": preview["step08_triangles"],
        "preview_points": preview["preview_surface_points"],
        "offline_html": str(html),
        "offline_html_bytes": html.stat().st_size,
        "offline_html_sha256": sha256_file(html),
        "offline_self_contained": viewer["offline_self_contained"],
        "viewer_validation_status": validation["status"],
        "s09_executed": False,
        "terminal_boundary": "S08 ground removal only; connected-component filtering and later stages are excluded.",
    }
    output.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
