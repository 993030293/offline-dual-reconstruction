"""Extract the frozen M3 target surface from one bound S07 mesh and receipt."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import open3d as o3d

import ground_target_extraction as base
import m3_ground


def target_from_ground(
    source: o3d.geometry.TriangleMesh,
    remove: np.ndarray,
    normal: np.ndarray,
    offset_m: float,
    classification_policy: base.GroundClassificationPolicy,
    component_policy: base.TargetComponentPolicy,
) -> tuple[np.ndarray, list[dict], dict]:
    remaining = o3d.geometry.TriangleMesh(source)
    remaining.remove_triangles_by_mask(remove)
    remaining.remove_unreferenced_vertices()
    source_indices = np.flatnonzero(~remove)
    if not remaining.has_triangles() or len(remaining.triangles) != len(source_indices):
        raise RuntimeError("Source-to-remaining triangle mapping is invalid")
    keep, decisions, summary = base.component_keep_decisions(
        remaining, normal, offset_m, classification_policy, component_policy
    )
    target = np.zeros(len(remove), dtype=bool)
    target[source_indices[keep]] = True
    return target, decisions, summary


def run(args: argparse.Namespace) -> dict:
    raw_mesh = args.raw_mesh.resolve(strict=True)
    tsdf_path = args.tsdf_receipt.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    log = output / "RUN_LOG.jsonl"
    base.append_event(log, {"stage": "initialize", "status": "running", "method": "m3_grid_mrf"})

    config = base.read_json(config_path)
    tsdf = base.read_json(tsdf_path)
    provenance = base.validate_no_llm_input(tsdf)
    fit_policy, classification_policy, component_policy = base.load_policies(config)
    sample_count = args.sample_count or int(config["preview"]["sample_count"])
    if sample_count <= 0:
        raise ValueError("Sample count must be positive")

    source = o3d.io.read_triangle_mesh(str(raw_mesh))
    if not source.has_vertices() or not source.has_triangles():
        raise RuntimeError("S07 source mesh is empty")
    source_hash = base.validate_mesh_receipt(raw_mesh, source, tsdf)
    source.compute_triangle_normals()
    vertices = np.asarray(source.vertices, dtype=np.float64)
    triangles = np.asarray(source.triangles, dtype=np.int64)
    triangle_normals = np.asarray(source.triangle_normals, dtype=np.float64)
    if not np.isfinite(vertices).all() or not np.isfinite(triangle_normals).all():
        raise RuntimeError("S07 mesh contains nonfinite geometry")

    fit = base.fit_ground_plane(
        vertices, triangles, triangle_normals, tsdf, fit_policy,
        raw_mcap=args.raw_mcap,
        registered_scan_receipt=args.registered_scan_receipt,
    )
    normal = np.asarray(fit.pop("normal"), dtype=np.float64)
    offset_m = float(fit.pop("offset_m"))
    plane = {"normal": normal.tolist(), "offset_m": offset_m}
    base.atomic_write_json(output / "ground_plane_receipt.json", {
        "schema_version": "e26_m3_ground_plane_v1",
        "status": "ground_plane_fitted",
        "source_mesh": str(raw_mesh),
        "source_mesh_sha256": source_hash,
        "source_tsdf_receipt": str(tsdf_path),
        "source_tsdf_receipt_sha256": base.sha256_file(tsdf_path),
        "ground_plane_world_lio": plane,
        "fit": fit,
        "policy": config["ground_fit"],
        "upstream_provenance": provenance,
    })
    base.append_event(log, {"stage": "ground_fit", "status": "complete"})

    classification = base.classify_ground_triangles(
        vertices, triangles, triangle_normals, normal, offset_m, classification_policy
    )
    baseline_remove = np.asarray(classification["remove"], dtype=bool)
    baseline_target, _, baseline_components = target_from_ground(
        source, baseline_remove, normal, offset_m, classification_policy, component_policy
    )
    remove, graph = m3_ground.compute_m3_mask(
        vertices, triangles, normal, offset_m, classification
    )
    target, decisions, components = target_from_ground(
        source, remove, normal, offset_m, classification_policy, component_policy
    )
    contact = np.asarray(classification["preserve_contact"], dtype=bool)
    guarded_contact = baseline_target & contact & ~target & ~remove
    target[guarded_contact] = True
    protected_loss = baseline_target & contact & ~target
    if np.any(protected_loss):
        raise RuntimeError("M3 removed protected baseline contact triangles")
    if np.any(target & remove):
        raise RuntimeError("A source triangle was classified as both target and ground")

    labels = np.full(len(triangles), 2, dtype=np.uint8)
    labels[remove] = 0
    labels[target] = 1
    np.save(output / "source_triangle_labels.npy", labels, allow_pickle=False)
    base.append_event(log, {
        "stage": "m3_classification", "status": "complete",
        "removed_triangles": int(remove.sum()),
        "retained_triangles": int(target.sum()),
        "protected_contact_loss_triangles": int(protected_loss.sum()),
    })
    base.atomic_write_json(output / "component_decisions.json", {
        "schema_version": "e26_m3_component_decisions_v1",
        "status": "complete",
        "policy": config["target_components"],
        "summary_before_contact_guard": components,
        "guarded_contact_triangles": int(guarded_contact.sum()),
        "components": decisions,
    })

    target_mesh = o3d.geometry.TriangleMesh(source)
    target_mesh.remove_triangles_by_mask(~target)
    target_mesh.remove_unreferenced_vertices()
    if not target_mesh.has_triangles():
        raise RuntimeError("M3 target mesh is empty")
    target_mesh.compute_vertex_normals()
    mesh_path = output / "ground_removed_target_observed_surface_mesh.ply"
    base.write_mesh(mesh_path, target_mesh)
    o3d.utility.random.seed(int(config["preview"]["random_seed"]))
    cloud = target_mesh.sample_points_uniformly(number_of_points=sample_count)
    point_path = output / "ground_removed_target_xyzrgb_f32_1m.ply"
    base.write_xyzrgb_f32_ply(point_path, cloud)
    receipt = {
        "schema_version": "e26_m3_ground_target_extraction_v1",
        "status": "complete_ground_removed_target_roi_observed_surface",
        "method": "m3_grid_mrf",
        "language_model_used": False,
        "visual_language_model_used": False,
        "manual_visual_decision_used": False,
        "semantic_model_used": False,
        "object_mask_used": False,
        "glass_processing_used": False,
        "input": {
            "raw_mesh": str(raw_mesh),
            "raw_mesh_sha256": source_hash,
            "tsdf_receipt": str(tsdf_path),
            "tsdf_receipt_sha256": base.sha256_file(tsdf_path),
            "config": str(config_path),
            "config_sha256": base.sha256_file(config_path),
            "m3_code_sha256": base.sha256_file(Path(m3_ground.__file__)),
            "extraction_code_sha256": base.sha256_file(Path(__file__)),
            "upstream_provenance": provenance,
            "registered_scan_receipt": (str(args.registered_scan_receipt.resolve(strict=True))
                                        if args.registered_scan_receipt is not None else None),
            "registered_scan_receipt_sha256": (base.sha256_file(args.registered_scan_receipt)
                                               if args.registered_scan_receipt is not None else None),
        },
        "ground_plane": plane,
        "ground_classification": {
            "input_triangle_count": len(triangles),
            "baseline_removed_triangle_count": int(baseline_remove.sum()),
            "removed_triangle_count": int(remove.sum()),
            "preserved_contact_triangle_count": int(contact.sum()),
            "graph": graph,
            "policy": config["ground_classification"],
        },
        "component_filter": components,
        "baseline_component_filter": baseline_components,
        "contact_guard": {
            "restored_triangle_count": int(guarded_contact.sum()),
            "protected_contact_loss_triangle_count": int(protected_loss.sum()),
            "baseline_target_triangle_count": int(baseline_target.sum()),
            "newly_lost_baseline_target_triangles": int((baseline_target & ~target).sum()),
            "newly_gained_target_triangles": int((target & ~baseline_target).sum()),
        },
        "output": {
            "mesh": str(mesh_path),
            "mesh_sha256": base.sha256_file(mesh_path),
            "mesh_vertices": len(target_mesh.vertices),
            "mesh_triangles": len(target_mesh.triangles),
            "point_cloud": str(point_path),
            "point_cloud_sha256": base.sha256_file(point_path),
            "point_count": len(cloud.points),
            "source_triangle_labels": str(output / "source_triangle_labels.npy"),
            "source_triangle_labels_sha256": base.sha256_file(output / "source_triangle_labels.npy"),
        },
        "runtime_seconds": time.monotonic() - started,
        "claim_boundary": config["claim_boundary"],
        "evaluation_boundary": "Engineering preview; no independent ground-truth labels or heldout purity metric.",
    }
    base.atomic_write_json(output / "EXTRACTION_RECEIPT.json", receipt)
    base.append_event(log, {"stage": "finalize", "status": receipt["status"]})
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-mesh", required=True, type=Path)
    parser.add_argument("--raw-mcap", type=Path)
    parser.add_argument("--registered-scan-receipt", type=Path)
    parser.add_argument("--tsdf-receipt", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sample-count", type=int)
    args = parser.parse_args()
    try:
        receipt = run(args)
    except Exception as exc:
        output = args.output_root.resolve()
        failure_path = output / "FAILURE.json" if output.exists() else output.with_name(output.name + ".FAILURE.json")
        base.atomic_write_json(failure_path, {
            "schema_version": "e26_m3_failure_v1",
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "derived_output_reusable": False,
        })
        raise
    print(json.dumps({
        "status": receipt["status"],
        "method": receipt["method"],
        "input_triangles": receipt["ground_classification"]["input_triangle_count"],
        "removed_triangles": receipt["ground_classification"]["removed_triangle_count"],
        "target_triangles": receipt["output"]["mesh_triangles"],
    }))


if __name__ == "__main__":
    main()
