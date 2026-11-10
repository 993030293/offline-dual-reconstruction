"""Audit M3 source labels, mesh identity, viewer, and machine-only boundary."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np

import ground_target_extraction as base


BANNED = {"anthropic", "google.generativeai", "httpx", "openai", "requests", "socket", "transformers", "urllib"}
FALSE_FLAGS = (
    "language_model_used", "visual_language_model_used", "manual_visual_decision_used",
    "semantic_model_used", "object_mask_used", "glass_processing_used",
)


def imports_under(root: Path) -> set[str]:
    found: set[str] = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for item in ast.walk(tree):
            if isinstance(item, ast.Import):
                found.update(alias.name for alias in item.names)
            elif isinstance(item, ast.ImportFrom) and item.module:
                found.add(item.module)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve(strict=True)
    package = args.package_root.resolve(strict=True)
    extraction = base.read_json(output / "EXTRACTION_RECEIPT.json")
    viewer = base.read_json(output / "offline_viewer_receipt.json")
    validation = base.read_json(output / "offline_viewer_validation.json")
    delivery = base.read_json(output / "DELIVERY.json")
    ground = extraction["ground_classification"]
    components = extraction["component_filter"]
    contact = extraction["contact_guard"]
    exported = extraction["output"]
    labels_path = Path(exported["source_triangle_labels"]).resolve(strict=True)
    labels = np.load(labels_path, allow_pickle=False)
    mesh = Path(exported["mesh"]).resolve(strict=True)
    points = Path(exported["point_cloud"]).resolve(strict=True)
    html = Path(delivery["preview_html"]).resolve(strict=True)
    code_imports = imports_under(package / "ground" / "code")
    banned = sorted(name for name in code_imports if any(name == item or name.startswith(item + ".") for item in BANNED))
    checks = {
        "method_m3": extraction.get("method") == "m3_grid_mrf",
        "extraction_complete": extraction.get("status") == "complete_ground_removed_target_roi_observed_surface",
        "machine_only_flags": all(extraction.get(name) is False for name in FALSE_FLAGS),
        "glass_disabled": delivery.get("glass_processing_used") is False,
        "source_mesh_identity": base.sha256_file(Path(extraction["input"]["raw_mesh"])) == extraction["input"]["raw_mesh_sha256"],
        "tsdf_receipt_identity": base.sha256_file(Path(extraction["input"]["tsdf_receipt"])) == extraction["input"]["tsdf_receipt_sha256"],
        "labels_identity": base.sha256_file(labels_path) == exported["source_triangle_labels_sha256"],
        "labels_domain": labels.dtype == np.uint8 and np.isin(labels, [0, 1, 2]).all(),
        "labels_cover_source": len(labels) == int(ground["input_triangle_count"]),
        "removed_labels_match": int(np.count_nonzero(labels == 0)) == int(ground["removed_triangle_count"]),
        "target_labels_match": int(np.count_nonzero(labels == 1)) == int(exported["mesh_triangles"]),
        "component_accounting": int(ground["input_triangle_count"]) - int(ground["removed_triangle_count"])
        == int(components["retained_triangle_count"]) + int(components["rejected_triangle_count"]),
        "contact_guard_accounting": int(exported["mesh_triangles"])
        == int(components["retained_triangle_count"]) + int(contact["restored_triangle_count"]),
        "protected_contact_preserved": contact.get("protected_contact_loss_triangle_count") == 0,
        "mesh_identity": base.sha256_file(mesh) == exported["mesh_sha256"],
        "point_identity": base.sha256_file(points) == exported["point_cloud_sha256"],
        "output_paths_current": mesh.parent == output and points.parent == output and html.parent == output,
        "viewer_input_identity": viewer.get("input_ply_sha256") == exported["point_cloud_sha256"],
        "viewer_generator_package_owned": Path(viewer["generator_path"]).resolve().is_relative_to(package),
        "viewer_offline": viewer.get("offline_self_contained") is True and viewer.get("external_script_count") == 0,
        "viewer_validation": validation.get("status") == "pass" and validation.get("javascript_syntax_valid") is True,
        "viewer_html_identity": base.sha256_file(html) == viewer.get("output_html_sha256"),
        "delivery_complete": delivery.get("status") == "complete" and delivery.get("historical_s09_executed") is False,
        "no_failure": not (output / "FAILURE.json").exists() and not (output / "WRAPPER_FAILURE.json").exists(),
        "no_network_or_foundation_model_imports": not banned,
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, ok in checks.items() if not ok]
    audit = {
        "schema_version": "e26_m3_delivery_audit_v1",
        "status": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "banned_imports_found": banned,
        "upstream_provenance": extraction["input"]["upstream_provenance"],
        "triangle_label_counts": {str(value): int(np.count_nonzero(labels == value)) for value in (0, 1, 2)},
        "claim_boundary": extraction["claim_boundary"],
        "evaluation_boundary": extraction["evaluation_boundary"],
    }
    base.atomic_write_json(output / "FINAL_AUDIT.json", audit)
    print(json.dumps({"status": audit["status"], "failed_checks": failed, "triangle_label_counts": audit["triangle_label_counts"]}))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
