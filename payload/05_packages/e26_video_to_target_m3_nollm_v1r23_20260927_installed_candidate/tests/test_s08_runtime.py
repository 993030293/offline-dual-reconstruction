"""Focused regression tests for the machine-only S02 and S08 preview runtime."""

from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


PACKAGE = Path(__file__).resolve().parents[1]
RUNTIME = PACKAGE / "code" / "runtime"
FRESH_RUNTIME = PACKAGE / "code" / "validated_fresh_runtime"


def load_fresh_tsdf_module():
    path = FRESH_RUNTIME / "run_auto_roi_tsdf.py"
    spec = importlib.util.spec_from_file_location("fresh_auto_roi_tsdf", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_fresh_module(name: str):
    path = FRESH_RUNTIME / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class S08RuntimeTest(unittest.TestCase):
    def test_metric_policy_separates_sim3_from_camera_geometry_admission(self) -> None:
        policy = json.loads(
            (PACKAGE / "config" / "METRIC_ALIGNMENT_POLICY.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(policy["schema_version"], "e26_metric_alignment_policy_v1")
        self.assertEqual(policy["sim3_initialization"]["maximum_error_m"], 0.2)
        self.assertGreaterEqual(
            policy["camera_geometry_admission"]["minimum_triangulated_points"], 30
        )
        self.assertIn("independent sparse-geometry support", policy["claim_boundary"])

    def test_camera_geometry_gate_is_independent_of_sim3_flag(self) -> None:
        module = load_fresh_module("metric_admission")

        class Point:
            def __init__(self, x: float, y: float):
                self.xy = np.asarray([x, y], dtype=np.float64)

            def has_point3D(self) -> bool:
                return True

        class Camera:
            width = 100
            height = 100

        class Image:
            camera_id = 1
            name = "frame.jpg"

            def __init__(self, points):
                self.points2D = points

        class Reconstruction:
            cameras = {1: Camera()}

        policy = {
            "minimum_triangulated_points": 30,
            "grid_rows": 4,
            "grid_columns": 4,
            "minimum_occupied_grid_cells": 4,
            "minimum_bbox_area_fraction": 0.01,
        }
        spread = [
            Point(5 + 15 * (index % 6), 5 + 15 * (index // 6)) for index in range(36)
        ]
        admitted = module.camera_geometry_metrics(
            Reconstruction(), Image(spread), policy
        )
        concentrated = module.camera_geometry_metrics(
            Reconstruction(), Image([Point(50, 50) for _ in range(36)]), policy
        )
        self.assertTrue(admitted["passed"])
        self.assertFalse(concentrated["passed"])

    def test_failure_writer_preserves_failed_closed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requested = root / "new_run"
            raw = root / "input.mcap"
            raw.write_bytes(b"not-an-mcap")
            subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE / "code" / "write_pipeline_failure_receipt.py"),
                    "--output-root",
                    str(requested),
                    "--profile",
                    "ReliableLidar",
                    "--raw-mcap",
                    str(raw),
                    "--error-type",
                    "RuntimeError",
                    "--error-message",
                    "identity mismatch",
                ],
                check=True,
            )
            receipt = json.loads(
                requested.with_name("new_run.FAILURE.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "failed_closed")
            self.assertEqual(receipt["failed_stage"], "preflight_before_output_root")
            self.assertFalse(receipt["derived_outputs_reusable"])
            self.assertFalse(receipt["language_model_used"])

    def test_failure_writer_identifies_roi_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            completed = (
                "00_contract/raw_identity_receipt.json",
                "01_visual_workspace/workspace_receipt.json",
                "02_machine/s02_machine_frame_inventory.json",
                "03_metric_context/metric_time_support_receipt.json",
                "04_trajectory_binding/interpolated_camera_poses.json",
                "05_sim3/sim3.json",
                "06_pgsr_inlier/metric_input_receipt.json",
            )
            for relative in completed:
                marker = run / relative
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("{}", encoding="utf-8")
            raw = root / "input.mcap"
            raw.write_bytes(b"test")
            subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE / "code" / "write_pipeline_failure_receipt.py"),
                    "--output-root",
                    str(run),
                    "--profile",
                    "ReliableLidar",
                    "--raw-mcap",
                    str(raw),
                    "--error-type",
                    "RuntimeError",
                    "--error-message",
                    "ROI extent failed",
                ],
                check=True,
            )
            receipt = json.loads((run / "PIPELINE_FAILURE.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["failed_stage"], "S06_target_roi")
            self.assertFalse(receipt["derived_outputs_reusable"])

    def test_failure_writer_legacy_does_not_require_reliable_roi(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            completed = (
                "00_contract/raw_identity_receipt.json",
                "01_visual_workspace/workspace_receipt.json",
                "02_machine/s02_machine_frame_inventory.json",
                "03_metric_context/metric_time_support_receipt.json",
                "04_trajectory_binding/interpolated_camera_poses.json",
                "05_sim3/sim3.json",
                "06_pgsr_inlier/metric_input_receipt.json",
            )
            for relative in completed:
                marker = run / relative
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("{}", encoding="utf-8")
            raw = root / "input.mcap"
            raw.write_bytes(b"test")
            subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE / "code" / "write_pipeline_failure_receipt.py"),
                    "--output-root",
                    str(run),
                    "--profile",
                    "LegacyPgsr",
                    "--raw-mcap",
                    str(raw),
                    "--error-type",
                    "RuntimeError",
                    "--error-message",
                    "Training failed",
                ],
                check=True,
            )
            receipt = json.loads((run / "PIPELINE_FAILURE.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["failed_stage"], "S06_pgsr_training")

    def test_3d_roi_cluster_separates_xy_overlapping_height_layers(self) -> None:
        module = load_fresh_tsdf_module()
        target = np.array(
            [
                [x, y, z]
                for x in np.arange(0.0, 0.44, 0.04)
                for y in np.arange(0.0, 0.44, 0.04)
                for z in np.arange(0.0, 0.44, 0.04)
            ],
            dtype=np.float64,
        )
        overhead = np.array(
            [
                [x, y, z]
                for x in np.arange(0.0, 0.28, 0.04)
                for y in np.arange(0.0, 0.28, 0.04)
                for z in np.arange(1.10, 1.26, 0.04)
            ],
            dtype=np.float64,
        )
        selected, details = module.largest_density_cluster(np.vstack([target, overhead]))
        self.assertEqual(details["algorithm"], "deterministic 3D DBSCAN implemented with cKDTree")
        self.assertEqual(len(selected), len(target))
        self.assertLess(float(selected[:, 2].max()), 0.50)

    def test_machine_s02_does_not_make_semantic_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            rows = []
            for index in range(3):
                image = np.full((48, 64, 3), 40 + 20 * index, dtype=np.uint8)
                name = f"image_{index:05d}.jpg"
                cv2.imwrite(str(images / name), image)
                rows.append({"image_index": index, "image_file": name})
            manifest = root / "image_manifest.json"
            receipt = root / "s02.json"
            manifest.write_text(json.dumps({"records": rows}), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(RUNTIME / "machine_frame_inventory.py"),
                    "--image-manifest",
                    str(manifest),
                    "--image-root",
                    str(images),
                    "--output",
                    str(receipt),
                ],
                check=True,
            )
            result = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertFalse(result["language_model_used"])
            self.assertEqual(result["semantic_static_configuration"], "not_assessed")
            self.assertEqual(result["target_visibility"], "not_assessed")

    def test_s08_stops_before_component_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vertices = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 1, 1],
                ],
                dtype=np.float64,
            )
            triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]], dtype=np.int32)
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
            )
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[0.2, 0.7, 0.9]], dtype=np.float64), (8, 1))
            )
            mesh_path = root / "raw_mesh.ply"
            o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False)
            ground_path = root / "ground.json"
            ground_path.write_text(
                json.dumps(
                    {
                        "ground_plane_world_lio": {"normal": [0, 0, 1], "offset_m": 0},
                        "ground_triangle_rule": {
                            "maximum_signed_height_m": 0.04,
                            "minimum_normal_cosine": 0.65,
                            "unconditional_low_artifact_maximum_height_m": 0.015,
                        },
                    }
                ),
                encoding="utf-8",
            )
            output_ply = root / "s08.ply"
            output_mesh = root / "s08_mesh.ply"
            output_receipt = root / "s08.json"
            subprocess.run(
                [
                    sys.executable,
                    str(RUNTIME / "build_s08_preview_input.py"),
                    "--raw-mesh",
                    str(mesh_path),
                    "--ground-receipt",
                    str(ground_path),
                    "--output-ply",
                    str(output_ply),
                    "--output-mesh",
                    str(output_mesh),
                    "--output-receipt",
                    str(output_receipt),
                    "--sample-count",
                    "1000",
                    "--expected-triangles",
                    "2",
                ],
                check=True,
            )
            result = json.loads(output_receipt.read_text(encoding="utf-8"))
            self.assertEqual(result["step08_triangles"], 2)
            self.assertEqual(result["removed_ground_or_low_artifact_triangles"], 2)
            self.assertFalse(result["connected_component_filter_applied"])
            self.assertFalse(result["component_filter_code_executed"])
            self.assertTrue(result["mesh_generated"])
            self.assertEqual(result["observed_surface_mesh"]["triangles"], 2)
            self.assertFalse(result["observed_surface_mesh"]["watertightness_claim"])
            materialized = o3d.io.read_triangle_mesh(str(output_mesh))
            self.assertEqual(len(materialized.triangles), 2)
            header = output_ply.read_bytes()[:300].decode("ascii", errors="ignore")
            self.assertIn("element vertex 1000", header)
            self.assertIn("property float x", header)
            self.assertNotIn("property double", header)


if __name__ == "__main__":
    unittest.main()
