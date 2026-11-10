"""Validate the readback checker and original invalid-projection oracle."""

import argparse
import json
from pathlib import Path
import time

STARTED = time.perf_counter()
import numpy as np
import pycolmap
from profile_direct_schur import same_model
import run_metric_refinement as original
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        if json.loads((args.fixtures / "RESULT.json").read_text())["status"] != "actual_backend_generic_passed":
            raise ValueError("Oracle check requires this candidate's own complete fresh fixtures")
        for number in (0, 1):
            for label in ("visual_metric_aligned", "weak_position_prior"):
                path = args.fixtures / ("fixture_" + str(number)) / label
                left, right = pycolmap.Reconstruction(path), pycolmap.Reconstruction(path)
                checks = same_model(left, right)
                point_id = next(iter(right.points3D))
                right.points3D[point_id].xyz = right.points3D[point_id].xyz + np.array([.01, .02, .03])
                changed = same_model(left, right)
                checks["readback_xyz_corruption_detected"] = changed["all_xyz_color_error_tracks"] is False
                malformed = pycolmap.Reconstruction(left)
                malformed.points3D[point_id].xyz = np.array([float("nan")]*3)
                invalid = original.projection_validity(malformed)
                checks["original_nonfinite_projection_rejected"] = not invalid["valid"] and invalid["invalid_count"] > 0
                image = next(image for image in left.images.values() if image.has_pose)
                entry = next(entry for entry in image.points2D if entry.has_point3D())
                malformed = pycolmap.Reconstruction(left)
                pose = image.cam_from_world()
                malformed.points3D[entry.point3D_id].xyz = pose.rotation.matrix().T @ (np.array([0., 0., -1.])-pose.translation)
                invalid = original.projection_validity(malformed)
                checks["original_negative_depth_rejected"] = not invalid["valid"] and any(r["depth_m"] is not None and r["depth_m"] <= 0 for r in invalid["invalid"])
                results.append({"fixture": number, "case": label, "checks": checks})
                if not all(checks.values()): raise AssertionError("Fresh checker oracle failed")
        atomic_json(args.output / "RESULT.json", {"status": "full_readback_and_projection_oracle_passed", "results": results,
                   "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("profile_direct_schur.py"))],
                   "whole_seconds": time.perf_counter()-STARTED, "counted": False})
    except BaseException as error:
        atomic_json(args.output / "FAILURE.json", {"status": "checker_setup_error_not_candidate_failure", "error": repr(error),
                   "results": results, "whole_seconds": time.perf_counter()-STARTED, "counted": False})
        raise


if __name__ == "__main__": main()
