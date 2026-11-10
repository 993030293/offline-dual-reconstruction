"""Check complete original camera support, boundaries, errors and real maps."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from grid_support import grid_support_execution
from machine_json import exact_json_values

PACKAGE = ROOT/"05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate"
SOURCE = PACKAGE/"code/validated_fresh_runtime/metric_admission.py"
POLICY = PACKAGE/"config/METRIC_ALIGNMENT_POLICY.json"


def source_module():
    specification = importlib.util.spec_from_file_location("private_grid_metric_source", SOURCE)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def outcome(operation):
    try:
        return {"status": "passed", "value": operation()}
    except BaseException as error:
        return {"status": "failed", "error_type": type(error).__name__, "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    module = source_module()
    original = module.camera_geometry_metrics
    policy = json.loads(POLICY.read_text(encoding="utf-8"))["camera_geometry_admission"]
    sources = [Path(__file__), Path(__file__).with_name("grid_support.py"),
               Path(__file__).with_name("machine_json.py"), SOURCE, POLICY]
    identities = [data_identity(path) for path in sources]
    checks, rows = {}, []

    def compare(name, model, image, rules):
        values = []
        for enabled in (False, True):
            with grid_support_execution(module, enabled):
                values.append(outcome(lambda: module.camera_geometry_metrics(model, image, rules)))
            if module.camera_geometry_metrics is not original:
                raise AssertionError("The original full camera support binding was not restored")
        exact_json_values(values[0], values[1])
        checks[name] = True
        rows.append({"case": name, "outcome": values[0]["status"],
                     "error_type": values[0].get("error_type"),
                     "all_fields_types_order_and_float_bits_exact": True})

    rng = np.random.default_rng(20261006)
    for width, height in ((1, 1), (13, 7), (800, 600), (1624, 1240)):
        for grid_rows, columns in ((1, 1), (4, 4), (3, 7), (64, 17)):
            xy = rng.uniform([-width, -height], [2*width, 2*height], (97, 2))
            edges = []
            for x in np.linspace(0, width, columns+1):
                edges.extend((np.nextafter(x, -np.inf), x, np.nextafter(x, np.inf)))
            for y in np.linspace(0, height, grid_rows+1):
                edges_y = (np.nextafter(y, -np.inf), y, np.nextafter(y, np.inf))
                xy = np.concatenate((xy, np.asarray([(x, edge) for x in edges for edge in edges_y])))
            definitions = {"random_and_all_grid_edge_nextafter": xy,
                           "empty": np.empty((0, 2)), "single": np.array([[0., 0.]]),
                           "duplicates": np.tile([width/2, height/2], (99, 1)),
                           "thin_curve": np.column_stack((np.linspace(0, width, 101),
                                height/2+height*.001*np.sin(np.arange(101)))),
                           "noncontinuous_source_xy": rng.normal(size=(34, 4))[:, ::2],
                           "nonfinite": np.array([[np.nan, 1.], [1., np.nan],
                                [np.inf, -np.inf], [-np.inf, np.inf], [-0., 0.]])}
            camera = SimpleNamespace(width=width, height=height)
            model = SimpleNamespace(cameras={1: camera})
            rules = {**policy, "grid_rows": grid_rows, "grid_columns": columns}
            for label, pixels in definitions.items():
                points = [SimpleNamespace(xy=p, has_point3D=lambda: True) for p in pixels]
                points += [SimpleNamespace(xy=np.array([np.nan, np.nan]), has_point3D=lambda: False)]
                image = SimpleNamespace(camera_id=1, name=f"generic_{width}_{height}_{label}", points2D=points)
                compare(f"{width}_{height}_{grid_rows}_{columns}_{label}", model, image, rules)
    points = [SimpleNamespace(xy=np.array([float(i), float(i%7)]), has_point3D=lambda: True)
              for i in range(31)]
    model = SimpleNamespace(cameras={1: SimpleNamespace(width=800, height=600)})
    image = SimpleNamespace(camera_id=1, name="unsupported_or_threshold_case", points2D=points)
    for label, change in (("zero_rows", {"grid_rows": 0}),
                          ("negative_columns", {"grid_columns": -1}),
                          ("large_integer_columns", {"grid_columns": 2**32}),
                          ("string_grid", {"grid_columns": "7"}),
                          ("invalid_grid", {"grid_columns": None}),
                          ("failed_minimum_points", {"minimum_triangulated_points": 32}),
                          ("exact_point_threshold", {"minimum_triangulated_points": 31}),
                          ("exact_grid_threshold", {"minimum_occupied_grid_cells": 1}),
                          ("failed_grid_count", {"minimum_occupied_grid_cells": 17}),
                          ("failed_bbox", {"minimum_bbox_area_fraction": 1.})):
        compare(label, model, image, {**policy, **change})
    boundary = (30./800)*(6./600)
    for label, threshold in (("below_bbox_threshold", np.nextafter(boundary, -np.inf)),
                             ("exact_bbox_threshold", boundary),
                             ("above_bbox_threshold", np.nextafter(boundary, np.inf))):
        compare(label, model, image, {**policy, "minimum_occupied_grid_cells": 1,
                                    "minimum_bbox_area_fraction": float(threshold)})
    for key in policy:
        compare("missing_policy_"+key, model, image, {k: v for k, v in policy.items() if k != key})
    for width, height in ((0, 0), (0, 600), (800, 0)):
        zero_model = SimpleNamespace(cameras={1: SimpleNamespace(width=width, height=height)})
        compare(f"zero_camera_size_{width}_{height}", zero_model, image, policy)
    folders = sorted(path for path in (ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/generic_binary_qa_001").iterdir()
                     if path.is_dir() and (path/"images.bin").is_file())
    if len(folders) != 9:
        raise ValueError("The nine existing real pycolmap geometry/calibration fixtures are missing")
    fixture_paths = [path for folder in folders for path in sorted(folder.glob("*.bin"))]
    fixture_identities = [data_identity(path) for path in fixture_paths]
    for folder in folders:
        real = pycolmap.Reconstruction(folder)
        for actual_image in real.images.values():
            compare(folder.name+"_"+actual_image.name, real, actual_image, policy)
    for error in (RuntimeError("caller failed"), KeyboardInterrupt()):
        try:
            with grid_support_execution(module, True):
                raise error
        except type(error):
            checks[type(error).__name__+"_restores_original_binding"] = module.camera_geometry_metrics is original
    try:
        module.camera_geometry_metrics = lambda *args: None
        try:
            with grid_support_execution(module, True):
                raise AssertionError("A changed complete source anchor was admitted")
        except ValueError:
            checks["changed_complete_source_anchor_rejected"] = True
    finally:
        module.camera_geometry_metrics = original
    checks["all_sources_and_nine_full_fixture_inputs_unchanged"] = (
        [data_identity(path) for path in sources] == identities
        and [data_identity(path) for path in fixture_paths] == fixture_identities)
    result = {"status": "all_complete_camera_grid_oracles_passed" if all(checks.values()) else "failed",
              "checks": checks, "checked": len(checks), "rows": rows,
              "sources": identities, "full_real_pycolmap_fixture_inputs": fixture_identities,
              "real_unknown_object_quality_or_GT_claim": False, "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "checked": len(checks)}))
    if not all(checks.values()):
        raise AssertionError("Original complete camera support or lifecycle protection changed")


if __name__ == "__main__":
    main()
