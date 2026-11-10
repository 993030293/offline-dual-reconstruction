"""Compare complete segmented results against the unchanged scalar support gates."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from check_grid_support import source_module, SOURCE, POLICY, ROOT
from grid_support import grid_support_execution
from machine_json import exact_json_values
from segmented_camera_support import native_support_many, segmented_rows
from runtime import atomic_json, data_identity


def outcome(operation):
    try:
        return {"status": "passed", "result": operation()}
    except BaseException as error:
        return {"status": "failed", "error_type": type(error).__name__, "error": str(error)}


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sg001")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("segmented_camera_support.py"),
             Path(__file__).with_name("native_observation_index.py"),
             Path(__file__).with_name("grid_support.py"), Path(__file__).with_name("machine_json.py"), SOURCE, POLICY]
    sources = [data_identity(path) for path in paths]
    module = source_module()
    original = module.camera_geometry_metrics
    policy = json.loads(POLICY.read_text(encoding="utf-8"))["camera_geometry_admission"]
    rng = np.random.default_rng(20261006)
    checks, records = {}, []
    try:
        with grid_support_execution(module, True):
            for rows, columns in ((1, 1), (4, 4), (3, 7), (64, 17), (65536, 65536)):
                for order in ("forward", "reverse"):
                    names, dimensions, pixels, images = [], [], [], []
                    for ordinal, (width, height) in enumerate(((1, 1), (13, 7), (800, 600), (1624, 1240))):
                        grid_x = np.linspace(0, width, min(columns, 64) + 1)
                        grid_y = np.linspace(0, height, min(rows, 64) + 1)
                        x = np.ravel(np.column_stack((np.nextafter(grid_x, -np.inf), grid_x,
                                                     np.nextafter(grid_x, np.inf))))
                        y = np.ravel(np.column_stack((np.nextafter(grid_y, -np.inf), grid_y,
                                                     np.nextafter(grid_y, np.inf))))
                        xx, yy = np.meshgrid(x, y)
                        fixtures = (np.empty((0, 2)), np.array([[-0., 0.]]),
                                    rng.uniform([-width, -height], [2*width, 2*height], (97, 2)),
                                    np.tile([width/2, height/2], (33, 1)),
                                    np.column_stack((np.linspace(0, width, 101),
                                                     height/2 + .001*height*np.sin(np.arange(101)))),
                                    np.column_stack((xx.ravel(), yy.ravel())))
                        for label, points in enumerate(fixtures):
                            name = f"fixture_{ordinal}_{label}"
                            names.append(name)
                            dimensions.append((width, height))
                            pixels.append(points)
                            values = [SimpleNamespace(xy=point.copy(), has_point3D=lambda: True) for point in points]
                            values.append(SimpleNamespace(xy=np.array([np.nan, np.nan]), has_point3D=lambda: False))
                            images.append(SimpleNamespace(camera_id=ordinal+1, name=name, points2D=values))
                    if order == "reverse":
                        names.reverse(); dimensions.reverse(); pixels.reverse(); images.reverse()
                    cameras = {ordinal+1: SimpleNamespace(width=size[0], height=size[1])
                               for ordinal, size in enumerate(((1, 1), (13, 7), (800, 600), (1624, 1240)))}
                    model = SimpleNamespace(cameras=cameras)
                    rules = {**policy, "grid_rows": rows, "grid_columns": columns}
                    before = [points.copy() for points in pixels]
                    expected = [module.camera_geometry_metrics(model, image, rules) for image in images]
                    actual = segmented_rows(names, dimensions, pixels, rules)
                    exact_json_values(expected, actual)
                    if any(not np.array_equal(left, right) for left, right in zip(before, pixels)):
                        raise AssertionError("Segmented support changed a caller pixel array")
                    key = f"all24_complete_fields_grid{rows}_{columns}_{order}"
                    checks[key] = True
                    records.append({"case": key, "images": len(images), "observations": sum(map(len, pixels)),
                                    "complete_values_types_order_float_bits_exact": True})
            camera = SimpleNamespace(width=13, height=7)
            foreign = SimpleNamespace(cameras={1: camera})
            for label, points, rules in (
                    ("observed_nan", np.array([[np.nan, 2.], [3., np.inf]]), policy),
                    ("all_empty_missing_grid", np.empty((0, 2)), {key:value for key, value in policy.items() if key != "grid_rows"}),
                    ("bad_grid", np.array([[3., 2.]]), {**policy, "grid_rows": "not_an_integer"})):
                image = SimpleNamespace(camera_id=1, name=label,
                    points2D=[SimpleNamespace(xy=point, has_point3D=lambda: True) for point in points])
                expected = outcome(lambda: [module.camera_geometry_metrics(foreign, image, rules)])
                actual = outcome(lambda: native_support_many(foreign, [image], rules,
                    args.output / label, module.camera_geometry_metrics)[0])
                exact_json_values(expected, actual)
                checks[f"unsupported_original_values_or_error_{label}"] = True
            expected = []
            actual = segmented_rows([], [], [], policy)
            exact_json_values(expected, actual)
            checks["empty_complete_list_preserved"] = True
        checks["original_metric_binding_restored"] = module.camera_geometry_metrics is original
        checks["source_policy_identity_unchanged"] = sources == [data_identity(path) for path in paths]
        result = {"status": "generic_complete_segmented_support_passed", "checks": checks,
                  "records": records, "sources": sources, "complete_seconds": time.perf_counter()-started,
                  "new_optimization_count": 0, "real_model_full_export_and_cost": "not_run"}
        atomic_json(args.output / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "checks": checks, "seconds": result["complete_seconds"]}))
    except BaseException as error:
        atomic_json(args.output / "FAILURE.json", {"status": "failed_original_segmented_support_contract",
            "error": repr(error), "checks_before_failure": checks, "records_before_failure": records,
            "sources": sources, "complete_seconds": time.perf_counter()-started, "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()
