"""Exercise actual official sparse PLY exports and original failure routes."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
import export_admitted_map as native
from runtime import atomic_json, data_identity
from sparse_ply_oracle import sparse_ply_execution


def outcome(function, model, path):
    try:
        function(model, path)
        return {"status": "passed"}
    except BaseException as error:
        return {"status": "failed", "error_type": type(error).__name__}


def make_model(count, scale, same_color=False, duplicate=False, special=None):
    model = pycolmap.Reconstruction()
    rng = np.random.default_rng(count)
    xyz = rng.normal(size=(count, 3))*scale
    if count:
        xyz[0] = [-0., 0., scale]
    if duplicate and count > 1:
        xyz[1:] = xyz[0]
    if special is not None and count:
        xyz[-1, 0] = special
    colors = rng.integers(0, 256, (count, 3), dtype=np.uint8)
    if same_color:
        colors[:] = [0, 255, 0]
    for position, color in zip(xyz, colors, strict=True):
        model.add_point3D(position, pycolmap.Track(), color)
    if count > 3:
        model.delete_point3D(2)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks, traces = {}, []
    original = native.write_initial_ply
    cases = [(0, 1., False, False, None), (1, 1., False, False, None)]
    cases += [(count, scale, same, duplicate, None)
              for count in (2, 17, 4097) for scale in (1e-6, 1., 1e6)
              for same, duplicate in ((False, False), (True, True))]
    cases += [(3, 1., False, False, value) for value in (np.nan, np.inf, -np.inf)]
    for index, values in enumerate(cases):
        model = make_model(*values)
        results, outputs = [], []
        for enabled in (False, True):
            path = args.output/f"case{index:02d}_{int(enabled)}.ply"
            with sparse_ply_execution(native, enabled) as trace:
                results.append(outcome(native.write_initial_ply, model, path))
            outputs.append(path.read_bytes() if path.exists() else None)
            traces.append(trace)
            assert native.write_initial_ply is original
        checks[f"actual_official_roundtrip_or_original_rejection_{index:02d}"] = (
            results[0] == results[1] and outputs[0] == outputs[1])
        if not checks[f"actual_official_roundtrip_or_original_rejection_{index:02d}"]:
            raise AssertionError((index, results))
    model = make_model(5, 1.)
    for name, prepare in (("existing_file", lambda p: p.write_bytes(b"existing")),
                          ("missing_parent", lambda p: None)):
        results = []
        for enabled in (False, True):
            folder = args.output/f"error_{name}_{int(enabled)}"
            if name == "existing_file":
                folder.mkdir()
            path = folder/"points.ply"
            prepare(path)
            with sparse_ply_execution(native, enabled):
                results.append(outcome(native.write_initial_ply, model, path))
            assert native.write_initial_ply is original
            if name == "existing_file":
                assert path.read_bytes() == b"existing"
        checks[name] = results[0] == results[1] and results[0]["status"] == "failed"
    saved_pycolmap = native.pycolmap
    def reader_factory():
        def fail_import(path):
            raise OSError("Injected official importer read failure")
        return SimpleNamespace(import_PLY=fail_import)
    try:
        native.pycolmap = SimpleNamespace(Reconstruction=reader_factory)
        results = []
        for enabled in (False, True):
            path = args.output/f"read_failure_{int(enabled)}.ply"
            with sparse_ply_execution(native, enabled):
                results.append(outcome(native.write_initial_ply, model, path))
            assert native.write_initial_ply is original
        checks["original_importer_failure_propagates"] = results[0] == results[1] == {
            "status": "failed", "error_type": "OSError"}
    finally:
        native.pycolmap = saved_pycolmap
    try:
        with sparse_ply_execution(native, True):
            raise KeyboardInterrupt("Injected caller cancellation")
    except KeyboardInterrupt:
        pass
    checks["cancellation_restores_original_binding"] = native.write_initial_ply is original
    result = {"status": "all_actual_sparse_export_oracles_passed" if all(checks.values()) else "failed",
              "checks": checks, "checked": len(checks), "traces": traces,
              "source": [data_identity(Path(__file__)), data_identity(Path(native.__file__)),
                         data_identity(Path(__file__).with_name("sparse_ply_oracle.py"))],
              "injected_import_failure_is_real_adapter_failure_not_bad_PLY_fixture": True,
              "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "checked": len(checks)}))
    if not all(checks.values()):
        raise AssertionError("Sparse PLY oracle contract failed")


if __name__ == "__main__":
    main()
