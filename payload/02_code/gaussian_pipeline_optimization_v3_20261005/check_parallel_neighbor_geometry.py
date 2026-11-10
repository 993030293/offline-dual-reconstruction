"""Check ordered native geometry and caller RNG parity for heterogeneous fixtures."""

import argparse
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
from observation_index import indexed_match
from parallel_neighbor_geometry import ParallelNeighborGeometry, parallel_neighbor_geometry


def fixture(count, scale, aspect, shape, neighbors):
    rng = np.random.default_rng(19+count)
    cloud = rng.uniform(-1, 1, (max(count, 1), 3))
    if shape == "curved":
        cloud[:, 2] += np.sin(cloud[:, 0]*2)
    elif shape == "thin":
        cloud[:, 1] *= .03
    else:
        cloud[:, 2] += cloud[:, 0]**2
    cloud[:, 2] += 5
    cloud *= scale
    focal = np.asarray([713, 713*aspect])
    def project(angle, translation):
        rotated = cloud @ Rotation.from_euler("y", angle).as_matrix().T+np.asarray(translation)*scale
        return (rotated[:, :2]/rotated[:, 2:]*focal+np.asarray([503, 417]))[:count].astype(np.float32)
    query = np.c_[project(0, [0, 0, 0]), np.ones((count, 2), np.float32)]
    descriptions = rng.integers(0, 256, (count, 128), dtype=np.uint8).astype(np.float32)/512
    original, arrays = {}, {}
    for i in range(neighbors):
        pixels = project(.025*i, [.15+i*.1, .01*i, .02*i])
        if count > 30:
            pixels[-7:] += rng.normal(0, 50, (7, 2))
        points = np.c_[pixels, np.ones((count, 2), np.float32)]
        original[i+1] = (f"generic_{i}.png", points, descriptions.copy())
        ids = np.arange(count, dtype=np.int64) % max(1, count-5)
        ids[::13] = -1
        arrays[i+1] = points[:, :2].astype(np.float64), ids, ids >= 0
    index = SimpleNamespace(get=lambda model, image_id: arrays[image_id])
    return SimpleNamespace(original=original), object(), query, descriptions, list(original), index


def rng_probe():
    probe = np.empty((17, 2), np.float64)
    cv2.randu(probe, 0., 1.)
    return probe.copy()


def invoke(function, values, geometric):
    cv2.setRNGSeed(7129)
    try:
        result, error = function(*values, geometric=geometric), None
    except Exception as failure:
        result, error = None, (type(failure).__name__, str(failure))
    return result, error, rng_probe()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if cv2.__version__ != "4.13.0":
        raise RuntimeError("Use the existing Windows OpenCV4.13.0 geometry runtime")
    scheduler = ParallelNeighborGeometry(indexed_match, True)
    rows = []
    try:
        for shape in ("curved", "thin", "irregular"):
            for scale, aspect in ((.01, .7), (1., 1.), (100., 1.9)):
                for count, neighbors in ((0, 3), (7, 3), (15, 3), (89, 1), (89, 4)):
                    values = list(fixture(count, scale, aspect, shape, neighbors))
                    for dtype in (np.float32, np.float64):
                        values[2] = values[2].astype(dtype)
                        for geometric in (False, True):
                            native = invoke(indexed_match, values, geometric)
                            candidate = invoke(scheduler, values, geometric)
                            rows.append({"shape": shape, "scale": scale, "aspect": aspect, "count": count,
                                "neighbors": neighbors, "query_dtype": str(values[2].dtype), "geometric": geometric,
                                "outputs_exact": native[:2] == candidate[:2],
                                "caller_RNG_exact": np.array_equal(native[2], candidate[2])})
    finally:
        scheduler.close()
    checks = {"all180_heterogeneous_empty_insufficient_degenerate_and_nongeometric_outputs_exact": len(rows) == 180
        and all(row["outputs_exact"] for row in rows),
        "all180_caller_RNG_states_preserved": len(rows) == 180 and all(row["caller_RNG_exact"] for row in rows),
        "bounded_queue_and_clean_ordinary_close": scheduler.trace["peak_pending"] <= 11 and scheduler.closed}
    limited = ParallelNeighborGeometry(indexed_match, True, maximum_array_bytes=64)
    try:
        values = fixture(89, 1, 1, "irregular", 4)
        native, candidate = invoke(indexed_match, values, True), invoke(limited, values, True)
        checks["geometry_memory_preadmission_uses_original_without_feature_deletion"] = (native[:2] == candidate[:2]
            and np.array_equal(native[2], candidate[2]) and limited.trace["routes"][-1]["route"] == "original_before_execution")
    finally:
        limited.close()
    values = fixture(15, 1, 1, "irregular", 13)
    bounded = ParallelNeighborGeometry(indexed_match, True)
    try:
        native, candidate = invoke(indexed_match, values, True), invoke(bounded, values, True)
        checks["over_neighbor_capacity_preselects_original_all_records"] = native[:2] == candidate[:2] and bounded.trace["routes"][-1]["route"] == "original_before_execution"
    finally:
        bounded.close()
    fault = ParallelNeighborGeometry(indexed_match, True)
    def injected(xy1, xy2):
        raise RuntimeError("owned_neighbor_failure")
    fault.fit = injected
    try:
        fault(*fixture(89, 1, 1, "curved", 4), geometric=True)
    except RuntimeError as error:
        checks["owned_worker_fault_propagates_without_retry"] = str(error) == "owned_neighbor_failure"
    else:
        checks["owned_worker_fault_propagates_without_retry"] = False
    finally:
        fault.close()
    import native_observation_index
    original = native_observation_index.indexed_match
    try:
        with parallel_neighbor_geometry("parallel"):
            raise RuntimeError("owned_context_failure")
    except RuntimeError as error:
        checks["global_indexed_geometry_restored_after_error"] = str(error) == "owned_context_failure" and native_observation_index.indexed_match is original
    checks["all_owned_threads_closed_after_fault_and_normal_paths"] = not any(t.name.startswith("owned-neighbor-F") for t in threading.enumerate())
    result = {"status": "passed" if all(checks.values()) else "failed_verification", "checks": checks,
        "rows": rows, "trace": scheduler.trace, "counted": False, "synthetic_contract_not_new_object_quality": True,
        "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("parallel_neighbor_geometry.py")),
            data_identity(Path(indexed_match.__code__.co_filename))], "opencv": cv2.__version__}
    atomic_json(args.output/"RESULT.json", result)
    print({key: result[key] for key in ("status", "checks", "opencv")}, flush=True)
    if not all(checks.values()):
        raise AssertionError("An original neighbor geometry or caller RNG guard failed")


if __name__ == "__main__":
    main()
