"""Run original undistortion and sparse artifact writers on one private disk map."""

import time
STARTED = time.perf_counter()

import argparse
import json
from pathlib import Path
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
import export_admitted_map as native
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8-sig"))
    output = args.contract.parent
    output.resolve().relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/export_qa_dag_001")
    dataset, filtered, images = (Path(contract[key]) for key in ("dataset", "filtered", "images"))
    dataset.resolve().relative_to(output.parent / "export")
    filtered.resolve().relative_to(output.parent / "export")
    owner = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": owner.pid, "creation_time": owner.create_time()})
    calls = []
    try:
        if contract["sources"] != [data_identity(row["path"]) for row in contract["sources"]]:
            raise ValueError("The original native worker source changed")
        if contract["map_inputs"] != [data_identity(row["path"]) for row in contract["map_inputs"]]:
            raise ValueError("The complete private filtered map changed before undistortion")
        before = time.perf_counter()
        native.pycolmap.undistort_images(dataset, filtered, images, image_names=contract["image_names"],
            output_type="COLMAP", num_patch_match_src_images=0, jpeg_quality=97, num_threads=8)
        calls.append({"stage": "original_undistort", "seconds": time.perf_counter() - before})
        atomic_json(output / "UNDISTORT_READY.json", {"status": "original_undistort_returned", "calls": calls,
            "complete_worker_before_ready_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
        before = time.perf_counter()
        pinhole = native.pycolmap.Reconstruction(dataset / "sparse")
        sparse = dataset / "sparse/0"
        sparse.mkdir()
        pinhole.write(sparse)
        pinhole.write_text(sparse)
        native.write_initial_ply(pinhole, sparse / "points3D.ply")
        calls.append({"stage": "original_sparse_write_and_PLY_roundtrip", "seconds": time.perf_counter() - before})
        if contract["map_inputs"] != [data_identity(row["path"]) for row in contract["map_inputs"]]:
            raise ValueError("The complete private filtered map changed during artifact export")
        if contract["sources"] != [data_identity(row["path"]) for row in contract["sources"]]:
            raise ValueError("The original native worker source changed during export")
        atomic_json(output / "RESULT.json", {"status": "complete_original_undistort_and_sparse_artifacts", "calls": calls,
            "registered": pinhole.num_reg_images(), "points": pinhole.num_points3D(), "observations": pinhole.compute_num_observations(),
            "source_and_private_map_unchanged": True, "complete_worker_before_receipt_seconds": time.perf_counter() - STARTED,
            "new_optimization_count": 0})
    except BaseException as error:
        atomic_json(output / "FAILURE.json", {"status": "failed_original_worker_no_admission", "error_type": type(error).__name__,
            "error": str(error), "repr": repr(error), "calls": calls, "complete_seconds": time.perf_counter() - STARTED,
            "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()
