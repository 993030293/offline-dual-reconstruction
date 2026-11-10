"""Compare actual training metadata and reject malformed binary camera inputs."""

import argparse
import collections
import importlib.util
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from training_camera_metadata import metadata_images, IMAGE_HEADER, COUNT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "dataset", "runtime"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    loader_path = args.runtime/"source/gaussian-splatting/scene/colmap_loader.py"
    specification = importlib.util.spec_from_file_location("isolated_metadata_oracle", loader_path)
    oracle = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(oracle)
    source = args.dataset/"sparse/0/images.bin"
    original = oracle.read_extrinsics_binary(source)
    candidate, receipt = metadata_images(source, oracle.Image)
    checks = {"all1016_actual_cameras": len(original) == len(candidate) == 1016,
        "actual_camera_order_and_ids_exact": list(original) == list(candidate),
        "all_binary_observation_rows_accounted_for": sum(len(r.xys) for r in original.values()) == receipt["observation_rows_not_allocated"],
        "all_metadata_fields_and_rotation_values_exact": all(a.id == b.id and a.camera_id == b.camera_id and a.name == b.name and
            np.array_equal(a.qvec, b.qvec) and np.array_equal(a.tvec, b.tvec) and np.array_equal(a.qvec2rotmat(), b.qvec2rotmat())
            for a, b in zip(original.values(), candidate.values())),
        "training_view_has_no_fake_observation_arrays": all(r.xys is None and r.point3D_ids is None for r in candidate.values())}
    header = IMAGE_HEADER.pack(1, 1., 0., 0., 0., 0., 0., 0., 1)
    good = COUNT.pack(1)+header+b"fixture.jpg\0"+COUNT.pack(1)+struct.pack("<ddq", 1., 2., -1)
    malformed = {"truncated_header": good[:40], "truncated_observations": good[:-1],
        "trailing_unclaimed_bytes": good+b"x", "duplicate_image_identity": COUNT.pack(2)+good[8:]+good[8:],
        "bad_image_identity": COUNT.pack(1)+IMAGE_HEADER.pack(-1, 1., 0., 0., 0., 0., 0., 0., 1)+good[72:],
        "escaping_image_name": COUNT.pack(1)+header+b"../outside.jpg\0"+COUNT.pack(0),
        "missing_filename_terminator": COUNT.pack(1)+header+b"fixture.jpg",
        "impossible_camera_count": COUNT.pack(10000000)+good[8:], "zero_camera_count": COUNT.pack(0)}
    fixtures = args.output/"fixtures"
    fixtures.mkdir()
    for name, raw in malformed.items():
        path = fixtures/(name+".bin")
        path.write_bytes(raw)
        try:
            metadata_images(path, oracle.Image)
            checks[name+"_rejected"] = False
        except (ValueError, UnicodeError):
            checks[name+"_rejected"] = True
    if checks["bad_image_identity_rejected"] is False:
        raise AssertionError("An invalid image identity passed the metadata reader")
    paths = [Path(__file__), Path(__file__).with_name("training_camera_metadata.py"), loader_path,
        args.runtime/"source/gaussian-splatting/scene/dataset_readers.py"]
    result = {"status": "lifecycle_passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "actual_metadata": receipt, "sources": [data_identity(p) for p in paths],
        "input": data_identity(source), "counted": False, "full_scene_and_training_quality_pending": True}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": checks, "actual_metadata": receipt}), flush=True)
    if not all(checks.values()):
        raise ValueError("The actual training metadata lifecycle did not pass")


if __name__ == "__main__":
    main()
