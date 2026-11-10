"""Verify the two real augmented-schema failures before resuming any BA."""

import argparse
from pathlib import Path

import pycolmap

from common import configure_stdout, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    run = args.run
    model = pycolmap.Reconstruction(run / "03_coverage_augmented/registered_model")
    rows = read_json(run / "01_priors/camera_prior_manifest.json")["images"]
    if not all(isinstance(r["quality_sharpness"], (int, float)) for r in rows):
        raise ValueError("Existing prior quality values do not match the schema")
    image_id = min(i for i in model.images if i > 180)
    image = model.images[image_id]
    path = run / f"diagnostics/frame_write_order_regression_attempt{args.attempt}.db"
    if path.exists():
        raise FileExistsError(path)
    with pycolmap.Database.open(path) as database:
        database.write_camera(model.cameras[image.camera_id], use_camera_id=True)
        database.write_rig(model.rigs[image.camera_id], use_rig_id=True)
        rejected = False
        try:
            database.write_image(image, use_image_id=True)
        except ValueError as error:
            if "frame.HasDataId" not in str(error):
                raise
            rejected = True
        if not rejected:
            raise RuntimeError("The wrong-order regression did not reproduce the actual failure")
        actual_frame = database.write_frame(model.frames[image.frame_id], use_frame_id=True)
        actual_image = database.write_image(image, use_image_id=True)
        restored = database.read_image(actual_image)
        restored_frame = database.read_frame(actual_frame)
        if actual_image != image_id or actual_frame != image.frame_id or restored.name != image.name:
            raise RuntimeError("The corrected database order changed frame or image identity")
        if image.data_id not in restored_frame.data_ids:
            raise RuntimeError("The corrected frame does not contain the real camera data ID")
    write_json(run / "diagnostics/AUGMENTED_CONTRACT_TESTS.json", {"status": "passed", "checks": [
        "existing_quality_sharpness_schema", "wrong_frame_order_reproduces_failure", "frame_before_image_readback_identity"],
        "source_version": "installed_pycolmap_4.1.1",
        "source_evidence": "https://github.com/colmap/colmap/blob/a0d785fba74b2664f31edc4a29026a8b27c00f67/src/colmap/scene/database_sqlite.cc#L1080"})
    print("Augmented quality schema and database write-order regression passed")


if __name__ == "__main__":
    configure_stdout()
    main()
