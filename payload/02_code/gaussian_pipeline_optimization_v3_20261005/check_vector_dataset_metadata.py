"""Verify real read-origin metadata abstention against complete native datasets."""

import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageCms
import pycolmap

from profile_vector_full_export import ROOT, BASE, LIBRARY
from probe_dataset_sampling_route import EXPORT, IMAGES
from dataset_sampling_shared_carrier import cached_dataset
from dataset_sampling_carrier import standard_raw_jpeg
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    output = BASE / "m0"
    output.mkdir(parents=True, exist_ok=False)
    full = pycolmap.Reconstruction(EXPORT / "training_opencv_model")
    registered = sorted((image for image in full.images.values() if image.has_pose), key=lambda image: image.name)
    names = [registered[index].name for index in (0, len(registered) // 2, len(registered) - 1)]
    keep = set(names)
    for image in list(full.images.values()):
        if image.has_pose and image.name not in keep:
            full.deregister_frame(image.frame_id)
    if full.num_reg_images() != 3:
        raise ValueError("The derived metadata fixture changed its registered denominator")
    model = output / "model"
    model.mkdir()
    full.write(model)
    source_paths = [Path(__file__), Path(__file__).with_name("dataset_sampling_shared_carrier.py"),
        Path(__file__).with_name("dataset_sampling_carrier.py"), LIBRARY]
    ids = [data_identity(p) for p in source_paths]
    original = pycolmap.undistort_images
    before_rng = np.random.get_state()
    rows = []
    for kind in ("exif", "dpi", "icc"):
        case = output / kind
        pictures = case / "inputs"
        pictures.mkdir(parents=True)
        exif = Image.Exif()
        exif[270] = "metadata abstention fixture"
        for name in names:
            with Image.open(IMAGES / name) as image:
                options = {"exif": exif} if kind == "exif" else ({"dpi": (72, 72)} if kind == "dpi" else
                    {"icc_profile": ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()})
                image.convert("RGB").save(pictures / name, format="JPEG", quality=97, **options)
        camera = next(iter(full.cameras.values()))
        if any(standard_raw_jpeg(pictures / name, (camera.width, camera.height)) for name in names):
            raise AssertionError("The metadata fixture did not exercise actual nonadmission")
        input_paths = [*sorted(model.glob("*.bin")), *[pictures / name for name in names]]
        input_ids = [data_identity(p) for p in input_paths]
        native = case / "native"
        current = case / "current"
        original(native, model, pictures, image_names=names, output_type="COLMAP",
                 num_patch_match_src_images=0, jpeg_quality=97, num_threads=8)
        trace = cached_dataset(current, model, pictures, names, LIBRARY, case / "proof")
        first = {p.relative_to(native).as_posix(): p for p in native.rglob("*") if p.is_file()}
        second = {p.relative_to(current).as_posix(): p for p in current.rglob("*") if p.is_file()}
        if first.keys() != second.keys() or any(first[k].read_bytes() != second[k].read_bytes() for k in first):
            raise AssertionError("A complete fallback dataset changed its native output bytes")
        if trace["route"] != "original_before_any_cached_dataset_write" or not trace["closed"]:
            raise AssertionError("Metadata abstention occurred after candidate writes")
        if input_ids != [data_identity(p) for p in input_paths]:
            raise AssertionError("A frozen metadata fixture changed")
        rows.append({"kind": kind, "three_real_read_origin_images": names, "all_complete_native_files_exact": len(first),
                     "route": trace["route"], "source_input_unchanged": True})
    after_rng = np.random.get_state()
    checks = {"three_EXIF_DPI_ICC_complete_native_abstentions_byte_exact": len(rows) == 3,
        "actual_original_callback_and_RNG_restored": pycolmap.undistort_images is original
            and before_rng[0] == after_rng[0] and np.array_equal(before_rng[1], after_rng[1]) and before_rng[2:] == after_rng[2:],
        "all_executed_source_identities_unchanged": ids == [data_identity(p) for p in source_paths]}
    if not all(checks.values()): raise AssertionError("A metadata fallback protection failed")
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    for p in source_paths:
        if p.suffix == ".py": shutil.copyfile(p, snapshots / p.name)
    atomic_json(output / "RESULT.json", {"status": "real_EXIF_DPI_ICC_complete_dataset_original_fallback_verified",
        "checks": checks, "rows": rows, "sources": ids, "complete_seconds": time.perf_counter() - started,
        "derived3_camera_model_not_external_full_pipeline_GT": True, "new_optimization_count": 0})
    print(json.dumps({"status": "metadata_original_fallback_verified", "checks": checks, "new_optimization_count": 0}))


if __name__ == "__main__":
    main()
