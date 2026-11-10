"""Screen read-origin JPEG preservation and the native partial-image carrier."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
import psutil
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from camera_pixel_remap import camera_fields
from native_camera_sampling import NativeCameraPixelRemap

OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/ud001"
V1OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005"
EXPORT = V1OUT / "new_map_export_003"
IMAGES = V1OUT / "e2_geometry_support_001/images"
LIBRARY = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_native001/b1/native_camera_sampling.dll"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jpeg(path):
    with Image.open(path) as image:
        pixels = np.array(image)
        metadata = dict(image.info)
        exif = dict(image.getexif())
    return pixels, metadata, exif


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OWN.resolve())
    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise OSError("The original disk headroom is unavailable")
    owner = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": owner.pid, "creation_time": owner.create_time()})
    native_undistort, native_dataset = pycolmap.undistort_image, pycolmap.undistort_images
    rng = np.random.get_state()
    operator = None
    result = {"status": "starting", "new_optimization_count": 0, "rows": [], "candidate_checks": {}}
    try:
        receipt = read(EXPORT / "TRAINING_ADMISSION.json")
        if receipt["status"] != "export_and_admission_verified_training_pending":
            raise ValueError("The original dataset is not admitted")
        model_path = Path(receipt["training_opencv_model"])
        dataset = Path(receipt["dataset"])
        model = pycolmap.Reconstruction(model_path)
        lookup = {image.name: image for image in model.images.values() if image.has_pose}
        names = sorted(lookup)
        if (len(names), model.num_points3D(), model.compute_num_observations()) != (1016, 75767, 959120):
            raise ValueError("The complete original native model denominator changed")
        selected = [names[index] for index in (0, len(names) // 2, len(names) - 1)]
        sources = [Path(__file__), CODE / "native_camera_sampling.py", CODE / "native_camera_sampling.cpp",
                   CODE / "camera_pixel_remap.py", V1 / "runtime.py", LIBRARY]
        build = read(LIBRARY.parent / "RESULT.json")
        if data_identity(LIBRARY) != build["DLL"]:
            raise ValueError("The source-pinned sampling DLL changed")
        inputs = [EXPORT / "TRAINING_ADMISSION.json", *sorted(model_path.glob("*.bin")),
                  *sorted((dataset / "sparse").glob("*.bin")),
                  *[path for name in selected for path in (IMAGES / name, dataset / "images" / name)]]
        source_ids, input_ids = [data_identity(path) for path in sources], [data_identity(path) for path in inputs]
        snapshots = output / "source_snapshot"
        snapshots.mkdir()
        for index, path in enumerate(sources):
            if path.suffix == ".py" or path.suffix == ".cpp":
                shutil.copyfile(path, snapshots / (f"{index:02d}_" + path.name))
        options = pycolmap.UndistortCameraOptions()
        atomic_json(output / "CONTRACT.json", {"sources": source_ids, "inputs": input_ids,
                    "original_options": options.todict(), "selection": "first_middle_last_of_all1016_sorted_registered_names",
                    "selected": selected, "JPEG_quality": 97, "no_new_optimization_or_formal_pair": True})
        operator = NativeCameraPixelRemap(native_undistort, LIBRARY)
        for index, name in enumerate(selected):
            image = lookup[name]
            camera = model.cameras[image.camera_id]
            bitmap = pycolmap.Bitmap.read(IMAGES / name, as_rgb=True)
            if bitmap is None:
                raise OSError("The original Bitmap reader rejected an input image")
            before = time.perf_counter()
            first, first_camera = native_undistort(options, bitmap, camera)
            native_seconds = time.perf_counter() - before
            before = time.perf_counter()
            second, second_camera = operator(options, bitmap, camera)
            candidate_seconds = time.perf_counter() - before
            paths = [output / (str(index) + "_" + mode + ".jpg") for mode in ("native", "dll")]
            for value, path in zip((first, second), paths):
                value.set_jpeg_quality(97)
                if not value.write(path):
                    raise OSError("The original JPEG writer rejected an owned output")
            left, right, original = (jpeg(path) for path in (*paths, dataset / "images" / name))
            checks = {"all_warp_pixels_exact": np.array_equal(first.to_array(), second.to_array()),
                      "all_camera_fields_exact": camera_fields(first_camera) == camera_fields(second_camera),
                      "native_and_DLL_complete_JPEG_bytes_exact": paths[0].read_bytes() == paths[1].read_bytes(),
                      "native_and_DLL_JPEG_decoded_pixels_exact": np.array_equal(left[0], right[0]),
                      "native_and_DLL_JPEG_metadata_exact": left[1:] == right[1:],
                      "native_saved_JPEG_matches_original_full_dataset_bytes": paths[0].read_bytes() == (dataset / "images" / name).read_bytes(),
                      "native_saved_JPEG_matches_original_full_dataset_pixels": np.array_equal(left[0], original[0]),
                      "native_saved_JPEG_matches_original_full_dataset_metadata": left[1:] == original[1:]}
            row = {"image_name": name, "checks": checks, "native_sample_seconds": native_seconds,
                   "DLL_sample_seconds_including_cold_build": candidate_seconds,
                   "native_metadata_keys": sorted(left[1]), "DLL_metadata_keys": sorted(right[1]),
                   "source_EXIF": {key: getattr(bitmap, key)() for key in ("exif_orientation", "exif_camera_model", "exif_focal_length")},
                   "native_JPEG_bytes": paths[0].stat().st_size, "DLL_JPEG_bytes": paths[1].stat().st_size}
            result["rows"].append(row)
        result["candidate_checks"]["three_read_origin_complete_pixels_camera_JPEG_metadata_and_original_files_exact"] = all(all(row["checks"].values()) for row in result["rows"])
        carrier = output / "carrier"
        before = time.perf_counter()
        native_dataset(carrier, model_path, IMAGES, image_names=selected[:1], output_type="COLMAP",
                       num_patch_match_src_images=0, jpeg_quality=97, num_threads=8)
        result["carrier_seconds"] = time.perf_counter() - before
        reread = pycolmap.Reconstruction(carrier / "sparse")
        dimensions = (reread.num_reg_images(), reread.num_points3D(), reread.compute_num_observations())
        result["carrier_actual_denominators"] = dimensions
        binaries = [path.name for path in sorted((dataset / "sparse").glob("*.bin"))]
        binary_checks = {name: (carrier / "sparse" / name).is_file() and
                         (carrier / "sparse" / name).read_bytes() == (dataset / "sparse" / name).read_bytes()
                         for name in binaries}
        structure = [path.relative_to(dataset).as_posix() for path in dataset.rglob("*")
                     if path.is_file() and not path.relative_to(dataset).as_posix().startswith(("images/", "sparse/"))]
        structure_checks = {name: (carrier / name).is_file() and (carrier / name).read_bytes() == (dataset / name).read_bytes()
                            for name in structure}
        result.update(carrier_original_sparse_byte_checks=binary_checks, carrier_original_structure_byte_checks=structure_checks,
                      carrier_saved_image_names=sorted(path.name for path in (carrier / "images").iterdir() if path.is_file()))
        result["candidate_checks"].update(carrier_all1016_75767_959120_and_all_native_sparse_bytes_preserved=
                                             dimensions == (1016, 75767, 959120) and bool(binaries) and all(binary_checks.values()),
                                         carrier_all_original_structure_bytes_preserved=bool(structure) and all(structure_checks.values()),
                                         carrier_requested_one_original_JPEG_exact=result["carrier_saved_image_names"] == selected[:1] and
                                             (carrier / "images" / selected[0]).read_bytes() == (dataset / "images" / selected[0]).read_bytes())
        result["protection_checks"] = {"all_source_and_input_identities_unchanged": source_ids == [data_identity(path) for path in sources]
                                       and input_ids == [data_identity(path) for path in inputs],
                                       "original_native_functions_unchanged": pycolmap.undistort_image is native_undistort and pycolmap.undistort_images is native_dataset,
                                       "numpy_RNG_unchanged": rng[0] == np.random.get_state()[0] and np.array_equal(rng[1], np.random.get_state()[1])
                                       and rng[2:] == np.random.get_state()[2:],
                                       "owned_output_under1GiB_and_original_disk_headroom": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) < 1024**3
                                       and shutil.disk_usage(ROOT).free >= 20 * 1024**3}
        if not all(result["protection_checks"].values()):
            raise AssertionError("A bounded dataset-route protection check failed")
        result["status"] = ("candidate_interface_screen_passed_full1016_cost_quality_recovery_unverified"
                            if all(result["candidate_checks"].values()) else "screen_rejected_pixel_JPEG_metadata_or_native_carrier_complete_contract")
    except BaseException as error:
        result.update(status="actual_probe_error_no_acceptance", error_type=type(error).__name__, error=str(error))
        atomic_json(output / "FAILURE.json", result)
        raise
    finally:
        if operator is not None:
            operator.close()
            result["sampler_trace"] = operator.trace
        result["seconds_before_receipt"] = time.perf_counter() - started
        atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "candidate_checks": result["candidate_checks"],
                      "three_image_checks": [row["checks"] for row in result["rows"]], "seconds": result["seconds_before_receipt"],
                      "carrier_actual_denominators": result.get("carrier_actual_denominators"), "new_optimization_count": 0}))


if __name__ == "__main__":
    main()
