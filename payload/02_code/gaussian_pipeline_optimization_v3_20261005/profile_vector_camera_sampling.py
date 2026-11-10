"""Screen the complete read/undistort/JPEG/close consumer on 32 fixed real frames."""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

import numpy as np
from PIL import Image
import pycolmap

from probe_dataset_sampling_route import ROOT, EXPORT, IMAGES
from dataset_sampling_carrier import standard_raw_jpeg
from camera_pixel_remap import camera_fields
from vector_camera_sampling import VectorCameraPixelRemap, library_supports_simd
from runtime import atomic_json, data_identity

OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/ud001/v5_simd_001"


def jpeg(path):
    with Image.open(path) as image:
        return np.asarray(image).copy(), dict(image.info), dict(image.getexif())


def same_rng(before, after):
    return before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OWN)
    output.mkdir(parents=True, exist_ok=False)
    model = pycolmap.Reconstruction(EXPORT / "training_opencv_model")
    if (model.num_reg_images(), model.num_points3D(), model.compute_num_observations(), len(model.cameras)) != (1016, 75767, 959120, 1):
        raise ValueError("The fixed original full-list source changed")
    names = sorted(image.name for image in model.images.values() if image.has_pose)
    selected = [names[index] for index in np.linspace(0, len(names) - 1, 32, dtype=int)]
    camera = next(iter(model.cameras.values()))
    options = pycolmap.UndistortCameraOptions()
    code = Path(__file__).resolve().parent
    sources = [Path(__file__), *[code / name for name in ("vector_camera_sampling.py", "native_vector_camera_sampling.cpp",
        "native_camera_sampling.py", "camera_pixel_remap.py", "dataset_sampling_carrier.py", "probe_dataset_sampling_route.py")],
        ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005/runtime.py", args.library.resolve()]
    inputs = [EXPORT / "TRAINING_ADMISSION.json", *sorted((EXPORT / "training_opencv_model").glob("*.bin")),
              *[IMAGES / name for name in selected]]
    source_ids, input_ids = [data_identity(p) for p in sources], [data_identity(p) for p in inputs]
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    for index, path in enumerate(sources):
        if path.suffix in (".py", ".cpp"):
            shutil.copyfile(path, snapshots / (f"{index:02d}_" + path.name))
    atomic_json(output / "CONTRACT.json", {"sources": source_ids, "inputs": input_ids, "all1016_sorted_names": names,
        "fixed32_equal_interval_selected_names": selected, "subset_not_full1050_or_formal": True,
        "raw_screen_minimum_stage_reduction": .03, "screen_maximum_whole_ratio": 1.1,
        "actual_ISA_admitted": library_supports_simd(args.library), "new_optimization_count": 0})
    setup_seconds = time.perf_counter() - started
    original = pycolmap.undistort_image
    rng = np.random.get_state()
    pairs = []
    for trial in range(3):
        rows = {}
        for mode in (("native", "vector") if trial % 2 == 0 else ("vector", "native")):
            side = output / ("trial" + str(trial)) / mode
            side.mkdir(parents=True, exist_ok=False)
            before = time.perf_counter()
            operator = None
            admitted = fallbacks = 0
            calibrations = []
            try:
                if mode == "vector": operator = VectorCameraPixelRemap(original, args.library)
                for name in selected:
                    bitmap = pycolmap.Bitmap.read(IMAGES / name, as_rgb=True)
                    if bitmap is None:
                        raise OSError("The unchanged original Bitmap reader failed")
                    if operator is not None and standard_raw_jpeg(IMAGES / name, (camera.width, camera.height)):
                        warped, returned = operator(options, bitmap, camera)
                        admitted += 1
                    else:
                        warped, returned = original(options, bitmap, camera)
                        fallbacks += int(operator is not None)
                    calibrations.append(camera_fields(returned))
                    warped.set_jpeg_quality(97)
                    if not warped.write(side / name):
                        raise OSError("The unchanged original JPEG97 writer failed")
            finally:
                if operator is not None: operator.close()
            operation_seconds = time.perf_counter() - before
            verify_start = time.perf_counter()
            if source_ids != [data_identity(p) for p in sources] or input_ids != [data_identity(p) for p in inputs]:
                raise ValueError("An original32 source or input changed")
            if pycolmap.undistort_image is not original or not same_rng(rng, np.random.get_state()):
                raise ValueError("An original callback or RNG changed")
            if mode == "vector" and not (admitted == 32 and fallbacks == 0 and operator.trace["builds"] == 1
                and operator.trace["sampler_calls"] == 32 and operator.trace["actual_vector_ISA_admitted"]
                and operator.trace["native_DLL_closed"] and operator.closed):
                raise AssertionError("The actual vector route or complete work changed")
            verified_seconds = time.perf_counter() - verify_start
            rows[mode] = {"operation_seconds": operation_seconds, "phase_seconds": operation_seconds + verified_seconds + setup_seconds / 6,
                "whole_seconds": time.perf_counter() - before + setup_seconds / 6,
                "calibrations": calibrations, "trace": operator.trace if operator is not None else None,
                "selected_count": len(selected), "fallbacks": fallbacks}
        compare_start = time.perf_counter()
        if rows["native"]["calibrations"] != rows["vector"]["calibrations"]:
            raise AssertionError("All32 output intrinsics differ")
        comparisons = []
        for name in selected:
            a = output / ("trial" + str(trial)) / "native" / name
            b = output / ("trial" + str(trial)) / "vector" / name
            ap, am, ae = jpeg(a)
            bp, bm, be = jpeg(b)
            if a.read_bytes() != b.read_bytes() or not np.array_equal(ap, bp) or am != bm or ae != be:
                raise AssertionError("Full original JPEG bytes, pixels or metadata differ: " + name)
            comparisons.append({"name": name, "all_JPEG_bytes_pixels_metadata_exact": True, "bytes": a.stat().st_size})
        compare_seconds = time.perf_counter() - compare_start
        for row in rows.values():
            row["phase_seconds"] += compare_seconds
            row["whole_seconds"] += compare_seconds
        pairs.append({"trial": trial, "rows": rows, "all32_comparisons": comparisons,
                      "actual_pair_comparison_seconds_charged_each_side": compare_seconds})
        atomic_json(output / "STATE.json", {"status": "component32_three_pairs_measuring_not_acceptance", "pairs": pairs,
                                           "new_optimization_count": 0})
    medians = {mode: {key: statistics.median(pair["rows"][mode][key] for pair in pairs)
                     for key in ("phase_seconds", "whole_seconds", "operation_seconds")} for mode in ("native", "vector")}
    phase_ratio = medians["vector"]["phase_seconds"] / medians["native"]["phase_seconds"]
    whole_ratio = medians["vector"]["whole_seconds"] / medians["native"]["whole_seconds"]
    passes = phase_ratio <= .97 and whole_ratio <= 1.1
    checks = {"all_three_pairs_fixed32_K_full_JPEG_metadata_and_pixels_exact": len(pairs) == 3
        and all(len(pair["all32_comparisons"]) == 32 for pair in pairs),
        "all_vector_calls_complete_and_closed": all(pair["rows"]["vector"]["selected_count"] == 32 for pair in pairs),
        "callback_and_RNG_unchanged": pycolmap.undistort_image is original and same_rng(rng, np.random.get_state()),
        "source_input_identities_unchanged": source_ids == [data_identity(p) for p in sources]
            and input_ids == [data_identity(p) for p in inputs],
        "private_output4GiB_and_free20GiB": sum(p.stat().st_size for p in OWN.rglob("*") if p.is_file()) <= 4 * 1024**3
            and shutil.disk_usage(ROOT).free >= 20 * 1024**3}
    if not all(checks.values()): raise AssertionError("A complete32 protection failed")
    atomic_json(output / "RESULT.json", {"status": "component32_screen_passed_full1050_pending" if passes else "component32_cost_negative_full1050_not_run",
        "checks": checks, "pairs": pairs, "medians": medians, "raw_complete_phase_reduction": 1 - phase_ratio,
        "raw_complete_whole_reduction": 1 - whole_ratio, "old_family_fee_not_yet_applied_not_acceptance": True,
        "continue_to_full_consumer": passes, "complete_seconds": time.perf_counter() - started,
        "new_optimization_count": 0})
    print(json.dumps({"medians": medians, "stage_reduction": 1 - phase_ratio, "whole_reduction": 1 - whole_ratio,
                      "continue_to_full_consumer": passes, "new_optimization_count": 0}))


if __name__ == "__main__":
    main()
