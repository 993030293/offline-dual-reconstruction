"""Screen direct native RGB decoding on fixed real frames without claiming Scene gains."""

import time

STARTED = time.perf_counter()

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import shutil
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2), str(CODE / "parallel_b")]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard


def main():
    output = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/native_rgb_decode_001/p0"
    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output).free < 20 * 1024**3:
        raise RuntimeError("The original free-disk reserve is unavailable")
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"scope": "WSL", "pid": process.pid,
        "creation_time": process.create_time(), "diagnostic_only": True})
    with LinuxRSSGuard(output, atomic_json, maximum_seconds=120, maximum_bytes=8 * 1024**3) as guard:
        reference = activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from utils.general_utils import PILtoTorch
        from opencv_decode import OpenCVPreparation
        torch.set_num_threads(4)
        if cv2.__version__ != "4.11.0" or not hasattr(cv2, "IMREAD_COLOR_RGB"):
            raise RuntimeError("The current existing native RGB decoder interface is unavailable")
        paths = sorted((ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003/dataset/images").glob("*.jpg"))
        if len(paths) != 1016:
            raise AssertionError("The current full training input count changed")
        indices = [position * (len(paths) - 1) // 11 for position in range(12)]
        selected = [paths[index] for index in indices]
        sources = [Path(__file__), CODE / "parallel_b/opencv_decode.py", V2 / "prefetch_camera_images.py",
            V1 / "runtime.py", CODE / "mapped_image_linux_guard.py",
            Path(sys.modules[PILtoTorch.__module__].__file__), Path(cv2.__file__)]
        identities = [data_identity(path) for path in sources + selected]
        copies = output / "source_copies"
        copies.mkdir()
        for index, path in enumerate(sources):
            shutil.copyfile(path, copies / (f"{index:02d}_" + path.name))
        atomic_json(output / "CONTRACT.json", {"reference": reference, "sources_inputs": identities,
            "full_training_count": 1016, "diagnostic_indices_before_results": indices,
            "workers": 4, "capacity": 8, "orders": [["pillow", "old_opencv", "direct_rgb"],
                ["old_opencv", "direct_rgb", "pillow"], ["direct_rgb", "pillow", "old_opencv"]],
            "no_new_mechanism_or_acceptance_for_backend_variant": True,
            "old_nonoverlapping_PB03_first_cost_seconds": 162.98888750001788,
            "counted": False})
        raw_checks = []
        for path in selected:
            with Image.open(path) as image:
                if image.format != "JPEG" or image.mode != "RGB":
                    raise AssertionError("The preselected current JPEG scope changed")
                baseline = np.asarray(image).copy()
                shape = (image.height, image.width, 3)
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            direct = cv2.imread(str(path), cv2.IMREAD_COLOR_RGB | cv2.IMREAD_IGNORE_ORIENTATION)
            checks = {"original_full_decoded_pixels": np.array_equal(baseline, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                and np.array_equal(baseline, direct), "original_shape_dtype": direct.shape == shape and direct.dtype == np.uint8}
            raw_checks.append({"path": str(path), "checks": checks})
            if not all(checks.values()):
                raise AssertionError("The direct RGB decoder changed original complete pixels")
        adapter = OpenCVPreparation(PILtoTorch, True)

        def prepare(path, mode):
            with Image.open(path) as image:
                resolution = (round(image.width / 2), round(image.height / 2))
                if mode == "pillow":
                    result = PILtoTorch(image, resolution)
                elif mode == "old_opencv":
                    result = adapter(image, resolution)
                else:
                    decoded = cv2.imread(str(path), cv2.IMREAD_COLOR_RGB | cv2.IMREAD_IGNORE_ORIENTATION)
                    if decoded is None or decoded.shape != (image.height, image.width, 3) or decoded.dtype != np.uint8:
                        raise AssertionError("The direct RGB decoder changed the declared image")
                    with Image.fromarray(decoded) as candidate:
                        result = PILtoTorch(candidate, resolution)
            if result.device.type != "cpu":
                raise AssertionError("The original image preparation must remain CPU only")
            return result

        def group(mode):
            before = time.perf_counter()
            jobs, results, next_submit = deque(), [], 0
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix="owned-rgb-profile") as pool:
                while next_submit < min(8, len(selected)):
                    jobs.append(pool.submit(prepare, selected[next_submit], mode))
                    next_submit += 1
                while jobs:
                    results.append(jobs.popleft().result())
                    if next_submit < len(selected):
                        jobs.append(pool.submit(prepare, selected[next_submit], mode))
                        next_submit += 1
            return results, time.perf_counter() - before

        rows = []
        for trial, order in enumerate((("pillow", "old_opencv", "direct_rgb"),
                ("old_opencv", "direct_rgb", "pillow"), ("direct_rgb", "pillow", "old_opencv"))):
            results, costs = {}, {}
            for mode in order:
                results[mode], costs[mode] = group(mode)
            checks = {"all12_prepared_bits_and_layouts_exact": all(
                torch.equal(left.view(torch.int32), right.view(torch.int32)) and left.stride() == right.stride()
                and left.storage_offset() == right.storage_offset() and left.dtype == right.dtype
                and left.device == right.device and left.requires_grad == right.requires_grad
                for mode in ("old_opencv", "direct_rgb") for left, right in zip(results["pillow"], results[mode]))}
            rows.append({"trial": trial, "checks": checks, "complete12_preparation_wall_seconds": costs})
            if not all(checks.values()):
                raise AssertionError("The bounded RGB preparation changed original output bits or layout")
            del results
            gc.collect()
        checks = {"all12_full_decoded_pixels": all(all(row["checks"].values()) for row in raw_checks),
            "all_three_diagnostic_groups_original_prepared_bits": all(all(row["checks"].values()) for row in rows),
            "complete_selected_inputs_and_sources_unchanged": identities == [data_identity(path) for path in sources + selected]}
        build = cv2.getBuildInformation()
        codecs = [line.strip() for line in build.splitlines() if any(word in line for word in ("JPEG:", "PNG:", "Parallel framework:", "JPEG 2000:"))]
        guard.sample()
        result = {"status": "direct_rgb_bounded_diagnostic_verified" if all(checks.values()) else "diagnostic_failed",
            "checks": checks, "raw_pixel_rows": raw_checks, "rows": rows,
            "runtime": {"opencv": cv2.__version__, "numpy": np.__version__, "torch": torch.__version__,
                "Pillow": Image.__version__, "OpenCV_threads_unchanged": cv2.getNumThreads(), "codec_build_lines": codecs},
            "actual_peak_RSS_bytes": guard.row["peak_tree_RSS_bytes"],
            "complete_child_seconds": time.perf_counter() - STARTED, "diagnostic_only": True,
            "full1016_Scene_generic_errors_external_recovery_and_training": "not_run", "new_optimization_count": 0}
        atomic_json(output / "RESULT.json", result)
        print(json.dumps({key: result[key] for key in ("status", "checks", "rows", "runtime", "complete_child_seconds")}), flush=True)
        if not all(checks.values()):
            raise AssertionError("A direct native RGB diagnostic failed")


if __name__ == "__main__":
    main()
