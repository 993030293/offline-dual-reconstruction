"""Measure original CPU preparation statements before the prefetch binding."""

from contextlib import contextmanager
import difflib
import inspect
from pathlib import Path
import statistics
import textwrap
import threading
import time

from runtime import atomic_json


def occupied_seconds(intervals):
    ordered = sorted(intervals)
    total = 0.
    end = None
    for begin, finish in ordered:
        if end is None or begin > end:
            total += finish - begin
        elif finish > end:
            total += finish - end
        end = finish if end is None else max(end, finish)
    return total


@contextmanager
def timed_camera_preparation(output):
    import scene
    import scene.cameras as cameras
    from scene import GaussianModel
    import utils.camera_utils as camera_utils

    output = Path(output)
    original = cameras.PILtoTorch
    native_dataset = scene.sceneLoadTypeCallbacks["Colmap"]
    native_create = GaussianModel.create_from_pcd
    native_camera = camera_utils.loadCam
    before = textwrap.dedent(inspect.getsource(original))
    anchors = (
        ("    resized_image_PIL = pil_image.resize(resolution)",
         "    _marks = [_clock()]\n    resized_image_PIL = pil_image.resize(resolution)\n    _marks.append(_clock())"),
        ("    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0",
         "    _array = np.array(resized_image_PIL)\n    _marks.append(_clock())\n    _tensor = torch.from_numpy(_array)\n    _marks.append(_clock())\n    resized_image = _tensor / 255.0\n    _marks.append(_clock())"),
        ("        return resized_image.permute(2, 0, 1)",
         "        return _finish(resized_image.permute(2, 0, 1), _marks, pil_image, resolution)"),
        ("        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)",
         "        return _finish(resized_image.unsqueeze(dim=-1).permute(2, 0, 1), _marks, pil_image, resolution)"),
    )
    after = before
    for old, new in anchors:
        if after.count(old) != 1:
            raise ValueError("An original preparation statement or layout branch changed")
        after = after.replace(old, new)
    rows, calls = [], {"dataset_reader": [], "Gaussian_create_from_pcd_host": [], "loadCam_serial_including_wait": []}
    lock = threading.Lock()

    def finish(result, marks, image, resolution):
        marks.append(time.perf_counter())
        row = {"filename": str(image.filename), "mode": image.mode,
               "input_size": list(image.size), "resolution": list(resolution),
               "thread": threading.current_thread().name, "marks": marks,
               "seconds": [b - a for a, b in zip(marks, marks[1:])]}
        with lock:
            rows.append(row)
        return result

    def measured(native, key):
        def invoke(*args, **kwargs):
            begin = time.perf_counter()
            result = native(*args, **kwargs)
            calls[key].append(time.perf_counter() - begin)
            return result
        return invoke

    namespace = dict(original.__globals__)
    namespace.update(_clock=time.perf_counter, _finish=finish)
    filename = output / "EXECUTED_TIMED_PILtoTorch.py"
    filename.write_text(after, encoding="utf-8")
    (output / "TIMED_PILtoTorch.diff").write_text("".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="original_PILtoTorch", tofile="original_statements_timed")), encoding="utf-8")
    exec(compile(after, str(filename), "exec"), namespace)
    cameras.PILtoTorch = namespace["PILtoTorch"]
    scene.sceneLoadTypeCallbacks["Colmap"] = measured(native_dataset, "dataset_reader")
    GaussianModel.create_from_pcd = measured(native_create, "Gaussian_create_from_pcd_host")
    camera_utils.loadCam = measured(native_camera, "loadCam_serial_including_wait")
    trace = {"new_optimization_count": 0, "no_algorithm_replacement": True}
    succeeded = False
    try:
        yield trace
        succeeded = True
    finally:
        cameras.PILtoTorch = original
        scene.sceneLoadTypeCallbacks["Colmap"] = native_dataset
        GaussianModel.create_from_pcd = native_create
        camera_utils.loadCam = native_camera
        names = ["resize_including_lazy_decode", "numpy_array", "torch_from_numpy", "torch_div255", "native_layout"]
        totals = {name: sum(row["seconds"][i] for row in rows) for i, name in enumerate(names)}
        unions = {name: occupied_seconds([(row["marks"][i], row["marks"][i + 1]) for row in rows]) for i, name in enumerate(names)}
        threads = sorted({row["thread"] for row in rows})
        checks = {
            "original_bound_prepare_called_all1016_in_bounded_workers": len(rows) == 1016 and 1 <= len(threads) <= 4 and all(name.startswith("camera-decode") for name in threads),
            "all_five_original_statement_intervals_finite_nonnegative": all(len(row["marks"]) == 6 and len(row["seconds"]) == 5 and all(0 <= value < 300 for value in row["seconds"]) for row in rows),
            "all1016_unique_original_image_names": len({row["filename"] for row in rows}) == 1016,
            "original_full_dataset_Gaussian_and_camera_calls": [len(calls[key]) for key in calls] == [1, 1, 1016],
            "all_original_prepare_dataset_Gaussian_camera_functions_restored": cameras.PILtoTorch is original and scene.sceneLoadTypeCallbacks["Colmap"] is native_dataset and GaussianModel.create_from_pcd is native_create and camera_utils.loadCam is native_camera,
        }
        trace.update(status="complete_original_preparation_cost_diagnostic0" if succeeded and all(checks.values()) else "failed_preparation_cost_diagnostic0",
            checks=checks, worker_threads=threads, worker_call_count=len(rows),
            worker_cumulative_seconds=totals, worker_occupied_union_seconds=unions,
            all_prepare_occupied_union_seconds=occupied_seconds([(r["marks"][0], r["marks"][-1]) for r in rows]),
            host_call_cumulative_seconds={key: sum(value) for key, value in calls.items()},
            host_call_counts={key: len(value) for key, value in calls.items()},
            prepare_call_median_seconds=statistics.median(sum(row["seconds"]) for row in rows) if rows else None,
            all_calls=rows, host_calls=calls,
            worker_sums_and_component_unions_are_not_additive_scene_wall=True,
            Gaussian_host_calls_not_extra_CUDA_synchronized=True,
            exact_original_outputs_pending_complete_reference_comparison=True)
        atomic_json(output / "PREPARATION_COST.json", trace)
        if succeeded and not all(checks.values()):
            raise AssertionError("The diagnostic missed an original complete preparation binding")
