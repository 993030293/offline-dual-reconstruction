"""Observe the original complete Scene failure at its native distance call."""

from contextlib import contextmanager
import inspect
from pathlib import Path
import sys

import profile_native_feature_zero as parent
from runtime import atomic_json, data_identity
from check_zero_transfer_contracts import prefix
from profile_native_zero_bounded import ZeroSceneGuard


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    output.relative_to(parent.OUT.resolve())
    dataset = parent.ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003/dataset"
    input_paths = sorted(path for path in dataset.rglob("*") if path.is_file())
    input_ids = [data_identity(path) for path in input_paths]
    captured, evidence = {}, {}
    source = inspect.getsource(parent.main)
    anchor = '    native_activate = namespace["activate_reference"]'
    if source.count(anchor) != 1:
        raise ValueError("The original fault observation boundary changed")
    changed = source.replace(anchor, anchor + '\n    namespace["camera_execution"] = _observed_camera\n    namespace["prefetch_camera_images"] = _observed_prefetch\n    namespace["LinuxRSSGuard"] = _registered_guard')
    namespace = dict(parent.main.__globals__)
    import profile_mapped_scene as native_parent

    @contextmanager
    def observed_camera(*args, **kwargs):
        with native_parent.camera_execution(*args, **kwargs) as trace:
            captured["camera"] = trace
            yield trace

    @contextmanager
    def observed_prefetch(*args, **kwargs):
        with native_parent.prefetch_camera_images(*args, **kwargs) as trace:
            captured["prefetch"] = trace
            yield trace

    @contextmanager
    def fault_zero(path, enabled):
        import scene.gaussian_model as gaussian
        native_dist = gaussian.distCUDA2
        def dist(value):
            evidence["actual_distance_argument"] = {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
            raise OSError("R157 registered original distCUDA2 failure")
        gaussian.distCUDA2 = dist
        try:
            with parent.native_feature_zero(path, enabled) as trace:
                captured["zero"] = trace
                original_create = gaussian.GaussianModel.create_from_pcd
                def observed(model, *args, **kwargs):
                    try:
                        return original_create(model, *args, **kwargs)
                    finally:
                        evidence["actual_original_model_prefix"] = prefix(model)
                gaussian.GaussianModel.create_from_pcd = observed
                try:
                    yield trace
                finally:
                    gaussian.GaussianModel.create_from_pcd = original_create
        finally:
            gaussian.distCUDA2 = native_dist
            evidence["original_dist_function_restored"] = gaussian.distCUDA2 is native_dist

    namespace.update(native_feature_zero=fault_zero, _observed_camera=observed_camera, _observed_prefetch=observed_prefetch,
                     _registered_guard=ZeroSceneGuard)
    exec(compile(changed, str(Path(__file__)), "exec"), namespace)
    try:
        namespace["main"]()
    except OSError as error:
        if str(error) != "R157 registered original distCUDA2 failure":
            raise
        failure = {"type": type(error).__name__, "message": str(error)}
    else:
        raise AssertionError("The registered actual native distance failure was not triggered")
    checks = {"actual_original_full1016_camera_calls": captured["camera"]["native_camera_calls"] == 1016,
              "actual_original_all1016_images_prepared_consumed": captured["prefetch"]["prepared"] == captured["prefetch"]["consumed"] == 1016,
              "native_distance_received_all75767_points": evidence["actual_distance_argument"]["shape"] == [75767, 3],
              "all_native_model_functions_restored": captured["zero"]["native_functions_restored"] and evidence["original_dist_function_restored"],
              "all1036_original_input_bytes_unchanged": len(input_ids) == 1036 and input_ids == [data_identity(path) for path in input_paths]}
    (output / "EXECUTED_FAULT_OBSERVER.py").write_text(changed, encoding="utf-8")
    atomic_json(output / "FAULT_RESULT.json", {"status": "expected_original_native_failure" if all(checks.values()) else "failed",
        "checks": checks, "error": failure, "evidence": evidence, "contexts": captured,
        "inputs": input_ids, "extra_actual_native_HtoD_bytes": captured["zero"]["actual_native_HtoD_bytes"],
        "new_optimization_count": 0})
    if not all(checks.values()):
        raise AssertionError("A complete native Scene fault contract failed")


if __name__ == "__main__":
    main()
