"""Exercise an actual private JPEG write failure and shared-map cleanup."""

import json
from pathlib import Path
import sys

import numpy as np

import dataset_sampling_shared_carrier as shared
from profile_vector_full_export import BASE, ROOT, read
from runtime import atomic_json, data_identity


def main():
    own = BASE / "v6_completion_001"
    folder = own / "filesystem_error"
    if folder.exists():
        raise FileExistsError("The fault must have a fresh owned directory")
    driver = Path(__file__).resolve()
    original = driver.with_name("profile_vector_full_export.py")
    source = original.read_text(encoding="utf-8")
    anchor = 'sources = [Path(__file__),'
    if source.count(anchor) != 1:
        raise AssertionError("The original complete source list changed")
    executed = source.replace(anchor, 'sources = [Path(__file__), _original_profile_source,')
    (own / "EXECUTED_FILESYSTEM_FAULT_V6.py").write_text(executed, encoding="utf-8")
    atomic_json(own / "FILESYSTEM_FAULT_CONTRACT.json", {"sources": [data_identity(driver), data_identity(original)],
        "sole_source_list_replacement": True, "new_optimization_count": 0})
    configuration = shared.complete_configuration
    previous_argv = sys.argv
    previous_environment = __import__("os").environ.get("OPENIMAGEIO_THREADS")
    before_rng = np.random.get_state()
    injected = []

    def block_owned_JPEG(dataset, names, templates, proof):
        result = configuration(dataset, names, templates, proof)
        path = Path(dataset).resolve() / "images" / names[1]
        path.relative_to(folder.resolve())
        if path.exists():
            raise AssertionError("The post-seed fault target was already written")
        path.mkdir()
        injected.append(str(path))
        atomic_json(folder / "ACTUAL_WRITE_FAULT.json", {"target": str(path), "actual_private_directory_created": path.is_dir(),
            "complete_image_count": len(names), "original_seed_count": len(templates), "original_inputs_untouched": True})
        return result

    shared.complete_configuration = block_owned_JPEG
    namespace = {"__file__": str(driver), "__name__": "owned_full_export_fault_bridge", "_original_profile_source": original}
    exec(compile(executed, str(original), "exec"), namespace)
    error = None
    try:
        sys.argv = [str(driver), "--mode", "vector", "--output", str(folder)]
        namespace["main"]()
    except OSError as actual:
        error = actual
    finally:
        sys.argv = previous_argv
        shared.complete_configuration = configuration
    if error is None:
        raise AssertionError("The actual native JPEG write failure did not propagate")
    closed = read(folder / "sampling_proof/SHARED_SEGMENTS_CLOSED.json")
    created = read(folder / "sampling_proof/SHARED_SEGMENTS_CREATED.json")
    bindings = read(folder / "BINDING_RESTORATION.json")
    after_rng = np.random.get_state()
    checks = {
        "actual_private_post_seed_JPEG_directory_fault": len(injected) == 1 and Path(injected[0]).is_dir()
            and read(folder / "ACTUAL_WRITE_FAULT.json")["complete_image_count"] == 1016,
        "actual_native_bitmap_write_OSError_propagated": type(error) is OSError and "original complete JPEG97 writer failed" in str(error),
        "actual_five_shared_segments_created_and_all_released": len(created["headers"]) == 5 and closed["headers"] == created["headers"]
            and closed["all_handles_closed"] and closed["all_names_unavailable_after_shutdown"],
        "all_original_bindings_and_fault_hook_restored": all(value for key, value in bindings.items() if key != "new_optimization_count")
            and shared.complete_configuration is configuration and sys.argv is previous_argv,
        "original_parent_environment_and_RNG_preserved": __import__("os").environ.get("OPENIMAGEIO_THREADS") == previous_environment
            and before_rng[0] == after_rng[0] and np.array_equal(before_rng[1], after_rng[1]) and before_rng[2:] == after_rng[2:],
        "original_actual_full_failure_saved": read(folder / "FAILURE.json")["status"] == "failed_full1050_source_or_sampling_boundary",
    }
    atomic_json(folder / "FAULT_RESULT.json", {"status": "actual_private_JPEG_write_error_and_shared_map_cleanup_verified"
        if all(checks.values()) else "failed_filesystem_fault_review", "checks": checks, "actual_exception": repr(error),
        "unknown_multi_CPP_fault_order_or_external_object_not_verified": True, "new_optimization_count": 0})
    if not all(checks.values()):
        raise AssertionError({key: value for key, value in checks.items() if not value})
    print(json.dumps({"status": "actual_JPEG_failure_cleanup_passed", "checks": len(checks)}), flush=True)


if __name__ == "__main__":
    main()
