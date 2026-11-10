"""Compare complete original Scene states and actual measured process-tree peaks."""

from pathlib import Path

from mapped_image_node_config import read


def compare_pair(pair, native_node, mapped_node):
    pair = Path(pair)
    native, mapped = read(pair/"native/RESULT.json"), read(pair/"mapped/RESULT.json")
    nodes = {"native": read(Path(native_node)/"NODE.json"), "mapped": read(Path(mapped_node)/"NODE.json")}
    resources = {mode: read(pair/mode/"WSL_RESOURCE.json") for mode in nodes}
    keys = set(native["scientific"])
    scientific_equal = keys == set(mapped["scientific"]) and all(native["scientific"][key] == mapped["scientific"][key] for key in keys)
    checks = {
        "both_original_complete_contracts": all(all(row["checks"].values()) for row in (native, mapped)),
        "all_full_pixels_K_aliases_gaussians_RNG_metadata_and_members_exact": scientific_equal,
        "identical_current_input_and_source_sets": native["inputs"] == mapped["inputs"] and native["sources"] == mapped["sources"],
        "identical_reference_runtime_and_configuration": native["runtime"] == mapped["runtime"] and native["runtime_config"] == mapped["runtime_config"],
        "full_native_generated_PLY_and_camera_JSON_bytes": all(native["generated_assets"][key]["bytes"] == mapped["generated_assets"][key]["bytes"] and native["generated_assets"][key]["sha256"] == mapped["generated_assets"][key]["sha256"] for key in ("input.ply", "cameras.json")),
        "both_original_Windows_GPU_disk_time_guards_closed": all(node["status"] == "completed" and not node["live_Windows_after_wait"] and all(not p["alive"] for p in node["actual_Linux_after_wait"]) and node["final_source_unchanged"] for node in nodes.values()),
        "complete_Linux_guard_windows_included": all(resource["status"] == "completed" and resource["peak_tree_RSS_bytes"] >= result["full_process_tree_peak_RSS_bytes"] for resource, result in zip(resources.values(), (native, mapped))),
    }
    actual_peaks = {mode: max(resources[mode]["peak_tree_RSS_bytes"], nodes[mode]["resource_peaks"]["Windows_tree_RSS_bytes"]) for mode in nodes}
    reduction = 1-actual_peaks["mapped"]/actual_peaks["native"]
    return {"status": "complete_Scene_pair_exact_memory_screen_passed" if all(checks.values()) and reduction >= .2 else "failed_verification",
        "checks": checks, "different_scientific_keys": sorted(key for key in keys if native["scientific"].get(key) != mapped["scientific"].get(key)),
        "actual_process_tree_peak_RSS_bytes": actual_peaks, "actual_peak_RSS_reduction": reduction,
        "Scene_load_seconds": {mode: result["Scene_load_seconds"] for mode, result in (("native", native), ("mapped", mapped))},
        "all_profile_and_node_seconds": {mode: node["complete_node_seconds"] for mode, node in nodes.items()},
        "first_build_is_not_hidden": mapped["mapped"]["created_files"] > 0,
        "mapped_payload_bytes": mapped["mapped"]["payload_bytes"],
        "independent_New_object_training_quality_recovery_or_acceptance": False, "new_optimization_count": 0}
