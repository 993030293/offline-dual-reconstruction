"""Close the original radial diagnostic and the first-pair cost-rejected candidate."""

import json
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import data_identity


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def same_data_identity(row):
    return row == data_identity(Path(row["path"]))


def paired(result):
    return sorted(result["episodes"], key=lambda item: item["hooks"])


def exact_views(left, right):
    keys = ("rgba_sha256", "position", "direction", "quaternion", "world", "projection", "dimensions", "count", "mode", "activeSplats", "currentSplats")
    return len(left["views"]) == len(right["views"]) == 33 and all(
        a["name"] == b["name"] and all(a["capture"][key] == b["capture"][key] for key in keys)
        for a, b in zip(left["views"], right["views"])
    ) and all(left["first"][key] == right["first"][key] for key in keys)


def alive(record):
    try:
        item = psutil.Process(record["pid"])
        return item.is_running() and item.create_time() == record["creation_time"]
    except psutil.NoSuchProcess:
        return False


def main():
    base = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup012"
    family = base / "R167"
    output = base / "r0"
    output.mkdir(exist_ok=False)
    parent = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    static = read(base / "f1/RESULT.json")
    diagnostic = read(base / "p0/RESULT.json")
    g0, g2, g3, formal = [read(family / tag / "RESULT.json") for tag in ("g0", "g2", "g3", "c1")]
    original, observed = paired(diagnostic)
    native_left = read(family / "p0/EPISODE_0.json")
    native_right = read(family / "p0/EPISODE_1.json")
    public = read(family / "p1/RESULT.json")
    records, leases, guards = [], [], []
    for directory, tag in [(base, "p0"), *[(family, name) for name in ("g0", "g1", "g2", "g3", "p0", "p1", "c1")]]:
        guard = read(directory / (tag + "_GUARD.json"))
        lease = read(directory / (tag + "_LEASE.json"))
        guards.append({"path": str(directory / (tag + "_GUARD.json")), **guard})
        leases.append(lease)
        records.extend(guard.get("owned_process_identities", []))
        for prefix in ("guard", "child"):
            records.append({"pid": lease[prefix + "_pid"], "creation_time": lease[prefix + "_creation_time"]})
    unique_records = {(row["pid"], row["creation_time"]): row for row in records}
    ended = [not alive(row) for row in unique_records.values()]
    trace = read(base / "p0/RADIAL_DEPENDENCY_TRACE.json")
    transitions = []
    for scope in ("scene", "box"):
        for k in range(1, 7):
            label = f"{scope}_pure_look_{k}"
            entry = next(row for row in trace["events"] if row["kind"] == "prepare" and row["label"] == label)
            previous, current = entry["before"]["models"][0], entry["after"]["models"][0]
            transitions.append({"name": label, "same_previous_origin": entry["before"]["currentOrigin"] == entry["nextOrigin"],
                "same_consumed_translate": previous["viewTranslate"] == current["viewTranslate"],
                "changed_rotate": previous["viewRotate"] != current["viewRotate"],
                "native_version_increased": entry["nextVersion"] == entry["before"]["currentVersion"] + 1,
                "mapping_preserved": entry["nextMappingVersion"] == entry["before"]["mappingVersion"]})
    before, after = paired(formal)
    baseline = before["complete_context_to_actual_close_ms"] / 1000
    candidate = after["complete_context_to_actual_close_ms"] / 1000
    known_guard_seconds = sum(row["elapsed_seconds"] for row in guards)
    reserved_seconds = 25
    fee_lower_bound = known_guard_seconds + reserved_seconds
    first_pair_net_upper = (baseline - candidate) / (baseline + fee_lower_bound / 6)
    component_keys = native_left["nativeComponents"]["keys"]
    component_left = native_left["nativeComponents"]["fixtures"]
    component_right = native_right["nativeComponents"]["fixtures"]
    originals = [*static["sources"], *static["inputs"]]
    for result in (diagnostic, g0, g2, g3, formal):
        originals.extend(result["sources"] + result["inputs"])
    native_protocol = read(family / "p0/PROTOCOL.json")
    originals.extend(native_protocol["sources"] + native_protocol["inputs"])
    originals.extend(public["sources"] + public["inputs"])
    originals = list({(row["path"], row["sha256"]): row for row in originals}.values())
    checks = {
        "parent36of50_and36_unique_acceptance_receipts_preserved": parent["additional_accepted_count"] == 36 and len(parent["accepted_mechanisms"]) == len(parent["acceptance_receipts"]) == 36 and len(set(parent["accepted_mechanisms"])) == 36,
        "static15_source_checks_pass_and_first_failed_source_retained": len(static["checks"]) == 15 and all(static["checks"].values()) and (base / "f0/FAILURE.json").is_file() and (base / "f0/SOURCE_FAILED.py").is_file(),
        "original_two_fresh_contexts_all33_plus_initial_exact_and15_checks": len(diagnostic["checks"]) == 15 and all(diagnostic["checks"].values()) and exact_views(original, observed),
        "ten_same_center_translate_native_direction_version_transitions_verified": sum(row["same_previous_origin"] and row["same_consumed_translate"] and row["changed_rotate"] and row["native_version_increased"] and row["mapping_preserved"] for row in transitions) == 10,
        "g0_V1_exact_positive_not_relabelled_as_current_V3": all(g0["checks"].values()) and exact_views(*paired(g0)) and any(Path(row["path"]).name == "radial_translation_dependency_001.js" for row in g0["sources"]),
        "g1_actual_prebrowser_syntax_failure_and_source_retained": read(family / "g1_GUARD.json")["returncode"] == 1 and (family / "g1/SOURCE_FAILED.cjs").is_file() and "SyntaxError" in (family / "g1_CONTROLLER.log").read_text(encoding="utf-8"),
        "g2_wrong_member_binding_failure_preserved_not_speed_proof": [key for key, value in g2["checks"].items() if not value] == ["actual_candidate_direction_and_unused_producer_changes_suppressed"] and exact_views(*paired(g2)) and paired(g2)[1]["radialCache"]["totals"]["eligibleDirectionChanges"] == 0,
        "g3_current_native_splats_V3_all16_exact_checks_pass": len(g3["checks"]) == 16 and all(g3["checks"].values()) and exact_views(*paired(g3)) and any(Path(row["path"]).name == "radial_translation_dependency_v3.js" for row in g3["sources"]),
        "native16_actual_components_both_pass_all_48_captures_exact": len(component_left) == len(component_right) == 16 and all(row["passed"] for row in component_left + component_right) and all(a["name"] == b["name"] and all(a[tag][key] == b[tag][key] for tag in ("reset", "first", "second") for key in component_keys) for a, b in zip(component_left, component_right)),
        "unsupported_routes_original_generation_not_suppressed": all(row["after"]["totals"]["suppressedUnusedProducerChanges"] == row["before"]["totals"]["suppressedUnusedProducerChanges"] and row["after"]["totals"]["nativeReadbacks"] > row["midpoint"]["totals"]["nativeReadbacks"] for row in component_right if row["unsupported"]),
        "native_component_public_path_failure_retained_no_native_rerun": read(family / "p0_GUARD.json")["returncode"] == 1 and "ENOENT" in (family / "p0_CONTROLLER.log").read_text(encoding="utf-8") and (family / "p0/FAILURE.json").is_file(),
        "current_public_controller_original1016_public360_generic128_all8_checks_pass": len(public["checks"]) == 8 and all(public["checks"].values()) and read(family / "p1_GUARD.json")["returncode"] == 0,
        "formal_first_AB_all33_exact16_checks_and_positive_raw_gain": read(family / "c1/PROTOCOL.json")["mode"] == "AB" and len(formal["checks"]) == 16 and all(formal["checks"].values()) and exact_views(before, after) and candidate < baseline,
        "first_pair_known_guard_plus_registered_budget_net_upper_under_V6_2percent": first_pair_net_upper < .02 and known_guard_seconds > 70 and reserved_seconds == 25,
        "formal_rest_lifecycle_context_and_fresh_recovery_not_run": not any((family / tag).exists() for tag in ("c2", "c3", "l0", "x0", "f0")),
        "no_accepted_R167_or_new_mechanism_counter": not any("R167" in str(row) for row in parent["accepted_mechanisms"] + parent["acceptance_receipts"]),
        "all_declared_native_source_and_input_data_identities_unchanged": all(same_data_identity(row) for row in originals),
        "all_owned_saved_birth_records_ended_and_leases_released": all(ended) and all(row["returncode"] in (0, 1) and row.get("lease_released_utc") for row in leases),
        "shared_lease_current_released": read(base.parent / "coordination_001/LOCK_OWNER.json")["status"] == "released",
        "tree_RSS_and_whole_device_GPURSS_bounds_preserved": all(row["peak_tree_RSS_bytes"] <= 4 * 1024 ** 3 for row in guards) and all(not read(path)["gpuFailures"] for path in base.glob("**/GPU_RESOURCE_SAMPLES.json")),
        "private_output2GiB_and_disk20GiB_bounds_preserved": sum(path.stat().st_size for path in base.rglob("*") if path.is_file()) < 2 * 1024 ** 3 and psutil.disk_usage(str(ROOT)).free >= 20 * 1024 ** 3,
    }
    result = {"status": "radial_diagnostic_and_R167_first_pair_V6_cost_negative_closed0" if all(checks.values()) else "failed_root_review0",
        "checks": checks, "source": data_identity(Path(__file__)), "first_formal_pair": {"baseline_seconds": baseline,
            "candidate_seconds": candidate, "raw_gain_percent": 100 * (baseline - candidate) / baseline,
            "known_guard_nonoverlap_seconds": known_guard_seconds, "registered_final_metadata_exit_budget_seconds": reserved_seconds,
            "known_fee_lower_bound_seconds": fee_lower_bound, "optimistic_first_pair_net_upper_percent": 100 * first_pair_net_upper,
            "other_known_source_review_fees_additional_not_zero": True, "not_formal_three_pair_median_or_final_net_result": True},
        "transitions": transitions, "owned_birth_records": list(unique_records.values()), "ended": ended,
        "originals": originals, "new_optimization_count": 0, "parent_count": 36, "remaining": 14,
        "review_is_root_self_review_not_independent_agent": True,
        "nonempty_cov_LOD_GT_cross_hardware_errors_cancel_and_full_recovery_not_verified": True}
    with (output / "RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"status": result["status"], "checks": checks, "first_pair": result["first_formal_pair"], "births": len(ended)}, ensure_ascii=False))
    if not all(checks.values()):
        raise RuntimeError("The radial negative closure or original-source evidence failed")


if __name__ == "__main__":
    main()
