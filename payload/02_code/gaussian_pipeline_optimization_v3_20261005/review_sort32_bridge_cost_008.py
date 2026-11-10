"""Review original complete bridge diagnostics, source copies, fees, and ended process births."""
import collections
import datetime
import json
from pathlib import Path
import shutil
import sys

import psutil

from review_orbit_convergence_dag import identity, closed

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup008"
RELEASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    own = {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()}
    attempts = []
    sources = {}
    births = []
    copied = []
    copy_bytes = 0
    ordering_requests_bytes = 0
    for name in ("p0", "p1", "p2"):
        folder = OUT / name
        result, protocol = read(folder / "RESULT.json"), read(folder / "PROTOCOL.json")
        guard, lease = read(OUT / (name + "_GUARD.json")), read(OUT / (name + "_lease.json"))
        births.extend(guard["owned_process_identities"])
        births.extend({"pid": lease[k + "_pid"], "creation_time": lease[k + "_creation_time"]} for k in ("guard", "child"))
        attempts.append({"name": name, "guard": guard, "lease": lease,
                         "actual_tool": {"p0": "6d163a", "p1": "ad5db5", "p2": "3d01e2"}[name],
                         "actual_tool_exit_code": 1 if name == "p0" else 0})
        for row in protocol["inputs"] + protocol["sources"]:
            if row["path"] in sources and row != sources[row["path"]]:
                raise AssertionError("A source or frozen input changed between diagnostic attempts")
            sources[row["path"]] = row
        for row in protocol["sources"]:
            is_donor = Path(row["path"]).name == "COMPILED_DRIVER_BODY.cjs"
            target = folder / "sources" / ("DONOR_COMPILED_DRIVER_BODY.cjs" if is_donor and name == "p2" else Path(row["path"]).name)
            current = identity(target)
            copied.append(current)
            assert current["sha256"] == row["sha256"] and current["bytes"] == row["bytes"]
        for episode in result["episodes"]:
            copy_bytes += episode["bridge"]["totals"]["observedImportCopyBytes"]
            ordering_requests_bytes += episode["flight"]["after"]["orderingAllocation"]["totals"]["verifiedNativeNewArrayBytes"]
        license_text = (RELEASE / "shared-build/vendor/SPARK-LICENSE.txt").read_text(encoding="utf-8")
        license_copy = folder / "sources/SPARK-LICENSE.txt"
        if license_copy.exists():
            assert license_copy.read_text(encoding="utf-8") == license_text
        else:
            shutil.copyfile(RELEASE / "shared-build/vendor/SPARK-LICENSE.txt", license_copy)
        assert "WORLD LABS TECHNOLOGIES" in license_copy.read_text(encoding="utf-8")
    final = read(OUT / "p2/RESULT.json")
    trace = final["episodes"][1]["bridge"]["traces"]
    groups = collections.defaultdict(lambda: {"calls": 0, "importCopyBytes": 0, "copyImportWallMs": 0., "handlerWallWithObserverMs": 0.})
    for row in trace:
        group = groups[row["numSplats"]]
        group["calls"] += 1
        group["importCopyBytes"] += row["observedImportCopyBytes"]
        group["copyImportWallMs"] += sum(c["nativeImportWallMs"] for c in row["copies"])
        group["handlerWallWithObserverMs"] += row["originalHandlerWallWithObserverMs"]
    worker = (OUT / "p2/sources/ORIGINAL_WORKER.js").read_text(encoding="utf-8")
    observed = (OUT / "p2/sources/OBSERVED_WORKER.js").read_text(encoding="utf-8")
    block = (CODE / "sort32_bridge_worker_observer.js").read_text(encoding="utf-8")
    native_viewer = (ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup007/p3/sources/EXECUTED_VIEWER.js").read_text(encoding="utf-8")
    expected_viewer = 'import { installGlResourceGenerationV2 } from "./gl_resource_generation_v2.js";\n' + native_viewer.replace("  const costSpark = new SparkRenderer", "  state.glGeneration = installGlResourceGenerationV2({ renderer });\n  const costSpark = new SparkRenderer")
    executed = (OUT / "p2/sources/COMPILED_DRIVER_BODY.cjs").read_text(encoding="utf-8")
    donor = (OUT / "p2/sources/DONOR_COMPILED_DRIVER_BODY.cjs").read_text(encoding="utf-8")
    report = (DOC / "VIEWER_SORT32_BRIDGE_COST_008_RESULTS.md").read_text(encoding="utf-8")
    master = read(DOC / "CURRENT_STATE.json")
    checks = {
        "final_complete_original_all13_quality_checks": all(final["checks"].values()) and len(final["checks"]) == 13,
        "both_successes_current_original_all13_exact": all(all(read(OUT / n / "RESULT.json")["checks"].values()) for n in ("p1", "p2")),
        "p0_actual_failed_binding_retained_not_zero_copy_evidence": read(OUT / "p0/RESULT.json")["checks"]["actual_sort32_import_traces_bound_to_original_RPC"] is False and (OUT / "p0/FAILURE.json").exists(),
        "original_worker_body_only_readonly_block_added": observed == worker.replace("initialize().catch(console.error);", block + "\ninitialize().catch(console.error);"),
        "original_complete_viewer_only_common_gl_registry_added": (OUT / "p2/sources/EXECUTED_VIEWER.js").read_text(encoding="utf-8") == expected_viewer,
        "original_probe_exact_no_extra_reference_or_scratch": (OUT / "p2/sources/EXECUTED_PROBE.js").read_bytes() == (ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup007/p3/sources/EXECUTED_PROBE.js").read_bytes(),
        "final_executed_and_donor_snapshot_names_independent": executed != donor and "original_sort32_WASM_import_copy_cost_diagnostic_only" in executed and "DONOR_COMPILED_DRIVER_BODY.cjs" in executed,
        "actual_bridge_uses_exact_readback_prefix_and_output_count": all(len(r["copies"]) == (1 if r["numSplats"] == 0 else 2) and all(c["bytes"] == r["numSplats"] * 4 for c in r["copies"]) for r in trace),
        "actual_nonzero_copy_directions_input_readback_output_ordering": all(r["copies"][0]["direction"] == "JS_to_WASM" and r["copies"][0]["source"]["sharesReadbackBuffer"] and (r["numSplats"] == 0 or r["copies"][1]["direction"] == "WASM_to_JS" and r["copies"][1]["destination"]["sharesOrderingBuffer"]) for r in trace),
        "no_observed_scalar_or_new_JS_backing_requests_not_internal_WASM_claim": all(r["scalarReadCalls"] == r["scalarWriteCalls"] == r["observedJSBackingRequestsBytes"] == 0 and r["internalWasmCopiesAndAllocationsUnobserved"] for r in trace),
        "all_frozen_inputs_actual_sources_and_copies_still_pinned": all(identity(row["path"]) == row for row in sources.values()),
        "all_three_actual_exits_and_prior_births_closed": all(a["guard"]["returncode"] == a["lease"]["returncode"] == a["actual_tool_exit_code"] and a["lease"]["lease_released_utc"] for a in attempts) and all(closed(row) for row in births),
        "normal_aliases_restored_and_original_initial_full_populations": all(e["bridge"]["originalAliasesRestored"] and e["glGeneration"]["originalAliasesRestored"] and e["first"]["count"] == [526886, 0] and e["flight"]["after"]["count"] == [526886, 60451] for e in final["episodes"]),
        "report_directly_read_no_speed_or_unknown_geometry_acceptance": all(t in report for t in ("本轮计0", "不支持", "首次", "命名覆盖", "不可", "未启用R164", "未执行检查接受", "23.2970000000")),
        "source_license_text_present_with_ownership": all((OUT / n / "sources/SPARK-LICENSE.txt").read_text(encoding="utf-8") == license_text for n in ("p0", "p1", "p2")),
        "no_new_candidate_no_count_change_and_resource_reserves": final["new_optimization_count"] == 0 and final["root_accepted_count"] == master["additional_accepted_count"] == 35 and shutil.disk_usage(str(ROOT)).free >= 20 * 1024 ** 3 and all(a["guard"]["peak_tree_RSS_bytes"] < 4 * 1024 ** 3 for a in attempts),
    }
    known_guard_seconds = sum(a["guard"]["elapsed_seconds"] for a in attempts)
    payload = {"status": "closed_original_bridge_copy_cost_negative_lead0" if all(checks.values()) else "failed_separate_text_and_source_review0",
               "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "checks": checks,
               "reviewer_process_identity": own, "review_kind": "separate_root_self_review_not_independent_agent",
               "source_identities": list(sources.values()) + [identity(Path(__file__)), identity(CODE / "review_orbit_convergence_dag.py")], "executed_copies": copied, "ended_process_birth_records": births,
               "attempts": attempts, "p2_numSplats_groups": dict(groups),
               "known_guard_seconds_no_nested_double_charge": known_guard_seconds,
               "all_attempts_observed_import_copy_bytes": copy_bytes, "all_attempts_original_ordering_request_bytes": ordering_requests_bytes,
               "known_RGBA_capture_requests_bytes": 6 * 23 * 960 * 600 * 4,
               "source_reader_failure": {"tool_chunk": "f8c061", "exit_code": 1, "seconds": .1784012, "cause": "strict JSON rejected the original JavaScript raw horizontal TAB"},
               "source_locator_read_failures": ["absent FAILED_EPISODE_1 after two completed episodes", "CURRENT_STATE was under research, not outputs", "license was under frozen v2 release, not v3"],
               "review_and_post_actual_exit_cost_not_yet_in_this_receipt": True,
               "parent_count": 35, "new_optimization_count": 0, "goal_status": "active", "next_route": "explore_other_complete_consumers"}
    save(OUT / "SOURCE_COST_REVIEW_001.json", payload)
    print(json.dumps({"status": payload["status"], "checks": checks, "known_guard_seconds": known_guard_seconds, "source_count": len(sources), "closed_birth_records": len(births), "new_count": 0}))
    if not all(checks.values()):
        raise AssertionError("The separate original bridge diagnostic review failed")


if __name__ == "__main__":
    main()
