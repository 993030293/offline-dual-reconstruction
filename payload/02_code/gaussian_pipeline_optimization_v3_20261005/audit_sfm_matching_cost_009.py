"""Measure the complete original readonly summary without rerunning reconstruction."""
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup009"
SOURCE = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/e2_sfm1002_001"
sys.path.insert(0, str(V1))
from run_sfm_preflight import database_summary
from runtime import data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(folder, name, payload):
    with (folder / name).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    output = Path(sys.argv[1]).resolve()
    output.relative_to(OUT)
    if output.exists() or shutil.disk_usage(str(ROOT)).free < 20 * 1024 ** 3:
        raise ValueError("A fresh bounded private output and disk reserve are required")
    output.mkdir(parents=True)
    sources = [data_identity(Path(__file__)), data_identity(V1 / "run_sfm_preflight.py"), data_identity(V1 / "runtime.py")]
    inputs = [data_identity(SOURCE / name) for name in ("features.db", "guided/database.db", "image_list.txt", "guided_pairs.txt", "STATE.json", "CONTRACT.json", "RESULT.json", "FEATURES.json", "guided/MATCHING.json")]
    save(output, "PROTOCOL.json", {"sources": sources, "inputs": inputs, "complete_function": "run_sfm_preflight.database_summary",
                                   "original1002_images_and11583_pair_file": True, "readonly_existing_SQLite_not_new_matching": True,
                                   "timed_fresh_calls_each": 3, "profiled_call_separate_from_timing": True, "new_optimization_count": 0,
                                   "project_kb_generation": "9d5a7b76-ad15-4d17-800b-b5c6b59efb96", "project_kb_status": "two_actual_MCP_calls_ok_four_unconfirmed_historical_locators_only"})
    (output / "sources").mkdir()
    for row in sources:
        shutil.copyfile(row["path"], output / "sources" / Path(row["path"]).name)
    names = (SOURCE / "image_list.txt").read_text(encoding="utf-8").splitlines()
    pairs = (SOURCE / "guided_pairs.txt").read_text(encoding="utf-8").splitlines()
    if len(names) != 1002 or len(pairs) != 11583:
        raise AssertionError("The frozen full input denominator changed")
    captured = []

    def profiler(frame, event, result):
        if frame.f_code is not database_summary.__code__ or event != "return" or not isinstance(result, dict):
            return
        values = frame.f_locals
        adjacency = values["adjacency"]
        measured = {"node_count": len(adjacency), "directed_adjacency_entries": sum(map(len, adjacency.values())),
                    "adjacency_dict_shallow_bytes": sys.getsizeof(adjacency),
                    "all_adjacency_sets_shallow_bytes": sum(sys.getsizeof(value) for value in adjacency.values()),
                    "final_BFS_reached_set_shallow_bytes": sys.getsizeof(values["reached"]),
                    "final_BFS_remaining_set_shallow_bytes": sys.getsizeof(values["remaining"]),
                    "final_BFS_queue_shallow_bytes": sys.getsizeof(values["queue"]),
                    "materialized_existing_int_references_not_new_integer_backing_bytes": True,
                    "not_peak_RSS_or_total_allocator_bytes": True}
        captured.append(measured)

    results = []
    try:
        for relative, oracle in (("features.db", "FEATURES.json"), ("guided/database.db", "guided/MATCHING.json")):
            expected = read(SOURCE / oracle)
            timed = []
            for _ in range(3):
                gc.collect()
                at = time.perf_counter()
                value = database_summary(SOURCE / relative, names)
                elapsed = time.perf_counter() - at
                timed.append(elapsed)
                if value != expected:
                    raise AssertionError("The current original full database summary differs from its frozen receipt")
            gc.collect()
            sys.setprofile(profiler)
            try:
                observed = database_summary(SOURCE / relative, names)
            finally:
                sys.setprofile(None)
            if observed != expected:
                raise AssertionError("The observed original full graph summary changed")
            graph = captured[-1]
            results.append({"database": relative, "complete_function_seconds_each": timed, "graph": graph,
                            "image_count": value["image_count"], "attempted_matching_pairs": value["attempted_matching_pairs"],
                            "verified_pairs_at_least15": value["geometric_pairs_at_least_15_inliers"],
                            "component_sizes": value["component_sizes"], "original_complete_output_exact": True})
            save(output, "ORIGINAL_" + ("FEATURES.json" if relative == "features.db" else "MATCHING.json"), value)
        checks = {"full_original_both_databases_and_all_fields_exact": all(row["original_complete_output_exact"] for row in results),
                  "real_complete1002_and11583_pair_file_preserved": len(names) == 1002 and len(pairs) == 11583,
                  "direct_original_function_source_bound": Path(database_summary.__code__.co_filename).resolve() == (V1 / "run_sfm_preflight.py").resolve(),
                  "all_current_inputs_and_sources_unchanged": all(data_identity(row["path"]) == row for row in inputs + sources),
                  "profiler_drained_and_no_global_alias_change": sys.getprofile() is None,
                  "registered_output_and_disk_bounds_kept": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) < 2 * 1024 ** 3 and shutil.disk_usage(str(ROOT)).free >= 20 * 1024 ** 3}
        state = read(SOURCE / "STATE.json")
        save(output, "RESULT.json", {"status": "complete_original_readonly_summary_cost_diagnostic0", "checks": checks, "cases": results,
                                     "historical_stages_only": [{"name": stage["name"], "seconds": stage["elapsed_seconds"], "status": stage["status"]} for stage in state["stages"]],
                                     "historical_1002_registration_guard": read(SOURCE / "RESULT.json")["registration_guard_passed"],
                                     "current1016_map_not_revalidated_here": True, "sources": sources, "inputs": inputs,
                                     "parent_count": 35, "new_optimization_count": 0, "goal_status": "active"})
        print(json.dumps({"checks": checks, "cases": results, "new_count": 0}))
        if not all(checks.values()):
            raise AssertionError("The original readonly summary diagnostic failed")
    except BaseException as error:
        sys.setprofile(None)
        save(output, "FAILURE.json", {"error": repr(error), "cases_completed": results, "sources": sources, "inputs": inputs, "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()
