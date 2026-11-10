"""Recheck one closed integer reduction under the authorized complete-stage policy."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/fm001"
OUT = BASE / "v2_screen_001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from run_fused_integer_campaign import compare_pair
import compare_actual_localization


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def bindings_restored(receipt):
    return receipt["worker_restored"] is True and receipt["argv_restored"] is True


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", choices=("v2_screen_001", "v2_screen_002"), default="v2_screen_001")
    args = parser.parse_args()
    OUT = BASE / args.screen
    OUT.mkdir(parents=True, exist_ok=False)
    original = read(BASE / "c0/t0/b/RESULT.json")
    closure = read(BASE / "UNACCEPTED_CLOSURE_001.json")
    generic = read(BASE / "g0/RESULT.json")
    if not all(closure["checks"].values()) or not all(original["checks"].values()) or len(generic["rows"]) != 9 or not all(generic["checks"].values()):
        raise ValueError("The retained complete control, GPU oracle or failure closure changed")
    paths = [Path(row["path"]) for row in read(BASE / "d0/CONTRACT.json")["sources"]]
    paths += [Path(__file__), CODE / "fused_match_node_002.py", CODE / "fused_match_node_config.py",
              CODE / "run_fused_integer_campaign.py", Path(compare_actual_localization.__file__)]
    paths = list(dict.fromkeys(paths))
    sources = [data_identity(path) for path in paths]
    fixed = {Path(row["path"]): row for row in original["sources"]}
    if any(fixed[path] != data_identity(path) for path in (CODE / "fused_integer_reduction.py", CODE / "integer_sift_worker.py", CODE / "fused_integer_match_worker.py")):
        raise ValueError("The GPU oracle's exact algorithm identity changed")
    inputs = original["inputs"]
    if inputs != [data_identity(row["path"]) for row in inputs]:
        raise ValueError("The complete frozen native input identity changed")
    copies = OUT / "source_copies"
    copies.mkdir()
    for index, path in enumerate(paths):
        shutil.copyfile(path, copies / (f"{index:02d}_" + path.name))
    setup_seconds = time.perf_counter() - STARTED
    atomic_json(OUT / "CONTRACT.json", {"sources": sources, "inputs": inputs, "old_shared_fees_seconds": closure["first_saved_child_seconds"],
        "setup_seconds": setup_seconds, "complete24_and48": True, "minimum_formal_phase_reduction": 0.05,
        "saved_control_is_science_comparator_not_current_fresh_cost": True, "new_optimization_count": 0})
    command = [sys.executable, "-B", str(CODE / "fused_match_node_002.py"), "--kind", "localization", "--mode", "fused",
        "--node", str(OUT / "n0"), "--output", str(OUT / "fused"), "--", sys.executable, "-B",
        str(CODE / "localize_fused_integer_matches_002.py"), "--mode", "fused", "--output", str(OUT / "fused")]
    subprocess.run(command, check=True, cwd=ROOT, timeout=1800)
    comparison = compare_pair(BASE / "c0/t0/b", OUT / "fused")
    candidate, node = read(OUT / "fused/RESULT.json"), read(OUT / "n0/NODE.json")
    checks = {"all_saved_complete_scientific_control_outputs_exact": comparison["all_protected"],
        "all_current_actual_complete_candidate_guards": all(candidate["checks"].values()),
        "all_current_input_source_identities_unchanged": sources == [data_identity(path) for path in paths]
            and inputs == [data_identity(row["path"]) for row in inputs],
        "actual_owned_node_completed": node["status"] == "completed" and node["returncode"] == 0,
        "both_original_private_bindings_restored": bindings_restored(read(OUT / "fused/BINDING_RESTORATION.json"))}
    status = "complete_candidate_science_exact_cost_screen_only" if all(checks.values()) else "failed_scientific_protection"
    atomic_json(OUT / "RESULT.json", {"status": status,
        "checks": checks, "comparison": comparison, "sources": sources, "inputs": inputs,
        "saved_control_complete_distance_stage_seconds": original["complete_distance_stage_seconds"],
        "current_candidate_complete_distance_stage_seconds": candidate["complete_distance_stage_seconds"],
        "saved_control_complete_seconds": original["complete_seconds"], "current_candidate_complete_seconds": candidate["complete_seconds"],
        "setup_seconds": setup_seconds, "complete_entry_seconds": time.perf_counter() - STARTED,
        "old_saved_nonoverlapping_fees_seconds": closure["first_saved_child_seconds"],
        "requires_fresh_three_pairs_supplements_and_fee_check_before_any_acceptance": True, "new_optimization_count": 0})
    print(json.dumps({"status": status, "checks": checks,
        "saved_control_phase_seconds": original["complete_distance_stage_seconds"], "candidate_phase_seconds": candidate["complete_distance_stage_seconds"]}), flush=True)
    if not all(checks.values()):
        raise AssertionError("The actual complete integer reduction recheck failed")


if __name__ == "__main__":
    main()
