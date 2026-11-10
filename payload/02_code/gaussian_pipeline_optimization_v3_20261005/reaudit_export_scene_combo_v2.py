"""Recheck the saved full R112/Q68 chain without restarting another chat."""

import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
sys.path[:0] = [str(CODE / "parallel_a_004"), str(CODE),
    str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]

import psutil
from runtime import data_identity, atomic_json
from profile_sparse_ply_oracle import compare_outputs
from machine_json import exact_json_values
from run_pair import normalized

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/parallel_a_004"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50/parallel_001/a_004"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    folder = OUT / "v2r0"
    folder.mkdir(exist_ok=False)
    contract = read(OUT / "freeze_001_work/CONTRACT.json")
    cost = read(DOC / "FULL_COST_REVISION_001.json")
    checks = {"V1_full_cost_rejection_preserved": cost["phase_reduction_fraction"] < .1
        and cost["new_mechanism_contribution"] == 0,
        "three_original_fresh_interleaved_orders": contract["orders"] == [
            ["native", "combined"], ["combined", "native"], ["native", "combined"]],
        "all1086_source_input_identities_current": len(contract["sources_and_inputs"]) == 1086
            and all(data_identity(row["path"]) == row for row in contract["sources_and_inputs"])}
    pairs, comparisons = [], []
    for index in range(3):
        pair = read(OUT / f"pair{index}_001_work/RESULT.json")
        sides = [Path(pair["rows"][mode]["folder"]) for mode in ("native", "combined")]
        proof = compare_outputs(sides[0] / "map/export", sides[1] / "map/export")
        checks[f"pair{index}_all1046_byte_files_and4_scientific_JSON_now"] = (
            proof["complete_byte_pairs"] == 1046 and proof["complete_JSON_pairs"] == 4
            and proof["complete_compared_bytes"] == 746975609)
        scenes = [read(side / "scene/RESULT.json") for side in sides]
        fields = ("rows", "initial_Gaussians", "camera_manifest", "camera_extent", "before_RNG", "after_RNG",
            "complete_scene_extended", "actual_combination_config", "full_source_identity")
        for field in fields:
            exact_json_values(normalized(scenes[0][field], sides[0]), normalized(scenes[1][field], sides[1]))
        exact_json_values(*(normalized(scene["identity_phase"]["native_manifest"], side)
            for scene, side in zip(scenes, sides)))
        checks[f"pair{index}_all1016_actual_Scene_fields_initialization_RNG_and_manifest_exact_now"] = (
            len(scenes[0]["rows"]) == len(scenes[1]["rows"]) == 1016
            and all(all(scene["checks"].values()) for scene in scenes))
        checks[f"pair{index}_original_generated_scene_assets_bytes_now"] = all(
            (sides[0] / "scene/Scene/scene" / name).read_bytes() ==
            (sides[1] / "scene/Scene/scene" / name).read_bytes() for name in ("input.ply", "cameras.json"))
        checks[f"pair{index}_all_private_owners_hooks_threads_and_weakrefs_closed"] = all(
            scene["all_actual_Scene_Gaussian_camera_weakrefs_released"] and scene["actual_new_export_used"]
            and scene["actual_chain_binding"]["source_bound_preserved_main_restored"]
            and all(scene["complete_close"][key] for key in ("native_evaluator_bindings_restored",
                "owned_prefetch_and_identity_threads_absent", "no_Scene_reference_retained")) for scene in scenes)
        checks[f"pair{index}_all348_source_copy_bytes_across_six_sides_current"] = all(
            (side / "map/source_snapshot" / (f"{number:03d}_" + Path(row["path"]).name)).read_bytes() ==
            Path(row["path"]).read_bytes() for side in sides for number, row in enumerate(contract["source_rows"]))
        comparisons.append(proof)
        pairs.append(pair)
    checks["all_original_pair_guards_retained"] = all(all(pair["checks"].values()) for pair in pairs)
    for name in ("generic_001", "camera_generic_001"):
        generic = read(OUT / (name + "_work/RESULT.json"))
        checks[name + "_all_original_error_cancel_and_shape_guards_retained"] = all(generic["checks"].values())
    for item in cost["receipt_inputs"]:
        checks[item["name"] + "_actual_old_fee_receipt_unchanged"] = read(item["path"]) == item["receipt"]
    reconstructed = []
    for raw, full in zip(pairs, cost["pairs"]):
        fee = full["additional_full_cost_each_side_seconds"]
        checks[f"pair{raw['pair']}_full_cost_formula_recomputed"] = all(
            abs(raw["rows"][mode][field + "_seconds"] + fee - full["full_" + mode + "_" + field + "_seconds"]) < 1e-9
            for mode in ("native", "combined") for field in ("phase", "whole"))
        reconstructed.append({mode: {field: raw["rows"][mode][field + "_seconds"] + fee
            for field in ("phase", "whole")} for mode in ("native", "combined")})
    resources = read(DOC / "RESOURCE_FINAL_001.json")
    windows, linux = resources["Windows"], resources["Linux"]
    live = []
    for row in windows:
        try:
            if psutil.Process(row["pid"]).create_time() == row["creation_time"]:
                live.append(row)
        except psutil.NoSuchProcess:
            pass
    code = "import json,psutil; rows=json.loads(" + repr(json.dumps(linux)) + "); live=[]\n"
    code += "for i in rows:\n try:\n  p=psutil.Process(i['pid'])\n  if p.create_time()==i['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE: live.append(i)\n except psutil.NoSuchProcess: pass\nprint(json.dumps(live))"
    live_linux = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=30))
    checks["all116_Windows_and7_Linux_saved_instances_actually_gone_now"] = (
        len(windows) == 116 and len(linux) == 7 and not live and not live_linux)
    result = {"status": "V2_exact_combination_review_verified_final_cost_pending" if all(checks.values()) else "failed_verification",
        "checks": checks, "prior_full_cost_pairs": reconstructed, "comparisons": comparisons,
        "sources_inputs_checked": len(contract["sources_and_inputs"]), "source_rows": len(contract["source_rows"]),
        "Windows_checked": len(windows), "Linux_checked": len(linux), "live_Windows": live, "live_Linux": live_linux,
        "owner": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "source": data_identity(Path(__file__)), "review_seconds_inside_script": time.perf_counter() - STARTED,
        "new_mechanism_contribution": 0, "prior_complete_cost": str(DOC / "FULL_COST_REVISION_001.json"),
        "review_kind": "root_self_review_no_subagents_or_other_chat_dispatch"}
    atomic_json(folder / "REVIEW.json", result)
    print({key: result[key] for key in ("status", "checks", "review_seconds_inside_script")}, flush=True)
    if not all(checks.values()):
        raise AssertionError("A full saved combination protection or original cost formula differs")


if __name__ == "__main__":
    main()
