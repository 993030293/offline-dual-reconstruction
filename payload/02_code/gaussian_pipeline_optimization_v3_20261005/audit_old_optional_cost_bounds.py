"""Screen old complete candidates using unchanged V5 cost definitions."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bound(name, original, candidate, fees, sources):
    saving = original - candidate
    fee = sum(fees.values())
    upper = saving / (original + fee / 6)
    return {"candidate": name, "original_saved_complete_phase_seconds": original,
            "candidate_saved_complete_phase_seconds": candidate, "known_extra_fee_lower_bound_seconds": fees,
            "six_side_shared_fee_lower_bound_seconds": fee / 6,
            "V5_stage_reduction_upper_bound_not_an_accepted_effect": upper,
            "minimum_V5_stage_reduction": .03, "below_gate_before_other_required_costs": upper < .03,
            "receipt_sources": [data_identity(path) for path in sources],
            "semantic_or_current_input_external_revalidation_not_run_after_sufficient_cost_rejection": True,
            "new_optimization_count": 0}


def main():
    base = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
    output = base / "v5_residual_cost_002"
    output.mkdir(parents=True, exist_ok=False)
    old = read(base / "acceptance_v2_reaudit_001/RESIDUAL_COST_SCREEN_001.json")
    item = next(row for row in old["items"] if row["candidate"] == "R101")
    paths = [base / "neighbor_geometry_001_process.json", base / "ng002_process.json"]
    fees = {path.name: read(path)["elapsed_seconds"] for path in paths}
    r101 = bound("R101", item["original_phase_median_seconds"]["incumbent"],
                 item["original_phase_median_seconds"]["parallel"], fees,
                 [*paths, base / "ng003/RESULT.json", base / "acceptance_v2_reaudit_001/RESIDUAL_COST_SCREEN_001.json"])
    path = base / "sparse_ply_oracle_001/CLOSURE.json"
    r111_source = read(path)
    first = next(row for row in r111_source["actual_own_nodes"] if row["prefix"] == "sparse_ply_profile_001")
    r111 = bound("R111", r111_source["recomputed_medians"]["native"]["phase_seconds"],
                 r111_source["recomputed_medians"]["columns"]["phase_seconds"],
                 {"original_failed_campaign001": first["elapsed_seconds"]},
                 [path, base / "sparse_ply_oracle_001/campaign_002/RESULT.json"])
    state = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    times = state["R118_complete_consumer_screen_raw_seconds"]
    if set(times) != {"native", "candidate"}:
        raise ValueError("The original complete R118 timing schema needs explicit inspection")
    r118 = bound("R118", times["native"], times["candidate"],
        {"already_saved_old_new_private_lease_fee_lower_bound": state["R118_complete_consumer_screen_old_and_new_lease_child_fees_excluding_R115_seconds"]},
        [base / "gpu_origin_dependency_001/COMPLETE_SCREEN_CLOSURE_001.json"])
    rows = [r101, r111, r118]
    atomic_json(output / "RESULT.json", {"status": "old_V5_cost_bounds_closed0" if all(row["below_gate_before_other_required_costs"] for row in rows) else "needs_review",
        "rows": rows, "old_failed_receipts_unchanged": True,
        "new_optimization_count": 0, "parent_accepted_count": state["additional_accepted_count"],
        "review_kind": "root_read_only_arithmetic_cost_screen_not_full_revalidation",
        "unknown_historical_research_text_time_not_counted_as_zero_or_total_ROI": True,
        "source": data_identity(Path(__file__))})
    if not all(row["below_gate_before_other_required_costs"] for row in rows):
        raise AssertionError("A residual old candidate needs its full remaining gates")
    print({row["candidate"]: row["V5_stage_reduction_upper_bound_not_an_accepted_effect"] for row in rows})


if __name__ == "__main__":
    main()
