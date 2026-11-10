"""Account for the actual failed family and current complete consumer costs."""

from pathlib import Path
import statistics

import profile_quality_corner_cost as reference

OUT = reference.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"


def cost_snapshot(base, complete, extra_seconds=0.0):
    rows = []
    original_paths = sorted((OUT / "gm001").glob("LEASE_*.json"))
    old = reference.load(OUT / "gm001/FINAL_SCREEN_PUBLICATION_001.json")
    paths = original_paths + [OUT / "rm001/LEASE_P0.json", OUT / "post_corner_cost_001/LEASE_P0.json"]
    paths += sorted(base.glob("LEASE_*.json"))
    for path in paths:
        lease = reference.load(path)
        if "child_elapsed_seconds" in lease:
            rows.append({"path": str(path), "seconds": lease["child_elapsed_seconds"], "returncode": lease["returncode"]})
    old_seconds = sum(row["seconds"] for row in rows if Path(row["path"]).parent == OUT / "gm001")
    if len(original_paths) != 5 or abs(old_seconds - old["all_five_actual_child_seconds"]) > 1e-8:
        raise ValueError("The original failed scalar-family fee ledger changed")
    leaf = sum(row["whole_seconds"] for row in complete["runs"])
    comparison = sum(row["comparison_seconds_charged_each_side"] for row in complete["pairs"])
    old_publication = old["publication_verification_seconds_before_write"]
    diagnostic = reference.load(base / "REVIEW_PROCESS_DIAGNOSTIC_001.json")
    investigation = diagnostic["measured_readonly_diagnostic_seconds"] + diagnostic["first_readonly_diagnostic_failed_tool_command_wall_seconds"]
    shared = sum(row["seconds"] for row in rows) - leaf - comparison + old_publication + investigation + extra_seconds
    if shared < 0:
        raise ValueError("The measured original vector-family fee ledger became negative")
    medians = {name: {mode: statistics.median(row[name][mode] for row in complete["pairs"]) + shared / 6
        for mode in ("current", "vector")} for name in ("phase_seconds", "whole_seconds")}
    return {"lease_rows": rows, "old_R144_five_leases_including_failed_review_seconds": old_seconds,
        "old_publication_measured_seconds": old_publication, "additional_saved_process_investigation_seconds": investigation,
        "leaf_windows_already_counted_seconds": leaf,
        "physical_comparison_already_counted_seconds": comparison, "known_extra_fee_seconds": shared,
        "amortization_sides": 6, "complete_known_medians": medians,
        "net_reductions": {name: 1 - values["vector"] / values["current"] for name, values in medians.items()},
        "whole_median_ratio": medians["whole_seconds"]["vector"] / medians["whole_seconds"]["current"],
        "text_research_wall_time_not_metered": True, "pending_write_stdout_exit_not_assumed_zero": True,
        "not_total_first_deployment_cumulative_research_ROI": True}
