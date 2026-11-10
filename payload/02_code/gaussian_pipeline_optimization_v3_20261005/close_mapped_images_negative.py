"""Close the completed negative review without changing acceptance criteria."""

import json
from datetime import datetime, timezone
from pathlib import Path
import time

import psutil


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def same_process(identity):
    try:
        process = psutil.Process(identity["pid"])
        return abs(process.create_time() - identity["creation_time"]) < 0.01 and process.is_running()
    except psutil.NoSuchProcess:
        return False


def main():
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    output = root / "04_outputs/gaussian_pipeline_optimization_v3_20261005/mi001"
    review = read(output / "r0/REVIEW.json")
    lease = read(output / "NEGATIVE_REVIEW_LEASE_001.json")
    guard = read(output / "NEGATIVE_REVIEW_GUARD_001.json")
    owner = read(output.parent / "coordination_001/LOCK_OWNER.json")
    state = read(root / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    identities = list(guard["owned_process_identities"])
    identities.extend({"pid": lease[key + "_pid"], "creation_time": lease[key + "_creation_time"]}
                      for key in ("guard", "child"))
    live = [identity for identity in identities if same_process(identity)]
    checks = {
        "negative_original_quality_gate_retained": review["status"] == "R131_closed_negative_required_quality_failure"
            and len(review["quality_failed_rows"]) == 30 and all(review["checks"].values()),
        "negative_review_completed_normally": guard["status"] == "completed" and guard["returncode"] == 0,
        "actual_review_processes_gone": not live,
        "same_owner_lease_released": lease["status"] == "child_exited" and lease["returncode"] == 0
            and bool(lease["lease_released_utc"]) and owner["status"] == "released"
            and owner["owner"] == lease["owner"],
        "parent_accepted17_unchanged": state["additional_accepted_count"] == 17
            and len(set(state["accepted_mechanisms"])) == 17 and review["new_optimization_count"] == 0,
    }
    result = {
        "status": "closed_negative_required_quality_failure" if all(checks.values()) else "failed_negative_closure",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "review_kind": "root_self_review_no_subagents_or_side_chats",
        "checks": checks, "review": str(output / "r0/REVIEW.json"),
        "own_review_process_identities_checked": identities, "live_instances": live,
        "new_optimization_count": 0, "parent_additional_accepted_count": 17,
        "known_nonoverlapping_children_and_campaign_seconds_including_review":
            review["recorded_children_and_campaign_seconds"] + lease["child_elapsed_seconds"],
        "cost_gaps": review["unmeasured_envelope"],
        "negative_review_lease_seconds": lease["child_elapsed_seconds"],
        "closure_observation_seconds": time.perf_counter() - started,
        "complete_three_seed_training_recovery_acceptance": False,
        "quality_cause": review["quality_root_cause"],
        "source_inputs_production_services_modified_by_closure": False,
    }
    target = output / "NEGATIVE_CLOSURE_001.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))
    if not all(checks.values()):
        raise AssertionError("The negative closure did not satisfy its exact scope")


if __name__ == "__main__":
    main()
