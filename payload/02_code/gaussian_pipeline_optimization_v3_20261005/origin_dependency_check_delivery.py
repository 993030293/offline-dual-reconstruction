"""Check R115 direct-text links and machine evidence without document hashes."""

import argparse
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/origin_dependency_001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--allow-pending-goal-receipt", action="store_true")
parser.add_argument("--receipt", type=Path, required=True)
args = parser.parse_args()
if args.receipt.resolve().parent != OUT or args.receipt.exists():
    raise ValueError("Text delivery receipt must be fresh and R115-owned")
files = [DOC / "R115_ORIGIN_DEPENDENCY_BRIEF.md", DOC / "R115_ORIGIN_DEPENDENCY_RESULT.md",
         *sorted(OUT.glob("*.md"))]
links = []
pending = []
for path in files:
    text = path.read_text(encoding="utf-8")
    if "failed_verification" not in text and path.name not in ("SETUP_REPAIRS.md", "SOURCE_ATTRIBUTION.md"):
        raise AssertionError("R115 negative scope is missing from a final document")
    for match in re.finditer(r"\]\(<?([^)>\n]+)>?\)", text):
        target = match.group(1)
        if target.startswith(("https://", "http://")):
            links.append({"source": str(path), "target": target, "kind": "primary_URL_previously_read"})
            continue
        resolved = Path(re.sub(r":\d+$", "", target))
        if not resolved.is_absolute():
            raise AssertionError("A local final artifact link is not absolute")
        if not resolved.exists():
            if args.allow_pending_goal_receipt and resolved == OUT / "GOAL_COMPLETION.json":
                pending.append(str(resolved))
                continue
            raise FileNotFoundError(target)
        links.append({"source": str(path), "target": target, "kind": "existing_local_artifact"})
final = json.loads((OUT / "FINAL_RESULT.json").read_text(encoding="utf-8"))
review = json.loads((OUT / "REVIEW_REQUEST.json").read_text(encoding="utf-8"))
closure = json.loads((OUT / "CLOSURE.json").read_text(encoding="utf-8"))
pids = json.loads((OUT / "POST_CLOSE_PID_RECEIPT.json").read_text(encoding="utf-8"))
assert final["status"] == "failed_verification"
assert final["classification"] == "actual_candidate_runtime_guard_failure"
assert not final["counted"] and final["contribution"] == 0
assert final["cost"]["formal_pairs_completed"] == 0
assert final["cost"]["phase_reduction"] is None and final["cost"]["whole_cost_effect"] is None
assert final["primary"]["paired_cases"] == 118 and not final["primary"]["full_semantic_acceptance"]
assert not final["first_required_failure"]["GL_error_numeric_code_saved"]
assert not final["first_required_failure"]["byte_mismatch_claimed"]
assert not final["first_required_failure"]["exact_root_cause_confirmed"]
assert closure["frozen_source_input_count"] == final["closure"]["source_input_identities"] == 34
assert closure["complete_source_snapshot_count"] == final["closure"]["frozen_source_snapshots"] == 85
assert pids["lease_count"] == final["closure"]["leases"] == 7
assert pids["process_identity_count"] == final["closure"]["owned_recorded_controller_child_identities"] == 14
assert review["candidate_status"] == final["status"] and not review["promotion_requested"]
assert all(Path(path).is_file() for path in review["evidence"])
original = json.loads((OUT / "lifecycle_002/RESULT.json").read_text(encoding="utf-8"))
assert original["status"] == "validator_setup_failed"
payload = {"status": "text_JSON_links_and_negative_claim_consistency_passed",
           "direct_text_documents": [str(path) for path in files], "links": links,
           "pending_native_goal_receipt_only": list(dict.fromkeys(pending)),
           "document_hash_or_screenshot_used": False, "independent_review": False,
           "original_auto_catch_RESULT_preserved": True, "counted": False}
args.receipt.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"status": payload["status"], "direct_text_documents": len(files),
                  "checked_links": len(links), "pending_native_goal_receipt_only": bool(pending),
                  "counted": False}))
