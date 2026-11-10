"""Bind pre-execution candidate rejection to the current full source and receipts."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
V3 = ROOT/"02_code/gaussian_pipeline_optimization_v3_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def record(path, anchors):
    text = path.read_text(encoding="utf-8")
    evidence = []
    for anchor in anchors:
        hits = [(i, line) for i, line in enumerate(text.splitlines(), 1) if anchor in line]
        if not hits:
            raise ValueError("A current source no longer has the audited mechanism: "+anchor)
        evidence.append({"anchor": anchor, "matches": [{"line": i, "text": line} for i, line in hits]})
    return {"identity": data_identity(path), "text_evidence": evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    q54_path = ROOT/"04_outputs/gaussian_pipeline_optimization_v2_20261005/q54_campaign_001/RESULT.json"
    typed_path = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/typed_trajectory_001/RESULT.json"
    q54, typed = (json.loads(path.read_text(encoding="utf-8")) for path in (q54_path, typed_path))
    if q54["status"] != "rejected_insufficient_wall_benefit" or typed["status"] != "failed_verification" or len(typed["pairs"]) != 3:
        raise ValueError("An older measured negative candidate changed its stated acceptance")
    shared = {"decision": "rejected", "counted": False, "rejection_phase": "pre_execution_source_deduplication",
        "implemented_new_candidate": False, "three_current_pairs": "not_applicable",
        "generic_external_recovery_trials": "not_applicable_no_new_implementation",
        "scope": "reject this proposed duplicate; not a proof the whole domain has no further optimization"}
    pa01 = {**shared, "candidate": "PA01", "proposal": "Bulk preload and keyed structured quality-score access",
        "reason": "The proposed complete-source-key table and batch-load lookup mechanism is already the measured failed Q54 family. No independently different access mechanism was selected. Changing table format, thread count or reusing another runtime composition does not create a new candidate in this batch.",
        "sources": [record(V1/"quality_engine.py", ["class ScoreCache", "def key(", "def read(", "quality_sha256", "cached = cache.read(key)"]),
            record(V2/"columnar_quality_scores.py", ["def build_table", "class ColumnarScoreCache", "self.lookup =", "self.key =", "def read(", "manifest[\"contract\"]"]),
            record(V1/"run_initial_pipeline.py", ["cache_contract =", "cache = ScoreCache", "score_messages("])],
        "prior_failure_receipt": data_identity(q54_path), "prior_status": q54["status"],
        "claims_not_made": ["No new speed result", "Not accepted", "Not a theorem against SQLite or other storage algorithms", "Old Q43 is not a score cache"]}
    pa03 = {**shared, "candidate": "PA03", "proposal": "Contiguous complete CDR pose decode and batch timestamp lookup",
        "reason": "The complete 7+36 float64 decode/typed allocation mechanism is already failed R89; integer batch support and searchsorted/SLERP already belong to accepted P18. Calibration-body memoization is existing R99. Merely composing or changing buffers/threads would repeat these families. No independently different parse or lookup candidate was selected.",
        "sources": [record(V1/"run_initial_pipeline.py", ["stamps, poses, covariance", "cdr.header()", "range(7)", "range(36)", "curve = baseline.CoarseCameraCurve"]),
            record(V3/"typed_odometry_cdr.py", ["def decode_view", "np.frombuffer", "MAXIMUM_RECORDS", "covariance_calls", "_typed_finish"]),
            record(V1/"pose_interpolation.py", ["def coarse_camera_at_many", "np.searchsorted", "curve.slerp", "offline_future_bracketing"]),
            record(V3/"calibration_body_cache.py", ["decode_camera_info", "camera_info"])] ,
        "prior_failure_receipt": data_identity(typed_path), "prior_status": typed["status"],
        "prior_full_phase_reduction": typed["stage_reduction"],
        "claims_not_made": ["No new trajectory timing", "No full LIO replay", "No new fast rough-pose estimator", "Not a theorem against new parsing algorithms"]}
    for row in (pa01, pa03):
        atomic_json(args.output/(row["candidate"]+"_REJECTION.json"), row)
    atomic_json(args.output/"RESULT.json", {"status": "source_rejections_bound", "PA01": pa01, "PA03": pa03, "counted": False})


if __name__ == "__main__":
    main()
