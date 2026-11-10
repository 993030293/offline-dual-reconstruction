"""Check explicit missing-odometry handling against raw evidence and real maps."""

import argparse
import copy
import json
from pathlib import Path

import pycolmap

from pose_gap_candidates import select_pose_gap_candidates
from run_metric_refinement import bind_supported_priors, verified_pose_gap_names
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pose-gap-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (args.initial / "frame_decisions.jsonl").read_text().splitlines()]
    lookup = {row["image_name"]: row for row in rows if row["coarse_pose_valid"] and not row["excluded_role"]
              and row["decision"] in ("eligible_not_selected", "selected_initial_sfm_pending")}
    model = pycolmap.Reconstruction(args.model)
    allowed = verified_pose_gap_names(args.initial, args.pose_gap_decisions)
    priors, gaps = bind_supported_priors(model, lookup, allowed)
    checks = {"all_six_raw_gaps_remain_registered": len(gaps) == 6 and set(gaps) == allowed,
        "no_missing_support_image_has_prior": not (set(gaps) & {row["image_name"] for row in priors}),
        "every_registered_camera_has_explicit_role": len(priors)+len(gaps) == model.num_reg_images()}

    def rejects(name, operation):
        try:
            operation()
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False

    rejects("strict_prior_contract_rejects_real_gap_map", lambda: bind_supported_priors(model, lookup, set()))
    rejects("undeclared_gap_is_rejected", lambda: bind_supported_priors(model, lookup, allowed-{gaps[0]}))
    injected = dict(lookup)
    injected[gaps[0]] = {"C_W_m": [0, 0, 0]}
    rejects("fabricated_prior_is_rejected", lambda: bind_supported_priors(model, injected, allowed))
    receipt = json.loads(args.pose_gap_decisions.read_text())
    receipt["extrapolation_used"] = True
    tampered = args.output / "TAMPERED_RECEIPT_NEGATIVE_CASE.json"
    atomic_json(tampered, receipt)
    rejects("tampered_gap_receipt_is_rejected", lambda: verified_pose_gap_names(args.initial, tampered))
    manifest = json.loads((args.initial / "INITIAL_FRAME_MANIFEST.json").read_text())
    bad_rows = copy.deepcopy([row for row in rows if row["image_name"] in allowed])
    bad_rows[0]["C_W_m"] = [0, 0, 0]
    rejects("invented_pose_in_raw_gap_row_is_rejected", lambda: select_pose_gap_candidates(
        bad_rows, manifest["policy"], manifest["receipt"]["sharpness_min"]))
    receipt = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "registered_images": model.num_reg_images(), "active_position_priors": len(priors),
        "visual_only_images": gaps, "real_BA_verified_here": False, "counted": False,
        "source": data_identity(Path(__file__)), "prior_binding_source": data_identity(Path(__file__).with_name("run_metric_refinement.py"))}
    atomic_json(args.output / "RESULT.json", receipt)
    print(json.dumps(receipt), flush=True)
    if not all(checks.values()):
        raise AssertionError("Partial prior provenance checks failed")


if __name__ == "__main__":
    main()
