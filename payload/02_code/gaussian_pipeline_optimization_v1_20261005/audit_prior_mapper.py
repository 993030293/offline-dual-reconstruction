"""Reconcile a native final BA report with its logged observation changes."""

import argparse
import json
from pathlib import Path
import re

import pycolmap

from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    target = args.run/"ACTIVATION_AUDIT.json"
    if target.exists():
        raise FileExistsError(target)
    result = json.loads((args.run/"RESULT.json").read_text())
    contract = json.loads((args.run/"CONTRACT.json").read_text())
    log = (args.run/"mapper.log").read_text(errors="replace")
    reports = list(re.finditer(r"Pose Prior Bundle adjustment report\s+Residuals\s*:\s*(\d+)", log))
    if len(result["models"]) != 1 or not reports:
        raise ValueError("Final report cannot be unambiguously bound to one output model")
    model = pycolmap.Reconstruction(result["models"][0]["path"])
    last = reports[-1]
    suffix = log[last.end():]
    if "Registering image" in suffix or "Bundle adjustment report" in suffix:
        raise ValueError("Further mapping/BA after the selected report makes this audit ambiguous")
    counts = {kind: sum(map(int, re.findall(kind+r" observations: (\d+)", suffix))) for kind in ["Completed", "Merged", "Filtered"]}
    if counts["Merged"]:
        raise ValueError("Nonzero post-BA merges need a richer observation lineage audit")
    final_observations = model.compute_num_observations()
    observations_at_ba = final_observations-counts["Completed"]+counts["Filtered"]
    actual = int(last.group(1))
    visual = 2*observations_at_ba
    prior_ids = {row["image_id"] for row in contract["prior_bindings"]}
    registered_ids = {image.image_id for image in model.images.values() if image.has_pose}
    active_expected = len(prior_ids & registered_ids)
    passed = actual == visual+3*active_expected and active_expected == model.num_reg_images()
    receipt = {"status": "native_final_position_factor_count_verified" if passed else "failed_residual_reconciliation",
        "registered_images": model.num_reg_images(), "bound_position_priors": active_expected,
        "native_report_residuals": actual, "final_model_observations": final_observations,
        "logged_post_BA_observation_changes": counts, "reconstructed_BA_observations": observations_at_ba,
        "expected_visual_residuals": visual, "observed_extra_residuals": actual-visual,
        "expected_position_residuals": 3*active_expected, "all_expected_position_factors_activated": passed,
        "native_prior_report_count": len(reports), "log_identity": data_identity(args.run/"mapper.log"),
        "audit_scope": "one final native BA report with zero subsequent image registrations/BA/track merges; intermediate fallback counts remain in run receipt",
        "absolute_accuracy_claim": False, "counted": False, "source": data_identity(Path(__file__))}
    atomic_json(target, receipt)
    print(json.dumps(receipt), flush=True)
    if not passed:
        raise AssertionError("Native residual count does not prove the intended position priors")


if __name__ == "__main__":
    main()
