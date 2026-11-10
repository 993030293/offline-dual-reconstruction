"""Exercise genuine prior-activation and failure contracts on a synthetic model."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pycolmap

from common import (ba_config, ba_options, configure_stdout, make_priors,
                    solver_receipt, validate_prior_rows, write_json)


def synthetic():
    pycolmap.set_random_seed(20261002)
    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 1
    options.num_frames_per_rig = 6
    options.num_points3D = 60
    model = pycolmap.synthesize_dataset(options)
    rows = [{"image_id": image.image_id, "image_name": image.name,
             "C_W_m": image.projection_center().tolist(), "position_covariance_m2": (np.eye(3) * .01).tolist(),
             "prior_active": True} for image in model.images.values()]
    noise = pycolmap.SyntheticNoiseOptions()
    noise.point3D_stddev = .01
    noise.rig_from_world_translation_stddev = .01
    pycolmap.synthesize_noise(noise, model)
    return model, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    model, rows = synthetic()
    checked = []
    for name, change in [
        ("unknown_image_id", lambda r: r[0].update(image_id=999999)),
        ("singular_covariance", lambda r: r[0].update(position_covariance_m2=np.zeros((3, 3)).tolist())),
        ("indefinite_covariance", lambda r: r[0].update(position_covariance_m2=np.diag([-1, 1, 1]).tolist())),
        ("too_few_priors", lambda r: [x.update(prior_active=i < 2) for i, x in enumerate(r)]),
        ("collinear_prior_positions", lambda r: [x.update(C_W_m=[i, 0, 0]) for i, x in enumerate(r)]),
    ]:
        bad = copy.deepcopy(rows)
        change(bad)
        try:
            validate_prior_rows(model, bad)
        except ValueError:
            checked.append(name)
        else:
            raise AssertionError(f"Invalid prior input was accepted: {name}")
    config = ba_config(model)
    expected_visual = config.num_residuals(model)
    priors = make_priors(model, rows)
    options = pycolmap.PosePriorBundleAdjustmentOptions()
    options.alignment_ransac.max_error = .2
    options.alignment_ransac.random_seed = 20261002
    options.ceres.prior_position_loss_function_type = pycolmap.LossFunctionType.CAUCHY
    summary = pycolmap.create_pose_prior_ceres_bundle_adjuster(
        ba_options(20), options, config, priors, model).solve()
    info = solver_receipt(summary)
    if not info["usable"] or info["num_residuals"] != expected_visual + 3 * len(priors):
        raise AssertionError("Normal synthetic priors did not enter the problem")
    checked.append("normal_prior_activation")
    wrong = copy.deepcopy(priors)
    for p in wrong:
        identity = p.corr_data_id
        identity.id += 10000
        p.corr_data_id = identity
    fallback = pycolmap.Reconstruction(model)
    fallback_config = ba_config(fallback)
    fallback_expected = fallback_config.num_residuals(fallback)
    fallback_summary = pycolmap.create_pose_prior_ceres_bundle_adjuster(
        ba_options(2), options, fallback_config, wrong, fallback).solve()
    fallback_info = solver_receipt(fallback_summary)
    if fallback_info["num_residuals"] != fallback_expected:
        raise AssertionError("The expected no-prior fallback was not observed")
    checked.append("native_wrong_id_fallback_detected_by_residual_count")
    write_json(args.run / "01_priors/CONTRACT_TEST_RECEIPT.json", {
        "status": "passed", "tests": checked, "normal": info, "wrong_id_native_fallback": fallback_info,
        "purpose": "engineering_contract_preflight_not_Box2_accuracy_evidence"})
    print(f"Prior contract preflight passed: {len(checked)} cases")


if __name__ == "__main__":
    configure_stdout()
    main()
