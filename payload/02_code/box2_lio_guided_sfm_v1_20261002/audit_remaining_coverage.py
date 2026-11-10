"""Audit the supported remaining recording without adding a second reconstruction run."""

import argparse
from pathlib import Path

import numpy as np

from augment_views import roi_contract, target_window
from common import MCAP, configure_stdout, image_quality, load_mcap_tools, read_json, write_json
from feature_localization import pose_curve
from run_prior_ba import quantiles
import pycolmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    run = parser.parse_args().run.resolve()
    roi, curve = roi_contract(run), pose_curve()
    selection = read_json(run / "03_coverage_augmented/SELECTION_MANIFEST.json")
    camera = next(iter(pycolmap.Reconstruction(run / "00_contract/initial_model").cameras.values()))
    all_body_sample_camera_centers = np.asarray([curve.at(int(t))["C_W_m"] for t in curve.times])
    McapIndex, _, decode, demosaic, _ = load_mcap_tools()
    index = McapIndex(MCAP)
    total = index.message_count("/sensing/camera/image_raw")
    lo, hi = round((total - 1) * .1), round((total - 1) * .68)
    outside = sorted(set(np.linspace(0, lo - 1, 64, dtype=int)) | set(np.linspace(hi + 1, total - 1, 128, dtype=int)))
    gate = read_json(run / "00_contract/split_manifest.json")["quality_gate"]
    top_center = np.r_[roi["center"][:2], roi["maximum"][2]]
    rows = []
    for frame_index, message in index.iter_selected_messages("/sensing/camera/image_raw", outside):
        stamp, _, raw = decode(message["payload"])
        row = {"camera_frame": frame_index, "camera_header_ns": stamp}
        try:
            rough = curve.at(int(stamp))
        except ValueError as error:
            rows.append({**row, "status": "unsupported", "reason": str(error)})
            continue
        center = np.asarray(rough["C_W_m"])
        vector = center - top_center
        angle = float(np.rad2deg(np.arctan2(vector[2], np.linalg.norm(vector[:2]))))
        window = target_window(camera, rough, roi)
        photo = demosaic(raw)
        global_quality = image_quality(photo)
        target_quality = None if window is None else image_quality(photo[window[1]:window[3], window[0]:window[2]])
        eligible = global_quality["sharpness"] >= gate["sharpness_min"] and global_quality["clipped_fraction"] <= .15
        eligible &= target_quality is not None and target_quality["sharpness"] >= selection["target_threshold"] and target_quality["clipped_fraction"] <= .15
        rows.append({**row, "status": "diagnostic_only_not_selected", "C_W_m": center.tolist(),
            "target_window": window, "global_quality": global_quality, "target_quality": target_quality,
            "same_quality_gate_eligible": bool(eligible), "potential_top_incidence_angle_deg": angle})
    eligible = [r for r in rows if r.get("same_quality_gate_eligible")]
    write_json(run / "03_coverage_augmented/REMAINING_COVERAGE_AUDIT.json", {"status": "diagnostic_completed",
        "sampled_outside_original_interval": len(rows), "rows": rows,
        "eligible_outside_samples": len(eligible),
        "max_optical_camera_height_at_all_odometry_knots_m": float(all_body_sample_camera_centers[:, 2].max()),
        "fixed_ROI_top_height_m": float(roi["maximum"][2]),
        "outside_eligible_top_incidence_deg": quantiles([r["potential_top_incidence_angle_deg"] for r in eligible]),
        "second_selection_or_training_performed": False, "H_or_D_cal_used": False,
        "boundary": "sampled_coverage_and_recorded_pose_diagnostic_not_occlusion_verified_top_surface_completeness"})
    print({"outside_samples": len(rows), "eligible": len(eligible),
           "max_camera_height_m": float(all_body_sample_camera_centers[:, 2].max()),
           "top_incidence_deg": quantiles([r["potential_top_incidence_angle_deg"] for r in eligible])})


if __name__ == "__main__":
    configure_stdout()
    main()
