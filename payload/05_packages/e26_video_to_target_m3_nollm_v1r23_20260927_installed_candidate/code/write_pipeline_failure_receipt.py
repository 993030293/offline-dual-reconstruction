"""Write an immutable machine-readable receipt for an unattended pipeline failure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid


STAGE_MARKERS_BEFORE_TRAIN = (
    ("S00_raw_identity", "00_contract/raw_identity_receipt.json"),
    ("S01_visual_workspace", "01_visual_workspace/workspace_receipt.json"),
    ("S02_machine_inventory", "02_machine/s02_machine_frame_inventory.json"),
    ("S03_metric_context", "03_metric_context/metric_time_support_receipt.json"),
    ("S04_trajectory_binding", "04_trajectory_binding/interpolated_camera_poses.json"),
    ("S05_sim3", "05_sim3/sim3.json"),
    ("S06_metric_pgsr_input", "06_pgsr_inlier/metric_input_receipt.json"),
)
RELIABLE_STAGE_MARKERS = (
    ("S06_target_roi", "06_pgsr_inlier/vehicle_roi_receipt.json"),
    ("S06_registered_lidar_anchors", "06_pgsr_inlier/stable_lidar/stable_lidar_receipt.json"),
)
STAGE_MARKERS_AFTER_ROI = (
    ("S06_pgsr_training", "06_pgsr_inlier/training_receipt.json"),
    ("S08_terminal_preview", "INTEGRATED_FINAL_STATUS.json"),
)


def write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--raw-mcap", required=True, type=Path)
    parser.add_argument("--error-type", required=True)
    parser.add_argument("--error-message", required=True)
    parser.add_argument("--output-root-was-preexisting", action="store_true")
    args = parser.parse_args()

    run = args.output_root.resolve()
    completed = []
    failed_stage = "preflight_before_output_root"
    if run.is_dir() and not args.output_root_was_preexisting:
        failed_stage = "unknown_after_output_root_creation"
        stage_markers = (
            *STAGE_MARKERS_BEFORE_TRAIN,
            *(RELIABLE_STAGE_MARKERS if args.profile == "ReliableLidar" else ()),
            *STAGE_MARKERS_AFTER_ROI,
        )
        for stage, relative in stage_markers:
            marker = run / relative
            if marker.is_file():
                completed.append({"stage": stage, "marker": str(marker)})
            elif failed_stage == "unknown_after_output_root_creation":
                failed_stage = stage
        receipt = run / "PIPELINE_FAILURE.json"
    else:
        receipt = run.with_name(f"{run.name}.FAILURE.json")
    if receipt.exists():
        receipt = receipt.with_name(
            f"{receipt.stem}.{uuid.uuid4().hex}{receipt.suffix}"
        )
    payload = {
        "schema_version": "e26_unattended_pipeline_failure_v1",
        "status": "failed_closed",
        "failed_stage": failed_stage,
        "completed_stage_markers": completed,
        "selected_profile": args.profile,
        "raw_mcap": str(args.raw_mcap.resolve()),
        "requested_output_root": str(run),
        "output_root_was_preexisting": args.output_root_was_preexisting,
        "error_type": args.error_type,
        "error_message": args.error_message,
        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        "language_model_used": False,
        "visual_language_model_used": False,
        "manual_visual_decision_used": False,
        "derived_outputs_reusable": False,
        "recovery_rule": "Start a new output root after correcting the recorded cause; do not resume or overwrite this failed run.",
    }
    write_new(receipt, payload)
    print(receipt)


if __name__ == "__main__":
    main()
