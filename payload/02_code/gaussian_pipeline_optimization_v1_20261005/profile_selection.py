"""Check full-sequence decision equivalence and measure batched coverage selection."""

import argparse
import copy
import importlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

from runtime import atomic_json, data_identity
from selection_engine import choose_batched


def fingerprint(rows, selected, eligible):
    return {"selected": [row["camera_frame"] for row in selected],
            "eligible": [row["camera_frame"] for row in eligible],
            "decisions": [(row["camera_frame"], row["decision"], row.get("selection_reason")) for row in rows]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project/"02_code/box2_initial_pose_quality_v2_20261002"))
    baseline = importlib.import_module("initial_selection")
    source = project/"04_outputs/box2_initial_pose_quality_v2_20261002T193102/01_initial_frames/frame_decisions.jsonl"
    receipt = json.loads(source.with_name("SELECTION_RECEIPT.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    historical = fingerprint(rows, sorted([row for row in rows if row["decision"] == "selected_initial_sfm_pending"],
                            key=lambda row: row["camera_header_ns"]),
                            [row for row in rows if row["decision"] in ("selected_initial_sfm_pending", "eligible_not_selected")])
    for row in rows:
        row.pop("selection_reason", None)
        row.pop("decision", None)
    atomic_json(args.output/"CONTRACT.json", {"input": data_identity(source), "frames": len(rows),
        "input_role": "raw_quality_and_coarse_poses_only", "threshold": receipt["sharpness_min"],
        "policy": baseline.POLICY, "repetitions": 3, "maximum_seconds_per_trial": 300,
        "acceptance": "all 7338 decisions/reasons and 1002 IDs identical; median wall reduction at least 10 percent",
        "historical_decisions_used_only_as_evaluation": True, "new_training_inputs_or_GT": False})
    snapshot = args.output/"source_snapshot"
    snapshot.mkdir()
    for path in [Path(__file__), Path(__file__).with_name("selection_engine.py"), Path(baseline.__file__)]:
        shutil.copyfile(path, snapshot/path.name)
    records = []
    for repetition in range(3):
        for variant in (["baseline", "batched"] if repetition != 1 else ["batched", "baseline"]):
            local = copy.deepcopy(rows)
            before = time.perf_counter()
            if variant == "baseline":
                selected, eligible = baseline.choose(local, baseline.POLICY, receipt["sharpness_min"])
            else:
                selected, eligible = choose_batched(local, baseline.POLICY, receipt["sharpness_min"], baseline)
            elapsed = time.perf_counter()-before
            if elapsed > 300:
                raise RuntimeError("Selection trial exceeded its registered time budget")
            if fingerprint(local, selected, eligible) != historical:
                atomic_json(args.output/(variant+"_mismatch.json"), fingerprint(local, selected, eligible))
                raise AssertionError("Full decision table or selected/eligible order differs from the baseline")
            records.append({"variant": variant, "repetition": repetition, "seconds": elapsed,
                            "all_decisions_equal": True, "selected_count": len(selected), "eligible_count": len(eligible)})
            atomic_json(args.output/"PROGRESS.json", {"status": "running", "trials": records})
            print(json.dumps(records[-1]), flush=True)
    medians = {name: statistics.median(row["seconds"] for row in records if row["variant"] == name)
               for name in ["baseline", "batched"]}
    reduction = 1-medians["batched"]/medians["baseline"]
    atomic_json(args.output/"RESULT.json", {"status": "full_sequence_selection_comparison_completed",
        "trials": records, "median_seconds": medians, "wall_reduction_fraction": reduction,
        "speed_gate_passed": reduction >= .1, "all_7338_decisions_equal": True,
        "selected_count": 1002, "eligible_count": 6797, "pipeline_integration_pending": True,
        "counted": False, "geometric_accuracy_claim": False})


if __name__ == "__main__":
    main()
