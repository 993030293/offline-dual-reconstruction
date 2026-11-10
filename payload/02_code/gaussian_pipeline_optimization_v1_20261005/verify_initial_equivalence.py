"""Compare rebuilt raw-only inputs against every frozen frame and exported image."""

import argparse
import json
from pathlib import Path
import numpy as np

from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    old = [json.loads(line) for line in (args.baseline/"frame_decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    new = [json.loads(line) for line in (args.candidate/"frame_decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(old) != len(new):
        raise AssertionError("Raw input denominator changed")
    differences = {"quality": 0., "pose": 0.}
    exact_fields = ["camera_frame", "camera_header_ns", "image_name", "width", "height", "optical_frame",
                    "excluded_role", "coarse_pose_valid", "decision", "selection_reason", "pose_rejection",
                    "left_pose_index", "right_pose_index", "bracket_span_ns"]
    for a, b in zip(old, new):
        for key in exact_fields:
            if a.get(key) != b.get(key):
                raise AssertionError(f"Frame {a['camera_frame']} field {key} changed")
        for key in a["quality"]:
            x, y = a["quality"][key], b["quality"][key]
            differences["quality"] = max(differences["quality"], abs(x-y))
            if not np.isclose(x, y, rtol=1e-12, atol=1e-10):
                raise AssertionError(f"Frame {a['camera_frame']} quality {key} changed")
        if a["coarse_pose_valid"]:
            for key in ["C_W_m", "R_WC", "quaternion_WC_xyzw", "interpolation_alpha"]:
                x, y = np.asarray(a[key]), np.asarray(b[key])
                differences["pose"] = max(differences["pose"], float(np.abs(x-y).max()))
                if not np.allclose(x, y, rtol=0, atol=1e-10):
                    raise AssertionError(f"Frame {a['camera_frame']} pose {key} changed")
    old_manifest = json.loads((args.baseline/"INITIAL_FRAME_MANIFEST.json").read_text(encoding="utf-8"))
    new_manifest = json.loads((args.candidate/"INITIAL_FRAME_MANIFEST.json").read_text(encoding="utf-8"))
    if [row["camera_frame"] for row in old_manifest["selected_frames"]] != [row["camera_frame"] for row in new_manifest["selected_frames"]]:
        raise AssertionError("Full exported frame set changed")
    image_count = 0
    for a, b in zip(old_manifest["selected_frames"], new_manifest["selected_frames"]):
        actual = data_identity(args.candidate/"images"/b["image_name"])["sha256"]
        if a["image_sha256"] != b["image_sha256"] or actual != a["image_sha256"]:
            raise AssertionError("Exported JPEG payload changed")
        image_count += 1
    if (args.baseline/"initial_matching_pairs.txt").read_text() != (args.candidate/"initial_matching_pairs.txt").read_text():
        raise AssertionError("Proposed matching pairs changed")
    receipt = {"status": "passed_full_sequence_equivalence", "raw_frame_count": len(old),
        "image_count": image_count, "all_decisions_and_selection_reasons_equal": True,
        "all_exported_JPEG_payloads_equal": True, "all_candidate_pairs_equal": True,
        "maximum_absolute_differences": differences, "SfM_geometry_verified": False,
        "baseline": str(args.baseline), "candidate": str(args.candidate)}
    atomic_json(args.candidate/"EQUIVALENCE.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
