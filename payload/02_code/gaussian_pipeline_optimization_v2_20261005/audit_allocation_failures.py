"""Summarize actual registration deficits without feeding SfM back into initial selection."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-summary", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model = json.loads(args.model_summary.read_text(encoding="utf-8"))["models"][0]
    raw = [json.loads(line) for line in (args.initial/"frame_decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    lookup = {row["image_name"]: row for row in raw}
    initial = {row["image_name"] for row in raw if row["decision"] == "selected_initial_sfm_pending"}
    eligible = {row["image_name"] for row in raw if row["decision"] in ("eligible_not_selected", "selected_initial_sfm_pending")}
    noninitial = eligible-initial
    registered = set(model["registered_names"])
    failures = set(model["remaining_failures"])
    if not registered <= initial or failures != initial-registered:
        raise ValueError("Expected the complete original1002 diagnostic denominator")
    adjacency = {name: [] for name in initial}
    with sqlite3.connect("file:"+args.database.resolve().as_posix()+"?mode=ro", uri=True) as connection:
        images = dict(connection.execute("SELECT image_id,name FROM images"))
        for pair, inliers in connection.execute("SELECT pair_id,rows FROM two_view_geometries WHERE rows>=15"):
            first, second = divmod(pair, 2147483647)
            a, b = images[first], images[second]
            adjacency[a].append((b, inliers))
            adjacency[b].append((a, inliers))
    details = []
    for name in sorted(failures, key=lambda key: lookup[key]["camera_header_ns"]):
        row = lookup[name]
        near = [candidate for candidate in raw if candidate["image_name"] in noninitial
                and abs(candidate["camera_header_ns"]-row["camera_header_ns"]) <= 800_000_000]
        details.append({"image_name": name, "camera_frame": row["camera_frame"], "camera_header_ns": row["camera_header_ns"],
                        "quality_score": row["quality"]["quality_score"], "sharpness": row["quality"]["sharpness"],
                        "verified_neighbors": len(adjacency[name]),
                        "verified_registered_neighbors": sum(other in registered for other, _ in adjacency[name]),
                        "best_pair_inliers": max((count for _, count in adjacency[name]), default=0),
                        "eligible_noninitial_within0_8seconds": len(near),
                        "candidate_names": [candidate["image_name"] for candidate in near]})
    clusters = []
    for item in details:
        if not clusters or item["camera_header_ns"]-clusters[-1][-1]["camera_header_ns"] > 800_000_000:
            clusters.append([])
        clusters[-1].append(item)
    summaries = []
    for group in clusters:
        available = set().union(*(set(row["candidate_names"]) for row in group))
        summaries.append({"first_frame": group[0]["camera_frame"], "last_frame": group[-1]["camera_frame"],
                          "failed_count": len(group), "span_seconds": (group[-1]["camera_header_ns"]-group[0]["camera_header_ns"])/1e9,
                          "eligible_noninitial_near_failures": len(available),
                          "quality_score_median": float(np.median([row["quality_score"] for row in group])),
                          "verified_registered_neighbor_median": float(np.median([row["verified_registered_neighbors"] for row in group])),
                          "failures_without_verified_registered_neighbor": sum(row["verified_registered_neighbors"] == 0 for row in group),
                          "all_candidate_names": sorted(available)})
    result = {"status": "actual_registration_deficit_diagnostic", "model_summary": data_identity(args.model_summary),
              "registered": len(registered), "required": len(initial), "failure_count": len(failures),
              "clusters": summaries, "details": details,
              "weak_registered": model["geometry_support_fail_rows"],
              "all_registered_quality_score_median": float(np.median([lookup[name]["quality"]["quality_score"] for name in registered])),
              "phase": "second_stage_only; initial allocation remains frozen and raw_only",
              "geometry_accuracy_claim": False, "counted": False, "elapsed_seconds": time.monotonic()-started}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "registered", "required", "failure_count", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
