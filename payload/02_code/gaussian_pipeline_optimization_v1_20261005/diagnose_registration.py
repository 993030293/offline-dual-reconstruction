"""Inspect failed-camera support in the actual verified match graph."""

import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
import pycolmap

from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = json.loads((args.run / "RESULT.json").read_text())
    model = pycolmap.Reconstruction(result["models"][0]["path"])
    failures = result["models"][0]["remaining_failures"]
    raw = {row["image_name"]: row for line in (args.initial / "frame_decisions.jsonl").read_text().splitlines()
           for row in [json.loads(line)]}
    db_path = args.run / "database.db"
    reports = []
    with sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True) as db:
        images = dict(db.execute("SELECT image_id,name FROM images"))
        name_to_id = {name: iid for iid, name in images.items()}
        for name in failures:
            iid = name_to_id[name]
            descriptor_count = db.execute("SELECT rows FROM descriptors WHERE image_id=?", (iid,)).fetchone()[0]
            edges, candidates = [], {}
            rows = db.execute("SELECT pair_id,rows,cols,data,config FROM two_view_geometries "
                              "WHERE pair_id / 2147483647 = ? OR pair_id % 2147483647 = ?", (iid, iid))
            for pair_id, count, cols, blob, config in rows:
                a, b = divmod(pair_id, 2147483647)
                other = b if a == iid else a
                matches = np.frombuffer(blob or b"", dtype=np.uint32).reshape(count, cols)
                query_col, reference_col = (0, 1) if a == iid else (1, 0)
                linked = []
                registered = other in model.images and model.images[other].has_pose
                if registered:
                    ref = model.images[other]
                    for match in matches:
                        point = ref.points2D[int(match[reference_col])]
                        if point.has_point3D():
                            key, pid = int(match[query_col]), int(point.point3D_id)
                            candidates.setdefault(key, set()).add(pid)
                            linked.append(key)
                edges.append({"other": images[other], "geometry_config": config, "inliers": count,
                              "other_registered": bool(registered), "triangulated_matches": len(linked)})
            nearby = [row for row in raw.values() if abs(row["camera_header_ns"]-raw[name]["camera_header_ns"]) <= 1e9]
            reports.append({"name": name, "feature_count": descriptor_count, "quality": raw[name]["quality"],
                "coarse_pose_valid": raw[name]["coarse_pose_valid"], "verified_nonempty_pairs": sum(e["inliers"] > 0 for e in edges),
                "unique_query_features_with_3d_support": len(candidates),
                "unambiguous_query_features": sum(len(pids) == 1 for pids in candidates.values()),
                "top_edges": sorted(edges, key=lambda e: (-e["triangulated_matches"], -e["inliers"]))[:25],
                "nearby_candidates": [{"name": row["image_name"], "relative_seconds": (row["camera_header_ns"]-raw[name]["camera_header_ns"])/1e9,
                    "decision": row["decision"], "in_database": row["image_name"] in name_to_id,
                    "quality_score": row["quality"]["quality_score"]} for row in nearby]})
    output = {"status": "diagnostic_only", "reports": reports, "registered_count": model.num_reg_images(),
              "input_image_count": len(images), "source_database": data_identity(db_path), "source": data_identity(Path(__file__))}
    atomic_json(args.output / "RESULT.json", output)
    print(json.dumps([{key: row[key] for key in ("name", "feature_count", "verified_nonempty_pairs",
                     "unique_query_features_with_3d_support", "unambiguous_query_features")} for row in reports]))


if __name__ == "__main__":
    main()
