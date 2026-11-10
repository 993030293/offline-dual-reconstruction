"""Measure preserved weak-camera support and triangulation without changing a map."""

import argparse
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import psutil
import pycolmap

from runtime import atomic_json, data_identity


def quantiles(values):
    return np.quantile(values, [0, .5, .95, 1]).tolist() if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.run/"RESULT.json").read_text())
    best = max(source["models"], key=lambda row: row["registered_count"])
    names = sorted(r["image_name"] for r in best["weak_cameras"])
    if not 1 <= len(names) <= 5:
        raise ValueError("This diagnostic requires one to five measured weak cameras")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model_path = Path(best["path"])
    identities = [data_identity(model_path/name) for name in ("cameras.bin", "images.bin", "points3D.bin")]
    atomic_json(args.output/"CONTRACT.json", {"source": data_identity(Path(__file__)), "models": identities,
        "weak_images": names, "maximum_seconds": 300, "maximum_RSS_bytes": 4*1024**3,
        "read_only": True, "GPU_used": False, "GT_used": False, "scope": "diagnose support; no threshold tuning or image removal"})
    model = pycolmap.Reconstruction(model_path)
    images = {im.name: im for im in model.images.values() if im.has_pose}
    centers = {im.image_id: im.projection_center() for im in images.values()}
    rows = []
    database = (args.run/"database.db").resolve()
    with sqlite3.connect(database.as_uri()+"?mode=ro&immutable=1", uri=True) as db:
        for name in names:
            image = images[name]
            lengths, angles, errors = [], [], []
            own_center = centers[image.image_id]
            for point2d in image.points2D:
                if not point2d.has_point3D():
                    continue
                point = model.points3D[point2d.point3D_id]
                lengths.append(point.track.length())
                errors.append(point.error)
                own_ray = point.xyz-own_center
                ray_angles = []
                for item in point.track.elements:
                    if item.image_id == image.image_id:
                        continue
                    ray = point.xyz-centers[item.image_id]
                    cosine = np.dot(own_ray, ray)/(np.linalg.norm(own_ray)*np.linalg.norm(ray))
                    angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
                    ray_angles.append(min(angle, 180-angle))
                angles.append(max(ray_angles, default=0))
            verified_rows = []
            associated_query_indices = set()
            for pair_id, count, cols, blob in db.execute(
                    "SELECT pair_id,rows,cols,data FROM two_view_geometries WHERE rows>0 AND "
                    "(pair_id / 2147483647 = ? OR pair_id % 2147483647 = ?)",
                    (image.image_id, image.image_id)):
                a, b = divmod(pair_id, 2147483647)
                if image.image_id not in (a, b):
                    continue
                other_id = b if image.image_id == a else a
                if other_id not in model.images or not model.images[other_id].has_pose:
                    continue
                indices = np.frombuffer(blob, np.uint32).reshape(count, cols)
                if image.image_id != a:
                    indices = indices[:, ::-1]
                other = model.images[other_id]
                mapped = [(int(q), int(j)) for q,j in indices if other.points2D[int(j)].has_point3D()]
                associated_query_indices.update(q for q,j in mapped)
                verified_rows.append({"other": other.name, "verified_matches": count, "matching_existing_3D": len(mapped)})
            rows.append({"image": name, "registered_feature_count": len(image.points2D),
                "triangulated_point_count": len(lengths), "point_track_length_min_p50_p95_max": quantiles(lengths),
                "max_supported_triangulation_angle_deg_min_p50_p95_max": quantiles(angles),
                "point_mean_reprojection_error_px_min_p50_p95_max": quantiles(errors),
                "unique_query_features_linked_to_existing_3D_by_verified_pairs": len(associated_query_indices),
                "verified_pairs": sorted(verified_rows, key=lambda row: (-row["matching_existing_3D"], row["other"]))})
    elapsed = time.monotonic()-started
    rss = psutil.Process().memory_info().rss
    if elapsed > 300 or rss > 4*1024**3:
        raise RuntimeError("Read-only geometry diagnostic exceeded its registered budget")
    unchanged = identities == [data_identity(model_path/name) for name in ("cameras.bin", "images.bin", "points3D.bin")]
    if not unchanged:
        raise ValueError("Source map changed during the read-only diagnostic")
    atomic_json(args.output/"RESULT.json", {"status": "weak_geometry_measured", "rows": rows,
        "elapsed_seconds": elapsed, "rss_bytes": rss, "source_unchanged": unchanged,
        "geometry_accuracy_claim": False, "counted": False})
    print(json.dumps({"status": "weak_geometry_measured", "rows": [{k:v for k,v in row.items() if k != "verified_pairs"} for row in rows], "elapsed_seconds": elapsed}))


if __name__ == "__main__":
    main()
