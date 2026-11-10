"""Inspect bounded problem-camera feature records without opening a writable DB."""

import argparse
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import psutil
import pycolmap

from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = json.loads((args.source_run/"RESULT.json").read_text(encoding="utf-8"))
    model = max(source["models"], key=lambda row: row["registered_count"])
    names = sorted(set(model["remaining_failures"]+[row["image_name"] for row in model["weak_registered_images_remaining"]]))
    if not 1 <= len(names) <= 5:
        raise ValueError("This diagnostic is limited to the registered five problematic cameras")
    database = (args.source_run/"database.db").resolve()
    stat = database.stat()
    atomic_json(args.output/"CONTRACT.json", {"database": str(database), "database_bytes": stat.st_size,
        "maximum_seconds": 300, "maximum_rss_bytes": 1024**3, "maximum_images": 5,
        "source_receipt": data_identity(args.source_run/"RESULT.json"), "source": data_identity(Path(__file__)),
        "database_access": "read_only_immutable", "GPU_used": False, "network_used": False})
    with sqlite3.connect(database.as_uri()+"?mode=ro&immutable=1", uri=True) as db:
        schema = {table: list(db.execute("PRAGMA table_info("+table+")")) for table in ("keypoints", "descriptors", "cameras")}
        rows = []
        for name in names:
            identity = db.execute("SELECT image_id,camera_id FROM images WHERE name=?", (name,)).fetchone()
            if identity is None:
                raise ValueError("An actual problem image is missing from the preserved database")
            image_id, camera_id = identity
            keypoints = db.execute("SELECT rows,cols,data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
            descriptors = db.execute("SELECT type,rows,cols,data FROM descriptors WHERE image_id=?", (image_id,)).fetchone()
            if keypoints is None or descriptors is None:
                raise ValueError("Feature records are incomplete")
            kr, kc, kb = keypoints
            descriptor_type, dr, dc, dd = descriptors
            if descriptor_type != pycolmap.FeatureExtractorType.SIFT.value:
                raise ValueError("The preserved feature type is not SIFT")
            if kr != dr or kc not in (4, 6) or dc != 128 or len(kb) != kr*kc*4 or len(dd) != dr*dc:
                raise ValueError("Existing keypoint/descriptor layout is not the proposed SIFT interface")
            kp = np.frombuffer(kb, dtype=np.float32).reshape(kr, kc)
            desc = np.frombuffer(dd, dtype=np.uint8).reshape(dr, dc)
            if kc == 4:
                scale, angle = kp[:, 2], kp[:, 3]
            else:
                scale = (np.hypot(kp[:, 2], kp[:, 4])+np.hypot(kp[:, 3], kp[:, 5]))/2
                angle = np.arctan2(kp[:, 4], kp[:, 2])
            norms = np.linalg.norm(desc.astype(np.float64), axis=1)
            if not np.isfinite(kp).all() or not (scale > 0).all() or not (norms > 0).all():
                raise ValueError("Nonfinite geometry or degenerate descriptors prevent a valid adapter")
            camera = db.execute("SELECT model,width,height FROM cameras WHERE camera_id=?", (camera_id,)).fetchone()
            rows.append({"name": name, "image_id": image_id, "camera_id": camera_id,
                "keypoint_rows": kr, "keypoint_cols": kc, "descriptor_rows": dr, "descriptor_cols": dc,
                "descriptor_type": descriptor_type, "descriptor_type_verified": "SIFT",
                "descriptor_dtype_inferred_from_layout": "uint8", "descriptor_L2_norm_min_median_max": np.quantile(norms, [0,.5,1]).tolist(),
                "scale_min_median_max": np.quantile(scale, [0,.5,1]).tolist(),
                "orientation_min_max_rad": [float(angle.min()), float(angle.max())],
                "camera_model_width_height": list(camera), "point2D_indices_modified": False})
    options = pycolmap.SiftExtractionOptions()
    rss = psutil.Process().memory_info().rss
    elapsed = time.perf_counter()-started
    if elapsed > 300 or rss > 1024**3:
        raise RuntimeError("The bounded read-only feature inspection exceeded its resource cap")
    after = database.stat()
    if after.st_size != stat.st_size or after.st_mtime_ns != stat.st_mtime_ns:
        raise RuntimeError("Source database changed while being inspected")
    atomic_json(args.output/"RESULT.json", {"status": "feature_layout_inspected_not_matching_verified", "schema": schema,
        "rows": rows, "pycolmap": pycolmap.__version__, "installed_default_normalization": str(options.normalization),
        "historical_normalization_requires_extraction_provenance": True,
        "database_size_and_mtime_unchanged": True, "elapsed_seconds": elapsed, "rss_bytes": rss,
        "feature_matching_executed": False, "weights_downloaded": False, "counted": False})
    print(json.dumps({"status": "feature_layout_inspected_not_matching_verified", "rows": rows,
        "installed_default_normalization": str(options.normalization), "elapsed_seconds": elapsed}))


if __name__ == "__main__":
    main()
