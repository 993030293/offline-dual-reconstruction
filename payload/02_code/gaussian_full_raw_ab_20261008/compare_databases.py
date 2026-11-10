"""Compare complete native feature and matching columns without comparing SQLite layout."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

V1 = Path(__file__).resolve().parents[2] / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def compare(first, second):
    with sqlite3.connect(first.resolve().as_uri()+"?mode=ro", uri=True) as left, sqlite3.connect(second.resolve().as_uri()+"?mode=ro", uri=True) as right:
        left_names = dict(left.execute("SELECT image_id,name FROM images"))
        right_names = dict(right.execute("SELECT image_id,name FROM images"))
        names_equal = set(left_names.values()) == set(right_names.values())
        identities_equal = left_names == right_names
        tables = {}
        for table in ("cameras", "images", "keypoints", "descriptors", "matches", "two_view_geometries"):
            columns_left = left.execute("PRAGMA table_info("+table+")").fetchall()
            columns_right = right.execute("PRAGMA table_info("+table+")").fetchall()
            if columns_left != columns_right:
                tables[table] = {"schema_equal": False, "rows_exact": False}
                continue
            key = columns_left[0][1]
            rows_left = left.execute("SELECT * FROM "+table+" ORDER BY "+key)
            rows_right = right.execute("SELECT * FROM "+table+" ORDER BY "+key)
            count_left, count_right, mismatches, examples = 0, 0, 0, []
            while True:
                a, b = rows_left.fetchone(), rows_right.fetchone()
                if a is None and b is None:
                    break
                count_left += a is not None
                count_right += b is not None
                if a != b:
                    mismatches += 1
                    if len(examples) < 12:
                        examples.append({"left_key": None if a is None else a[0],
                            "right_key": None if b is None else b[0],
                            "different_columns": [columns_left[index][1] for index in range(len(columns_left))
                                if a is None or b is None or a[index] != b[index]]})
            tables[table] = {"schema_equal": True, "left_count": count_left, "right_count": count_right,
                "rows_exact": mismatches == 0, "mismatched_rows": mismatches, "first_mismatch_columns": examples}
        return {"image_name_set_equal": names_equal, "native_image_ids_equal": identities_equal, "tables": tables,
                "all_native_columns_exact": identities_equal and all(row["rows_exact"] for row in tables.values()),
                "independent_fresh_database_comparison": True,
                "different_ids_or_features_require_same_feature_input_module_ablation": not identities_equal or
                    not tables["keypoints"]["rows_exact"] or not tables["descriptors"]["rows_exact"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("The complete database comparison requires a new receipt")
    started = time.monotonic()
    result = compare(args.baseline, args.optimized)
    result.update(status="exact_complete_database_columns" if result["all_native_columns_exact"] else "fresh_database_differences_require_diagnosis",
                  elapsed_seconds=time.monotonic()-started, source=data_identity(Path(__file__)))
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "elapsed_seconds", "native_image_ids_equal")}), flush=True)


if __name__ == "__main__":
    main()
