"""Build one immutable image-quality table from verified original keyed score records."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from quality_engine import FIELDS
from columnar_quality_scores import build_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    args = parser.parse_args()
    result = build_table(args.source, args.output, FIELDS, args.expected_records)
    result["sources"] = [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("columnar_quality_scores.py")),
        data_identity(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005/quality_engine.py")]
    atomic_json(args.output/"BUILD_RECEIPT.json", result)
    print({key: result[key] for key in ("status", "records", "build_seconds", "artifact_bytes", "original_score_files_bytes")}, flush=True)


if __name__ == "__main__":
    main()
