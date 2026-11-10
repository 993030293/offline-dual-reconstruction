"""Check that a real image beyond the exact histogram budget uses native scoring."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from quality_engine import compact_quality
from quality_histogram import make_quality, PIXEL_LIMIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    image = np.zeros((4097, 4097, 3), np.uint8)
    image[:2000] = 255
    old = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        reference = compact_quality(image)
        quality, trace = make_quality(compact_quality)
        value = quality(image)
    finally:
        cv2.setNumThreads(old)
    checks = {"actual_image_exceeds_histogram_integer_count_budget": image.shape[0]*image.shape[1] > PIXEL_LIMIT,
        "actual_full_native_route_before_histogram": trace["native_calls"] == 1 and trace["histogram_calls"] == 0,
        "all_original_quality_fields_exact": value == reference,
        "original_OpenCV_thread_setting_restored": cv2.getNumThreads() == old}
    result = {"status": "passed" if all(checks.values()) else "failed_verification", "checks": checks,
        "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("quality_histogram.py"))],
        "actual_pixels": image.shape[0]*image.shape[1], "reference": reference, "candidate": value,
        "trace": trace, "counted": False, "new_object_whole_pipeline_pending": True}
    atomic_json(args.output/"RESULT.json", result)
    print({key: result[key] for key in ("status", "checks")}, flush=True)
    if not all(checks.values()):
        raise ValueError("An original histogram resource admission boundary failed")


if __name__ == "__main__":
    main()
