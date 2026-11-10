"""Exercise complete export comparisons and the original assertion-only API."""

import argparse
import json
from pathlib import Path

from profile_sparse_ply_oracle import compare_outputs
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing-campaign", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks = {}
    cases = ("identical", "observed_top_level", "output_prefix", "type", "key_order",
             "signed_zero", "float_value", "sequence", "nested_elapsed", "binary",
             "trailing_binary", "file_membership")
    for name in cases:
        left, right = args.output/f"{name}_left", args.output/f"{name}_right"
        left.mkdir()
        right.mkdir()
        value = {"count": 7, "zero": 0., "rows": [1, 2, 3],
                 "nested": {"elapsed_seconds": .5}}
        first, second = dict(value), dict(value)
        if name == "observed_top_level":
            first.update(elapsed_seconds=1., peak_RSS_bytes=123)
            second.update(elapsed_seconds=2., peak_RSS_bytes=456)
        elif name == "output_prefix":
            first["dataset"] = str(left.resolve()/"dataset")
            second["dataset"] = str(right.resolve()/"dataset")
        elif name == "type":
            second["count"] = 7.
        elif name == "key_order":
            second = dict(reversed(tuple(second.items())))
        elif name == "signed_zero":
            second["zero"] = -0.
        elif name == "float_value":
            second["zero"] = 1e-100
        elif name == "sequence":
            second["rows"] = [3, 2, 1]
        elif name == "nested_elapsed":
            second["nested"] = {"elapsed_seconds": .6}
        atomic_json(left/"MODEL.json", first)
        atomic_json(right/"MODEL.json", second)
        (left/"points.bin").write_bytes(bytes(range(64)))
        (right/"points.bin").write_bytes(bytes(range(64)))
        if name == "binary":
            (right/"points.bin").write_bytes(bytes(range(63))+b"x")
        elif name == "trailing_binary":
            (right/"points.bin").write_bytes(bytes(range(64))+b"x")
        elif name == "file_membership":
            (right/"extra.bin").write_bytes(b"extra")
        accepted = False
        try:
            verification = compare_outputs(left.resolve(), right.resolve())
            accepted = verification["complete_byte_pairs"] == 1 and verification["complete_JSON_pairs"] == 1
        except AssertionError:
            pass
        checks[name] = accepted == (name in ("identical", "observed_top_level", "output_prefix"))
    existing = None
    if args.existing_campaign is not None:
        existing = compare_outputs((args.existing_campaign/"pair0_native_").resolve(),
                                   (args.existing_campaign/"pair0_columns").resolve())
        checks["existing_actual_export_pair_complete_contents_exact"] = True
    sources = [Path(__file__), Path(__file__).with_name("profile_sparse_ply_oracle.py"),
               Path(__file__).with_name("machine_json.py")]
    result = {"status": "all_actual_export_comparison_guards_passed" if all(checks.values()) else "failed",
              "checks": checks, "checked": len(checks), "existing_export_verification": existing,
              "existing_failed_campaign_becomes_formal_performance_evidence": False,
              "counted": False, "source": [data_identity(path) for path in sources]}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "checked": len(checks)}))
    if not all(checks.values()):
        raise AssertionError("Complete output or original assertion-only API protection changed")


if __name__ == "__main__":
    main()
