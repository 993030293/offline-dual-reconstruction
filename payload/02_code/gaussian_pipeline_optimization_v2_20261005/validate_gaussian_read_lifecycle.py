"""Check exact Gaussian readback and fail-closed malformed native PLY inputs."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import activate_reference, atomic_json, data_identity
from vectorized_gaussian_read import vectorized_gaussian_read, native_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "campaign", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    activate_reference(args.runtime)
    import torch
    from plyfile import PlyData, PlyElement, PlyParseError
    from scene.gaussian_model import GaussianModel
    source = Path(__file__).with_name("vectorized_gaussian_read.py")
    campaign = json.loads((args.campaign/"RESULT.json").read_text())
    identity = data_identity(source)
    if identity not in campaign["sources"]:
        raise ValueError("The current reader differs from the actual three-pair campaign")
    original = GaussianModel.load_ply
    names = native_names(3)
    values = np.zeros(5, dtype=[(name, "<f4") for name in names])
    for index, name in enumerate(names):
        values[name] = np.arange(5, dtype=np.float32)*.01+index*.001
    fixtures = {}

    def save(name, rows, byte_order="<", text=False):
        path = args.output/(name+".ply")
        PlyData([PlyElement.describe(rows, "vertex")], text=text, byte_order=byte_order).write(path)
        fixtures[name] = path
        return path

    valid = save("native", values)
    native, fast = GaussianModel(3), GaussianModel(3)
    native.load_ply(str(valid))
    with vectorized_gaussian_read() as trace:
        fast.load_ply(str(valid))
    fields = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
    checks = {"small_complete_native_parameter_oracle": all(torch.equal(getattr(native, key), getattr(fast, key))
        and getattr(native, key).stride() == getattr(fast, key).stride() for key in fields),
        "context_restores_original_after_success": GaussianModel.load_ply is original}
    missing = np.zeros(5, dtype=[(name, "<f4") for name in names[:-1]])
    save("missing", missing)
    nonfinite = values.copy()
    nonfinite["opacity"][0] = np.nan
    save("nonfinite", nonfinite)
    save("empty", values[:0])
    save("bigendian", values, ">")
    save("float64", np.zeros(5, dtype=[(name, "<f8") for name in names]))
    truncated = args.output/"truncated.ply"
    truncated.write_bytes(valid.read_bytes()[:-7])
    fixtures["truncated"] = truncated
    for name in ("missing", "nonfinite", "empty", "bigendian", "float64", "truncated"):
        rejected = False
        try:
            with vectorized_gaussian_read():
                GaussianModel(3).load_ply(str(fixtures[name]))
        except (ValueError, FloatingPointError, EOFError, PlyParseError):
            rejected = True
        checks[name+"_rejected"] = rejected
        checks[name+"_restored_original"] = GaussianModel.load_ply is original
    for name, degree, exposure in (("wrong_degree", 2, False), ("exposure", 3, True)):
        try:
            with vectorized_gaussian_read():
                GaussianModel(degree).load_ply(str(valid), exposure)
            checks[name+"_rejected"] = False
        except ValueError:
            checks[name+"_rejected"] = True
    checks.update(three_actual_parameter_pairs_exact=len(campaign["pairs"]) == 3 and
        all(row["all_native_parameters_exact"] for row in campaign["pairs"]),
        all24_actual_render_metrics_exact=campaign["all24_pixels_metrics_preserved"],
        real_model_load_reduction_at_least10pct=campaign["load_reduction"] >= .1,
        original_model_preserved=campaign["original_model_identity_unchanged"],
        current_reader_source_matches_campaign=data_identity(source) == identity)
    accepted = all(checks.values())
    result = {"status": "stage_accepted_combination_pending" if accepted else "failed_verification",
        "mechanism": "Q61_EXACT_NATIVE_GAUSSIAN_PLY_READER", "stage_counted": accepted,
        "checks": checks, "campaign": str(args.campaign), "median_load_seconds": campaign["median_load_seconds"],
        "load_reduction": campaign["load_reduction"], "sources": [data_identity(Path(__file__)), identity],
        "fixture_native_trace": trace, "no_training_or_object_quality_claim": True, "combination_pending": True}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    print({key: result[key] for key in ("status", "stage_counted", "median_load_seconds", "load_reduction")}, flush=True)
    if not accepted:
        raise AssertionError("A Gaussian reader lifecycle or actual-output guard failed")


if __name__ == "__main__":
    main()
