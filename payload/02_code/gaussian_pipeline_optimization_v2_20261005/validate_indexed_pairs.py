"""Check immutable spatial-pair identity, ties, rejected input, and restoration."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
import run_initial_pipeline
from indexed_candidate_pairs import candidate_pairs, indexed_pair_generation


def rows(positions):
    return [{"image_name": f"fixture_{i:05d}.jpg", "C_W_m": p.tolist()} for i, p in enumerate(positions)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    actual = manifest["selected_frames"]
    generator = np.random.default_rng(20261005)
    sphere = generator.normal(size=(40, 3))
    sphere /= np.linalg.norm(sphere, axis=1)[:, None]
    cases = {"actual1002": actual, "small2": rows(generator.normal(size=(2, 3))),
        "small17": rows(generator.normal(size=(17, 3))), "boundary18": rows(generator.normal(size=(18, 3))),
        "duplicate_positions40": rows(np.zeros((40, 3))),
        "spherical_ties40": rows(np.vstack([np.zeros((1, 3)), sphere])),
        "random80": rows(generator.normal(size=(80, 3)))}
    checks, details = {}, {}
    for name, data in cases.items():
        reference, _ = candidate_pairs(data, False)
        indexed, receipt = candidate_pairs(data, True)
        checks[name+"_pair_set_exact"] = indexed == reference
        details[name] = receipt
    invalid = {"NaN": rows(np.array([[0., 0, 0], [np.nan, 0, 0]])),
        "overflow": rows(np.array([[0., 0, 0], [1e160, 0, 0]])),
        "one_frame": rows(np.zeros((1, 3))),
        "duplicate_identity": [actual[0], actual[0]]}
    for name, data in invalid.items():
        try:
            candidate_pairs(data)
            checks[name+"_rejected"] = False
        except ValueError:
            checks[name+"_rejected"] = True
    original = run_initial_pipeline.main
    try:
        with indexed_pair_generation(run_initial_pipeline):
            raise RuntimeError("Intentional adapter restoration check")
    except RuntimeError:
        pass
    checks["entry_restored_after_error"] = run_initial_pipeline.main is original
    sources = [Path(__file__), Path(__file__).with_name("indexed_candidate_pairs.py"), Path(run_initial_pipeline.__file__)]
    result = {"status": "lifecycle_passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "details": details, "source": data_identity(args.manifest),
        "sources": [data_identity(p) for p in sources], "counted": False,
        "not_a_speed_or_quality_result": True}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps(result), flush=True)
    if not all(checks.values()):
        raise ValueError("Exact candidate graph lifecycle did not pass")


if __name__ == "__main__":
    main()
