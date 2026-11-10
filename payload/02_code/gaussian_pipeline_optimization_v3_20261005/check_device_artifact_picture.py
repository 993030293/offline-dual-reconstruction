"""Compare native display pixels over rounding edges and explicit unsupported routes."""

import argparse
import pickle
import random
from pathlib import Path
import sys
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from device_artifact_picture import DeviceArtifactPicture, native_picture, device_artifact_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if torch.__version__ != "2.7.1+cu128" or not torch.cuda.is_available():
        raise RuntimeError("Use the existing pinned CUDA runtime")
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    torch.cuda.init()
    before = (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()),
        torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
    codes = np.arange(256, dtype=np.float32)/np.float32(255)
    ties = (np.arange(-8, 264, dtype=np.float32)+np.float32(.5))/np.float32(255)
    values = np.concatenate((codes, ties, np.nextafter(ties, np.float32(-np.inf)),
        np.nextafter(ties, np.float32(np.inf)), np.asarray([-0., 0., -12., 12., 1e20, -1e20], np.float32)))
    adapter = DeviceArtifactPicture(True)
    rows = []
    for height, width in ((1, 1), (1, 11), (11, 1), (17, 67), (31, 49), (620, 812), (487, 1937)):
        raw = np.resize(values, (3, height, width))
        value = torch.from_numpy(raw.copy()).cuda()
        for view in (value, value.transpose(1, 2), value[:, ::2, ::2]):
            reference = native_picture(view)
            original = view.clone()
            candidate = adapter(view)
            row = {"shape": list(view.shape), "stride": list(view.stride()),
                "all_pixels_exact": reference.mode == candidate.mode and reference.size == candidate.size
                    and reference.tobytes() == candidate.tobytes(), "input_unchanged": torch.equal(view, original),
                "route": adapter.calls[-1]["route"]}
            rows.append(row)
    checks = {"all21_shapes_layouts_256_values_tie_neighbors_and_extremes_exact": all(row["all_pixels_exact"]
        and row["input_unchanged"] and row["route"] == "device_uint8" for row in rows)}
    for dtype in (torch.float16, torch.float64):
        value = torch.tensor([-.1, .5, 1.1], dtype=dtype, device="cuda").reshape(3, 1, 1)
        checks[str(dtype)+"_preselected_original_exact"] = native_picture(value).tobytes() == adapter(value).tobytes()
        checks[str(dtype)+"_native_route"] = adapter.calls[-1]["route"] == "native"
    cpu = torch.tensor([-.1, .5, 1.1], dtype=torch.float32).reshape(3, 1, 1)
    checks["CPU_original_route_exact"] = native_picture(cpu).tobytes() == adapter(cpu).tobytes() and adapter.calls[-1]["route"] == "native"
    nonfinite = torch.tensor([float("nan"), float("inf"), -float("inf")], device="cuda").reshape(3, 1, 1)
    with warnings.catch_warnings(record=True) as native_warnings:
        warnings.simplefilter("always")
        native = native_picture(nonfinite)
    with warnings.catch_warnings(record=True) as candidate_warnings:
        warnings.simplefilter("always")
        candidate = adapter(nonfinite)
    checks["nonfinite_preselected_original_pixels_and_warning_preserved"] = (native.tobytes() == candidate.tobytes()
        and [str(row.message) for row in native_warnings] == [str(row.message) for row in candidate_warnings]
        and adapter.calls[-1]["route"] == "native")
    limited = DeviceArtifactPicture(True, maximum_elements=3)
    value = torch.from_numpy(np.resize(values, (3, 3, 7)).copy()).cuda()
    checks["explicit_budget_preselects_original_without_pixel_drop"] = (native_picture(value).tobytes() == limited(value).tobytes()
        and limited.calls[-1]["route"] == "native")
    import evaluate_pair
    original = evaluate_pair.main
    try:
        with device_artifact_evaluation(True):
            raise RuntimeError("injected_owned_error")
    except RuntimeError as error:
        checks["global_evaluation_restored_after_error"] = str(error) == "injected_owned_error" and evaluate_pair.main is original
    checks["all_rng_states_preserved"] = (before[0] == pickle.dumps(random.getstate()) and before[1] == pickle.dumps(np.random.get_state())
        and torch.equal(before[2], torch.get_rng_state()) and torch.equal(before[3], torch.cuda.get_rng_state()))
    checks["actual_CPU_transfer_dtype_and_bytes_are_uint8"] = all(row["actual_returned_CPU_tensor_dtype"] == "torch.uint8"
        and row["device_to_host_tensor_bytes"]*4 == row["original_float_tensor_bytes"] for row in adapter.calls[:len(rows)])
    result = {"status": "passed" if all(checks.values()) else "failed_verification", "checks": checks,
        "rows": rows, "calls": adapter.calls, "torch": torch.__version__, "numpy": np.__version__, "counted": False,
        "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("device_artifact_picture.py")),
            data_identity(Path(evaluate_pair.__file__))], "new_object_whole_pipeline_pending": True}
    atomic_json(args.output/"RESULT.json", result)
    print({key: result[key] for key in ("status", "checks", "torch", "numpy")}, flush=True)
    if not all(checks.values()):
        raise AssertionError("An original final artifact pixel or routing oracle failed")


if __name__ == "__main__":
    main()
