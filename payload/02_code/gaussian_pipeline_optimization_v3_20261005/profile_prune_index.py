"""Audit complete native densification and pruning on a saved full state."""

import argparse
import copy
import gc
import json
from pathlib import Path
import random
import resource
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import activate_reference, atomic_json, data_identity

NAMES = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
AUX = ("xyz_gradient_accum", "denom", "max_radii2D", "tmp_radii")


def exact(left, right, torch):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.shape == right.shape
                and left.dtype == right.dtype and left.device == right.device
                and left.stride() == right.stride() and torch.equal(left, right))
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(exact(left[k], right[k], torch) for k in left))
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            exact(a, b, torch) for a, b in zip(left, right))
    return left == right


def models_exact(a, b, torch):
    return (exact(a.capture(), b.capture(), torch)
            and all(exact(getattr(a, n), getattr(b, n), torch) for n in AUX)
            and exact(a.exposure_optimizer.state_dict(), b.exposure_optimizer.state_dict(), torch)
            and exact(a._exposure, b._exposure, torch)
            and all(a.optimizer.param_groups[i]["params"][0] is getattr(a, n)
                    and b.optimizer.param_groups[i]["params"][0] is getattr(b, n)
                    for i, n in enumerate(NAMES)))


def comparison_differences(a, b, torch):
    differences = []

    def walk(left, right, path):
        if isinstance(left, torch.Tensor):
            if not exact(left, right, torch):
                differences.append({"path": path, "left_shape": list(left.shape),
                    "right_shape": list(right.shape), "left_stride": list(left.stride()),
                    "right_stride": list(right.stride()), "left_dtype": str(left.dtype),
                    "right_dtype": str(right.dtype), "left_device": str(left.device),
                    "right_device": str(right.device), "values_equal": torch.equal(left, right)})
        elif isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
            for key in left:
                walk(left[key], right[key], path + "/" + str(key))
        elif isinstance(left, (list, tuple)) and type(left) is type(right) and len(left) == len(right):
            for index, (x, y) in enumerate(zip(left, right)):
                walk(x, y, path + "/" + str(index))
        elif left != right:
            differences.append({"path": path, "left_type": type(left).__name__, "right_type": type(right).__name__})

    walk(a.capture(), b.capture(), "capture")
    for name in AUX:
        walk(getattr(a, name), getattr(b, name), name)
    walk(a.exposure_optimizer.state_dict(), b.exposure_optimizer.state_dict(), "exposure_optimizer")
    walk(a._exposure, b._exposure, "exposure")
    for model, label in ((a, "native"), (b, "candidate")):
        for i, name in enumerate(NAMES):
            if model.optimizer.param_groups[i]["params"][0] is not getattr(model, name):
                differences.append({"path": label + "/parameter_owner/" + name})
    return differences


def fixture(base, torch, options, count, has_state, strided=False, device="cuda"):
    model = base(3)
    shapes = ((3,), (1, 3), (15, 3), (1,), (3,), (4,))
    for name, tail in zip(NAMES, shapes):
        shape = (count, *tail)
        value = torch.arange(int(np.prod(shape)), dtype=torch.float32, device=device).reshape(shape) * .0001
        if strided:
            backing = torch.zeros((*shape[:-1], shape[-1] * 2), device=device)
            backing[..., ::2] = value
            value = backing[..., ::2]
        setattr(model, name, torch.nn.Parameter(value))
    model._exposure = torch.nn.Parameter(torch.eye(3, 4, device="cuda")[None])
    model.exposure_mapping = {"owned_fixture": 0}
    model.pretrained_exposures = None
    model.spatial_lr_scale = 1.
    model.training_setup(options)
    for name in AUX:
        tail = (1,) if name in ("xyz_gradient_accum", "denom") else ()
        setattr(model, name, torch.arange(count, dtype=torch.float32, device=device).reshape(count, *tail))
    if has_state:
        for group in model.optimizer.param_groups:
            parameter = group["params"][0]
            model.optimizer.state[parameter] = {"step": torch.tensor(2.),
                "exp_avg": torch.full_like(parameter, .002), "exp_avg_sq": torch.full_like(parameter, .003)}
    return model


def generic(base, torch, options, adapter):
    rows = []
    for count in (0, 1, 2, 7, 31, 127, 257):
        for has_state in (False, True):
            for kind in ("keep_all", "remove_all", "alternating", "first_only"):
                a = fixture(base, torch, options, count, has_state, strided=count % 2 == 1)
                b = fixture(base, torch, options, count, has_state, strided=count % 2 == 1)
                raw = torch.arange(count, device="cuda")
                masks = {"keep_all": raw < 0, "remove_all": raw >= 0,
                         "alternating": raw % 2 == 0, "first_only": raw == 0}
                mask = masks[kind]
                backing = torch.zeros(count * 2, dtype=torch.bool, device="cuda")
                backing[::2] = mask
                mask = backing[::2]
                rng = torch.cuda.get_rng_state().clone()
                a.prune_points(mask)
                with adapter(base) as trace:
                    b.prune_points(mask)
                checks = {"all_parameters_optimizer_aux_and_layout_exact": models_exact(a, b, torch),
                    "one_complete_owned_index": trace["indices_created"] == 1,
                    "empty_selection_uses_original_layout": trace["native_empty_route_calls"] == int((~mask).sum().item() == 0),
                    "native_binding_restored": trace["original_method_restored"],
                    "RNG_unchanged": torch.equal(rng, torch.cuda.get_rng_state())}
                row = {"count": count, "has_state": has_state, "mask": kind, "strided": count % 2 == 1,
                       "checks": checks}
                rows.append(row)
                if not all(checks.values()):
                    row["differences"] = comparison_differences(a, b, torch)
                    raise AssertionError("An original generic prune contract changed: " + str(row))
                del a, b
    a = fixture(base, torch, options, 7, True, device="cpu")
    b = fixture(base, torch, options, 7, True, device="cpu")
    with adapter(base) as trace:
        b.prune_points(torch.tensor([False, True, False, True, False, True, False]))
    a.prune_points(torch.tensor([False, True, False, True, False, True, False]))
    rows.append({"name": "CPU_selects_original_before_state_changes", "checks": {
        "exact": models_exact(a, b, torch), "native_route": trace["native_route_calls"] == 1,
        "restored": trace["original_method_restored"]}})
    for defect in ("wrong_mask_length", "missing_auxiliary"):
        a = fixture(base, torch, options, 7, True)
        b = fixture(base, torch, options, 7, True)
        mask = torch.zeros(6 if defect == "wrong_mask_length" else 7, dtype=torch.bool, device="cuda")
        if defect == "missing_auxiliary":
            del a.tmp_radii, b.tmp_radii
        errors = []
        for model, enabled in ((a, False), (b, True)):
            try:
                with adapter(base, enabled=enabled):
                    model.prune_points(mask)
            except Exception as error:
                errors.append((type(error).__name__, str(error)))
        partial = exact(a.capture(), b.capture(), torch)
        checks = {"same_native_error_and_partial_state": len(errors) == 2 and errors[0] == errors[1] and partial,
                  "binding_restored": base.prune_points.__name__ == "prune_points"}
        rows.append({"name": defect, "checks": checks, "errors": errors})
        if not all(checks.values()):
            raise AssertionError("An invalid input or binding restoration changed")
    gc.collect()
    torch.cuda.empty_cache()
    if not all(all(row["checks"].values()) for row in rows):
        raise AssertionError("A generic original-route check failed")
    return rows


def restore_saved(base, state, options, torch):
    model = base(3)
    model._exposure = torch.nn.Parameter(state["exposure"].detach().to("cuda").clone())
    model.exposure_mapping = copy.deepcopy(state["exposure_mapping"])
    model.pretrained_exposures = copy.deepcopy(state["pretrained_exposures"])
    capture = list(copy.deepcopy(state["gaussian_capture"]))
    for i in range(1, 10):
        value = capture[i].detach().to("cuda").clone()
        capture[i] = torch.nn.Parameter(value) if isinstance(capture[i], torch.nn.Parameter) else value
    model.restore(tuple(capture), options)
    model.exposure_optimizer.load_state_dict(copy.deepcopy(state["exposure_optimizer"]))
    model.tmp_radii = model.max_radii2D.clone()
    return model


def restore_rng(state, torch):
    random.setstate(state["rng_python"])
    np.random.set_state(state["rng_numpy"])
    torch.set_rng_state(state["rng_torch_cpu"])
    torch.cuda.set_rng_state_all(state["rng_torch_cuda"])


def rng_snapshot(torch):
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "checkpoint", "config", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = activate_reference(args.runtime)
    import torch
    from scene.gaussian_model import GaussianModel
    from shared_prune_index import shared_prune_index
    torch.set_num_threads(4)
    paths = [Path(__file__), Path(__file__).with_name("shared_prune_index.py"),
             Path(GaussianModel.prune_points.__code__.co_filename), args.checkpoint, args.checkpoint.with_suffix(".json"), args.config]
    identities = [data_identity(p) for p in paths]
    options = SimpleNamespace(**json.loads(args.config.read_text())["parameters"])
    receipt = json.loads(args.checkpoint.with_suffix(".json").read_text())
    if receipt.get("status") != "complete" or receipt["checkpoint"]["sha256"] != identities[3]["sha256"]:
        raise ValueError("The original complete checkpoint identity changed")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state["schema_version"] != 1 or state["identity"] != receipt["identity"]:
        raise ValueError("The original complete schema or input/schedule identity changed")
    protocol = {"candidate": "R108_SINGLE_PRUNE_GATHER_INDEX", "source_inputs": identities,
        "reference": reference, "torch": torch.__version__, "input_version": state["identity"].get("input_version"),
        "camera_count": len(state["camera_order"]), "saved_iteration": state["iteration"],
        "three_pairs_of_native_complete_densify_and_prune": True, "all_saved_parameters_optimizer_aux_RNG_exact": True,
        "radii_fixture": "owned_clone_of_saved_max_radii2D_not_original_last_render_radii",
        "extent_fixture": "native_saved_spatial_lr_scale", "original_opacity_threshold": .005,
        "screen_size_gate": None if state["iteration"] <= options.opacity_reset_interval else 20,
        "minimum_complete_stage_reduction_including_shared_verification": .1,
        "maximum_each_whole_ratio": 1.1, "CUDA_budget_bytes": 12 * 1024**3,
        "WSL_RSS_budget_bytes": 32 * 1024**3, "requires_three_seed_full12000_quality_and_recovery_after_diagnostic": True,
        "diagnostic_is_not_Scene_training_or_render_gradient_validation": True,
        "empty_retained_indices_select_original_before_state_changes": True,
        "GT_geometry_or_frame_reduction_used": False, "counted": False}
    atomic_json(args.output / "PROTOCOL.json", protocol)
    try:
        generic_rows = generic(GaussianModel, torch, options, shared_prune_index)
        atomic_json(args.output / "GENERIC.json", {"checks": generic_rows, "count": len(generic_rows), "passed": True})
        pairs = []
        for trial in range(3):
            row = {"trial": trial, "modes": {}, "counted": False}
            models, rngs = {}, {}
            for mode in (("native", "indexed") if trial % 2 == 0 else ("indexed", "native")):
                whole_started = time.perf_counter()
                before_load = time.perf_counter()
                source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
                model = restore_saved(GaussianModel, source, options, torch)
                restore_rng(source, torch)
                load_seconds = time.perf_counter() - before_load
                free, _ = torch.cuda.mem_get_info()
                if free < 2 * 1024**3:
                    raise MemoryError("The admitted complete diagnostic lost its CUDA headroom")
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with shared_prune_index(GaussianModel, mode == "indexed") as trace:
                    model.densify_and_prune(options.densify_grad_threshold, .005, model.spatial_lr_scale,
                        protocol["screen_size_gate"], model.tmp_radii)
                    torch.cuda.synchronize()
                stage_seconds = time.perf_counter() - started
                rngs[mode] = rng_snapshot(torch)
                models[mode] = model
                row["modes"][mode] = {"complete_stage_seconds": stage_seconds, "load_restore_seconds": load_seconds,
                    "entry_without_pair_validation_seconds": time.perf_counter() - whole_started,
                    "trace": trace, "points_after": len(model.get_xyz),
                    "CUDA_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "CUDA_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                if row["modes"][mode]["CUDA_peak_reserved_bytes"] > protocol["CUDA_budget_bytes"]:
                    raise MemoryError("The complete diagnostic exceeded the CUDA budget")
                del source
            before_validation = time.perf_counter()
            checks = {"complete_parameters_optimizer_aux_layout_and_owner_exact": models_exact(models["native"], models["indexed"], torch),
                      "all_saved_random_generators_exact": exact(rngs["native"], rngs["indexed"], torch),
                      "method_restored_both_sides": all(x["trace"]["original_method_restored"] for x in row["modes"].values()),
                      "indexed_route_exercised": row["modes"]["indexed"]["trace"]["indices_created"] > 0}
            validation_seconds = time.perf_counter() - before_validation
            row["checks"] = checks
            row["pair_validation_seconds"] = validation_seconds
            for record in row["modes"].values():
                record["complete_stage_with_shared_validation_seconds"] = record["complete_stage_seconds"] + validation_seconds / 2
                record["whole_seconds"] = record["entry_without_pair_validation_seconds"] + validation_seconds / 2
            row["whole_ratio"] = row["modes"]["indexed"]["whole_seconds"] / row["modes"]["native"]["whole_seconds"]
            pairs.append(row)
            atomic_json(args.output / "PROGRESS.json", {"status": "running", "pairs": pairs, "counted": False})
            print(json.dumps({"trial": trial, "checks": checks, "whole_ratio": row["whole_ratio"],
                  "stage_seconds": {m: r["complete_stage_with_shared_validation_seconds"] for m, r in row["modes"].items()}}), flush=True)
            if not all(checks.values()) or row["whole_ratio"] > 1.1:
                raise AssertionError("A complete original diagnostic contract or whole-cost guard failed")
            del model, models, rngs
            gc.collect()
            torch.cuda.empty_cache()
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > protocol["WSL_RSS_budget_bytes"]:
                raise MemoryError("The diagnostic exceeded its independently measured WSL RSS budget")
        medians = {mode: {field: statistics.median(p["modes"][mode][field] for p in pairs)
            for field in ("complete_stage_seconds", "complete_stage_with_shared_validation_seconds", "whole_seconds")}
            for mode in ("native", "indexed")}
        field = "complete_stage_with_shared_validation_seconds"
        reduction = 1 - medians["indexed"][field] / medians["native"][field]
        all_current = identities == [data_identity(p) for p in paths]
        checks = {"all_three_exact": all(all(p["checks"].values()) for p in pairs),
            "complete_stage_reduction_at_least10percent": reduction >= .1,
            "whole_median_not_regressed": medians["indexed"]["whole_seconds"] <= medians["native"]["whole_seconds"],
            "source_and_complete_input_identity_current": all_current}
        result = {"status": "diagnostic_passed_full_training_pending" if all(checks.values()) else "failed_verification",
            "checks": checks, "pairs": pairs, "medians": medians, "complete_stage_reduction": reduction,
            "WSL_peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "full_training_three_seeds12000_evaluation_and_recovery_executed": False, "counted": False}
        atomic_json(args.output / "RESULT.json", result)
        print(json.dumps({k: result[k] for k in ("status", "checks", "medians", "complete_stage_reduction")}), flush=True)
    except BaseException as error:
        atomic_json(args.output / "FAILURE.json", {"status": "failed_original_contract_or_harness",
            "error_type": type(error).__name__, "error": str(error), "counted": False,
            "completed_pairs_file": str(args.output / "PROGRESS.json"), "full_training_not_started": True})
        raise


if __name__ == "__main__":
    main()
