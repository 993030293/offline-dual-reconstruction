"""Check the registered argument boundary and real Adam/PLY composition."""

import argparse
import copy
import inspect
import json
from pathlib import Path
import resource
import sys
import textwrap
import time

from train_current_fused_adam import ROOT, V1, OUT, compose_main, argument_prefix, bind_original_globals, train_case
from runtime import activate_reference, atomic_json, data_identity


def validate_arguments():
    originals = [textwrap.dedent(inspect.getsource(train_case.main)), compose_main()]
    functions = []
    for source in originals:
        namespace = dict(train_case.main.__globals__)
        exec(compile(argument_prefix(source), "<registered_original_argument_boundary>", "exec"), namespace)
        functions.append(namespace["main"])
    common = ["--runtime", "unused_runtime", "--dataset", "unused_dataset", "--output", "unused_output",
              "--projection", "full", "--seed", "0", "--iterations", "12000", "--gsplat-build", "unused_build",
              "--input-version", "new_map_admission_v1", "--expected-image-count", "1016"]
    tests = [("native_original", ["--vectorized-ply", "--verify-native-ply"], (True, True)),
             ("fused_original", ["--dense-adam-backend", "fused"], (True, True)),
             ("registered_combination", ["--dense-adam-backend", "fused", "--vectorized-ply", "--verify-native-ply"], (False, True)),
             ("verify_requires_vectorized", ["--verify-native-ply"], (False, False)),
             ("negative_checkpoint", ["--checkpoint-every", "-1"], (False, False)),
             ("crash_requires_checkpoint", ["--simulate-crash-after", "600"], (False, False))]
    combination = ["--dense-adam-backend", "fused", "--vectorized-ply", "--verify-native-ply"]
    for label, flags in [("compact_cache", ["--compact-image-cache"]), ("resident_combination", ["--resident-fused-adam"]),
                         ("pyramid", ["--pyramid-until", "5000"]), ("target", ["--target-density-roi", "unused_roi"]),
                         ("ssim", ["--ssim-backend", "fused"]), ("sparse", ["--optimizer-type", "sparse_adam"]),
                         ("host_compact", ["--compact-host-images", "--trim-host-after-load"]),
                         ("GPU_residency", ["--gpu-resident-inputs", "--trim-host-after-load"])]:
        tests.append((label, combination + flags, (False, False)))
    rows = []
    original_argv = sys.argv
    try:
        for label, flags, expected in tests:
            observations = []
            for function in functions:
                sys.argv = ["registered_argument_check", *common, *flags]
                try:
                    result = function()
                    observations.append({"passed": True, "dense_adam_backend": result["dense_adam_backend"]})
                except ValueError as error:
                    observations.append({"passed": False, "error": str(error)})
            if tuple(row["passed"] for row in observations) != expected:
                raise AssertionError("An original argument rejection or registered combination changed: " + label)
            if expected == (False, False) and observations[0]["error"] != observations[1]["error"]:
                raise AssertionError("An unrelated original argument error changed: " + label)
            rows.append({"label": label, "expected": list(expected), "observations": observations})
    finally:
        sys.argv = original_argv
    return rows


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OUT.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    import psutil
    process = psutil.Process()
    atomic_json(args.output / "PROCESS_IDENTITY.json", {
        "scope": "WSL", "pid": process.pid, "creation_time": process.create_time()})
    arguments = validate_arguments()
    original_loader, original_json = train_case.camera_loader_adapter, train_case.atomic_json
    probe = bind_original_globals("def main():\n    return camera_loader_adapter, atomic_json, __file__\n")
    try:
        loader_marker, publication_marker = object(), object()
        train_case.camera_loader_adapter, train_case.atomic_json = loader_marker, publication_marker
        observed_loader, observed_json, observed_file = probe()
        live_binding = (probe.__globals__ is train_case.main.__globals__ and observed_loader is loader_marker
                        and observed_json is publication_marker and observed_file == train_case.__file__)
        if not live_binding:
            raise AssertionError("A copied namespace must not lose actual live entry bindings")
    finally:
        train_case.camera_loader_adapter, train_case.atomic_json = original_loader, original_json
    metadata = activate_reference(args.runtime)
    import torch
    import torch.optim.adam as adam_module
    from arguments import OptimizationParams
    from scene.gaussian_model import GaussianModel
    from optimizer_adapter import fused_adam_model
    from ply_serialization import vectorized_ply_model
    from checkpoint_adapter import exact_equal
    torch.set_num_threads(2)
    torch.manual_seed(3921)
    parser_options = argparse.ArgumentParser()
    group = OptimizationParams(parser_options)
    native = parser_options.parse_args([])
    frozen = json.loads(args.training_config.read_text(encoding="utf-8"))
    for name, value in frozen["parameters"].items():
        if hasattr(native, name):
            setattr(native, name, value)
    options = group.extract(native)
    paths = [Path(__file__), Path(__file__).with_name("train_current_fused_adam.py"), V1 / "train_case.py",
             V1 / "optimizer_adapter.py", V1 / "ply_serialization.py", V1 / "checkpoint_adapter.py",
             Path(adam_module.__file__), Path(sys.modules[GaussianModel.__module__].__file__), args.training_config]
    identities = [data_identity(path) for path in paths]
    limits = {"parameter_atol": 3e-6, "parameter_rtol": 1e-5, "moment_atol": 3e-6, "moment_rtol": 1e-5}
    atomic_json(args.output / "CONTRACT.json", {"sources_inputs": identities, "runtime": metadata,
        "limits": limits, "steps": 200, "original_interventions": {"60": "append64", "110": "prune_each11", "150": "reset_opacity"},
        "save_checks": [60, 110, 200], "arguments": arguments, "speed_measurement": False,
        "full_training_quality_or_complete_recovery": False, "new_optimization_count": 0})
    names = ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]
    shapes = [(4096, 3), (4096, 1, 3), (4096, 15, 3), (4096, 1), (4096, 3), (4096, 4)]
    initial = [torch.randn(shape, device="cuda") * .2 for shape in shapes]
    fused = fused_adam_model(GaussianModel)
    models = [GaussianModel(3), fused(3), vectorized_ply_model(fused, True)(3)]
    for model in models:
        model.spatial_lr_scale = 1.
        model.pretrained_exposures = None
        model._exposure = torch.nn.Parameter(torch.eye(3, 4, device="cuda")[None])
        model.exposure_mapping = {"synthetic": 0}
        for name, value in zip(names, initial):
            setattr(model, name, torch.nn.Parameter(value.clone()))
        model.training_setup(options)
        model.max_radii2D = torch.zeros(4096, device="cuda")
        model.tmp_radii = torch.zeros(4096, device="cuda")
    rows, saves = [], []
    free, _ = torch.cuda.mem_get_info()
    budget = min(12 * 1024**3, free - 2 * 1024**3)
    if budget <= 0:
        raise MemoryError("Insufficient registered GPU headroom")
    for step in range(1, 201):
        for name in names:
            gradient = torch.randn_like(getattr(models[0], name)) * (.001 if step % 3 else 1e-10)
            if step % 17 == 0:
                gradient.zero_()
            for model in models:
                getattr(model, name).grad = gradient.clone()
        for model in models:
            model.update_learning_rate(step)
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
        if step == 60:
            extension = [getattr(models[0], name)[:64].detach().clone() for name in names]
            for model in models:
                model.densification_postfix(*[value.clone() for value in extension], torch.zeros(64, device="cuda"))
        if step == 110:
            mask = torch.arange(models[0].get_xyz.shape[0], device="cuda") % 11 == 0
            for model in models:
                model.prune_points(mask)
        if step == 150:
            for model in models:
                model.reset_opacity()
        if step in (1, 60, 110, 150, 200):
            for name in names:
                a, b, c = [getattr(model, name) for model in models]
                row = {"step": step, "name": name, "parameter_max_abs": float((a-b).abs().max()),
                       "original_parameter_tolerance": bool(torch.allclose(a, b, atol=3e-6, rtol=1e-5)),
                       "fused_composition_parameter_exact": bool(torch.equal(b, c))}
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    ma, mb, mc = [model.optimizer.state[getattr(model, name)][key] for model in models]
                    row[key + "_original_tolerance"] = bool(torch.allclose(ma.cpu(), mb.cpu(), atol=3e-6, rtol=1e-5))
                    row[key + "_composition_exact"] = bool(torch.equal(mb, mc))
                rows.append(row)
            if not exact_equal(models[1].capture(), models[2].capture()):
                raise AssertionError("The serializer composition changed the complete fused model capture")
        if step in (60, 110, 200):
            states = copy.deepcopy([model.capture() for model in models])
            before_rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
            paths_save = [args.output / f"save{step}" / label / "point_cloud.ply" for label in ("native", "fused", "combined")]
            for model, path in zip(models, paths_save):
                model.save_ply(str(path))
            event = models[2].ply_serialization_events[-1]
            if (not event["native_byte_oracle"]["identical"] or paths_save[1].read_bytes() != paths_save[2].read_bytes()
                    or not exact_equal(states, [model.capture() for model in models])
                    or not exact_equal(before_rng, (torch.get_rng_state(), torch.cuda.get_rng_state_all()))):
                raise AssertionError("The actual fused/serializer PLY, state or RNG save invariant failed")
            saves.append({"step": step, "same_fused_file_bytes": True, "oracle": event,
                          "all_complete_state_and_RNG_unchanged": True})
        if torch.cuda.memory_reserved() > budget or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 32 * 1024**3:
            raise MemoryError("The original numerical check resource bound was exceeded")
    boolean_keys = [key for key in rows[0] if key.endswith("_tolerance") or key.endswith("_exact")]
    checks = {"all14_argument_boundaries": len(arguments) == 14,
        "original_live_adapter_globals_and_restore": live_binding and train_case.camera_loader_adapter is original_loader and train_case.atomic_json is original_json,
        "all30_original_numeric_and_exact_composition_rows": len(rows) == 30 and all(row[key] for row in rows for key in boolean_keys),
        "all3_complete_fused_save_oracles_state_and_RNG": len(saves) == 3,
        "all9_source_input_identities_unchanged": identities == [data_identity(path) for path in paths],
        "original_flags_native_and_fused": all(not model.optimizer.param_groups[0]["fused"] if i == 0 else model.optimizer.param_groups[0]["fused"] for i, model in enumerate(models))}
    result = {"status": "current_fused_numeric_and_serializer_composition_passed_training_pending" if all(checks.values()) else "failed_generic_guards",
        "checks": checks, "rows": rows, "saves": saves, "argument_checks": arguments,
        "max_parameter_difference": max(row["parameter_max_abs"] for row in rows),
        "elapsed_seconds": time.perf_counter() - started, "WSL_peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "full_training_quality_recovery_or_speed_acceptance": False, "new_optimization_count": 0}
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "max_parameter_difference", "elapsed_seconds")}), flush=True)
    if not all(checks.values()):
        raise AssertionError("The registered current Adam composition failed")


if __name__ == "__main__":
    main()
