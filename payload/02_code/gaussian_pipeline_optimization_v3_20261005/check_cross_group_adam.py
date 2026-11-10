"""Compare actual optimizer operations, state growth, reset and native recovery."""

import argparse
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import activate_reference, atomic_json, data_identity


def states_exact(left, right, names, torch):
    return all(torch.equal(getattr(left, name), getattr(right, name))
        and all(torch.equal(left.optimizer.state[getattr(left, name)][key], right.optimizer.state[getattr(right, name)][key])
            for key in ("exp_avg", "exp_avg_sq", "step")) for name in names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "model", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = activate_reference(args.runtime)
    import torch
    from arguments import OptimizationParams
    from scene.gaussian_model import GaussianModel
    from cross_group_adam import cross_group_model, CrossGroupAdam
    sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"))
    from vectorized_gaussian_read import vectorized_gaussian_read
    torch.set_num_threads(4)
    torch.manual_seed(20261005)
    option_parser = argparse.ArgumentParser()
    group = OptimizationParams(option_parser)
    options = group.extract(option_parser.parse_args([]))
    names = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
    shapes = ((4096, 3), (4096, 1, 3), (4096, 15, 3), (4096, 1), (4096, 3), (4096, 4))
    first = [torch.randn(shape, device="cuda")*.2 for shape in shapes]
    identities = [data_identity(p) for p in (Path(__file__), Path(__file__).with_name("cross_group_adam.py"), args.model)]
    protocol = {"sources_inputs": identities, "reference": reference, "torch": torch.__version__,
        "original_parameters_moments_and_step_bit_exact": True, "200_growth_prune_reset_and_native_state_recovery_steps": True,
        "three_pairs_all_real_saved_model_parameters_optimizer_step": True, "minimum_optimizer_stage_reduction": .1,
        "first_state_allocation_and_group_dispatch_cost_included": True, "maximum_CUDA_GiB": 12,
        "whole_training_and_three_seeds_not_yet_run": True, "counted": False,
        "official_API_reference": "https://github.com/pytorch/pytorch/blob/v2.7.1/torch/optim/adam.py"}
    atomic_json(args.output/"PROTOCOL.json", protocol)

    def prepare(model):
        model.spatial_lr_scale = 1.
        model.pretrained_exposures = None
        model._exposure = torch.nn.Parameter(torch.eye(3, 4, device="cuda")[None])
        model.exposure_mapping = {"owned_contract_fixture": 0}
        model.training_setup(options)
        count = len(model.get_xyz)
        model.max_radii2D = torch.zeros(count, device="cuda")
        model.tmp_radii = torch.zeros(count, device="cuda")

    models = [GaussianModel(3), cross_group_model(GaussianModel)(3)]
    for model in models:
        for name, value in zip(names, first):
            setattr(model, name, torch.nn.Parameter(value.clone()))
        prepare(model)
    rows = []
    for iteration in range(1, 201):
        for name in names:
            value = getattr(models[0], name)
            grad = torch.randn_like(value)*(.001 if iteration%3 else 1e-10)
            if iteration%17 == 0:
                grad.zero_()
            for model in models:
                getattr(model, name).grad = None if iteration%29 == 0 and name == "_scaling" else grad.clone()
        for model in models:
            model.update_learning_rate(iteration)
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
        if iteration == 60:
            extension = [getattr(models[0], name)[:64].detach().clone() for name in names]
            for model in models:
                model.densification_postfix(*[value.clone() for value in extension], torch.zeros(64, device="cuda"))
        if iteration == 110:
            mask = torch.arange(len(models[0].get_xyz), device="cuda")%11 == 0
            for model in models:
                model.prune_points(mask)
        if iteration == 150:
            for model in models:
                model.reset_opacity()
        if iteration in (1, 60, 110, 150, 180, 200):
            exact = states_exact(*models, names, torch)
            rows.append({"iteration": iteration, "all_parameters_moments_steps_bit_exact": exact})
            if not exact:
                raise ValueError("Cross-group batching changed an original parameter or optimizer state bit")
        if iteration == 180:
            path = args.output/"native_complete_optimizer.pth"
            torch.save(models[1].optimizer.state_dict(), path)
            restored = CrossGroupAdam([dict(g) for g in models[1].optimizer.param_groups], lr=0., eps=1e-15)
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
            models[1].optimizer = restored
    del models, first
    torch.cuda.empty_cache()
    timing, pairs = {}, []
    for trial in range(3):
        state_pairs = {}
        for mode in (["native", "cross_group"] if trial%2 == 0 else ["cross_group", "native"]):
            model = GaussianModel(3) if mode == "native" else cross_group_model(GaussianModel)(3)
            with vectorized_gaussian_read(True):
                model.load_ply(str(args.model))
            prepare(model)
            for name in names:
                parameter = getattr(model, name)
                parameter.grad = torch.full_like(parameter, .001)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for iteration in range(1, 81):
                model.update_learning_rate(iteration)
                model.optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            state_pairs[mode] = {name: {"parameter": getattr(model, name).detach().cpu(),
                **{key: model.optimizer.state[getattr(model, name)][key].detach().cpu() for key in ("exp_avg", "exp_avg_sq", "step")}} for name in names}
            timing[mode] = elapsed
            if torch.cuda.max_memory_reserved() > 12*1024**3:
                raise ValueError("The original CUDA budget was exceeded")
            del model
            torch.cuda.empty_cache()
        exact = all(torch.equal(state_pairs["native"][name][key], state_pairs["cross_group"][name][key])
            for name in names for key in state_pairs["native"][name])
        row = {"trial": trial, "seconds": dict(timing), "all_real_saved_model_optimizer_state_exact": exact}
        pairs.append(row)
        print(row, flush=True)
        if not exact:
            raise ValueError("A real optimizer stage changed its complete parameter state")
        del state_pairs
    median = {mode: statistics.median(p["seconds"][mode] for p in pairs) for mode in ("native", "cross_group")}
    reduction = 1-median["cross_group"]/median["native"]
    checks = {"complete_lifecycle_and_native_restore_exact": all(r["all_parameters_moments_steps_bit_exact"] for r in rows),
        "all_three_real_parameter_stage_pairs_exact": all(p["all_real_saved_model_optimizer_state_exact"] for p in pairs),
        "optimizer_stage_reduction_at_least10percent": reduction >= .1,
        "sources_inputs_unchanged": identities == [data_identity(p) for p in (Path(__file__), Path(__file__).with_name("cross_group_adam.py"), args.model)]}
    result = {"status": "exact_optimizer_stage_passed_full_training_pending" if all(checks.values()) else "failed_verification",
        "checks": checks, "lifecycle": rows, "pairs": pairs, "median_seconds": median, "reduction": reduction,
        "whole_training_executed": False, "additional_accepted_count": 0}
    atomic_json(args.output/"RESULT.json", result)
    print({k: result[k] for k in ("status", "checks", "median_seconds", "reduction")}, flush=True)


if __name__ == "__main__":
    main()
