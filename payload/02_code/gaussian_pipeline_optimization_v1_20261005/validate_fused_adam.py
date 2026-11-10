"""Compare real Gaussian optimizer groups through updates, growth, pruning and reset."""

import argparse
import json
from pathlib import Path

from optimizer_adapter import fused_adam_model
from runtime import activate_reference, atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = activate_reference(args.runtime)
    import torch
    from arguments import OptimizationParams
    from scene.gaussian_model import GaussianModel
    torch.set_num_threads(2)
    torch.manual_seed(3921)
    option_parser = argparse.ArgumentParser()
    group = OptimizationParams(option_parser)
    native = option_parser.parse_args([])
    frozen = json.loads((args.runtime/"MAIN_CONFIG.json").read_text())["parameters"]
    for name, value in frozen.items():
        if hasattr(native, name):
            setattr(native, name, value)
    options = group.extract(native)
    names = ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]
    shapes = [(4096, 3), (4096, 1, 3), (4096, 15, 3), (4096, 1), (4096, 3), (4096, 4)]
    initial = [torch.randn(shape, device="cuda")*.2 for shape in shapes]
    models = [GaussianModel(3), fused_adam_model(GaussianModel)(3)]
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
    limits = {"parameter_atol": 3e-6, "parameter_rtol": 1e-5, "moment_atol": 3e-6, "moment_rtol": 1e-5}
    atomic_json(args.output/"CONTRACT.json", {"limits": limits, "steps": 200,
        "interventions": {"60": "append 64 gaussians", "110": "remove every eleventh gaussian", "150": "reset opacity"},
        "same_gradients_and_learning_rate_schedule": True, "torch": torch.__version__, "runtime": metadata,
        "source": data_identity(Path(__file__)), "source_adapter": data_identity(Path(__file__).with_name("optimizer_adapter.py")),
        "speed_measurement": False, "parallel_CPU_SfM_may_be_active": True})
    rows = []
    for iteration in range(1, 201):
        for name in names:
            value = getattr(models[0], name)
            gradient = torch.randn_like(value)*(.001 if iteration % 3 else 1e-10)
            if iteration % 17 == 0:
                gradient.zero_()
            for model in models:
                getattr(model, name).grad = gradient.clone()
        for model in models:
            model.update_learning_rate(iteration)
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
        if iteration == 60:
            extension = [getattr(models[0], name)[:64].detach().clone() for name in names]
            for model in models:
                model.densification_postfix(*[value.clone() for value in extension], torch.zeros(64, device="cuda"))
        if iteration == 110:
            mask = torch.arange(models[0].get_xyz.shape[0], device="cuda") % 11 == 0
            for model in models:
                model.prune_points(mask)
        if iteration == 150:
            for model in models:
                model.reset_opacity()
        if iteration in [1, 60, 110, 150, 200]:
            for name in names:
                a, b = [getattr(model, name) for model in models]
                row = {"iteration": iteration, "parameter": name, "count": a.numel(),
                       "parameter_max_abs": float((a-b).abs().max()),
                       "parameter_passed": torch.allclose(a, b, atol=limits["parameter_atol"], rtol=limits["parameter_rtol"])}
                for moment in ["exp_avg", "exp_avg_sq"]:
                    ma, mb = [model.optimizer.state[getattr(model, name)][moment] for model in models]
                    row[moment+"_max_abs"] = float((ma-mb).abs().max())
                    row[moment+"_passed"] = torch.allclose(ma, mb, atol=limits["moment_atol"], rtol=limits["moment_rtol"])
                rows.append(row)
    passed = all(row["parameter_passed"] and row["exp_avg_passed"] and row["exp_avg_sq_passed"] for row in rows)
    result = {"status": "numerical_passed_training_pending" if passed else "failed_numerical_guard", "rows": rows,
        "max_parameter_difference": max(row["parameter_max_abs"] for row in rows), "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "max_parameter_difference": result["max_parameter_difference"]}))
    if not passed:
        raise AssertionError("Fused dense Adam differed beyond registered numerical limits")


if __name__ == "__main__":
    main()
