"""Exercise native growth/pruning and inspect fixed target-volume membership."""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from runtime import activate_reference, atomic_json, data_identity
from target_density_adapter import target_densification_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "dataset", "model-run", "roi-source", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    metadata = activate_reference(args.runtime)
    import torch
    from arguments import OptimizationParams
    from scene.gaussian_model import GaussianModel
    from plyfile import PlyData
    free, _ = torch.cuda.mem_get_info()
    budget = min(12*1024**3, free-2*1024**3)
    atomic_json(args.output/"CONTRACT.json", {"maximum_seconds": 300, "maximum_rss_bytes": 32*1024**3,
        "gpu_budget_bytes": budget, "source": data_identity(Path(__file__)), "counted": False})
    native_parser = argparse.ArgumentParser()
    group = OptimizationParams(native_parser)
    options = group.extract(native_parser.parse_args([]))
    # Each pair covers inside/outside points with small or large covariance.
    positions = np.array([[0, 0, 0], [2, 0, 0], [0, .5, 0], [2, .5, 0], [0, 0, .5], [2, 0, .5]], dtype=np.float32)
    names = ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]
    def make(klass):
        model = klass(3)
        model.spatial_lr_scale = 1.
        model.pretrained_exposures = None
        model._exposure = torch.nn.Parameter(torch.eye(3, 4, device="cuda")[None])
        model.exposure_mapping = {"synthetic": 0}
        model._xyz = torch.nn.Parameter(torch.tensor(positions, device="cuda"))
        model._features_dc = torch.nn.Parameter(torch.arange(6, dtype=torch.float32, device="cuda")[:, None, None].repeat(1, 1, 3))
        model._features_rest = torch.nn.Parameter(torch.zeros((6, 15, 3), device="cuda"))
        model._opacity = torch.nn.Parameter(torch.tensor([[0.], [0.], [0.], [0.], [-20.], [-20.]], device="cuda"))
        model._scaling = torch.nn.Parameter(torch.log(torch.tensor([.001, .001, .02, .02, .001, .001], device="cuda"))[:, None].repeat(1, 3))
        model._rotation = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]], device="cuda").repeat(6, 1))
        model.training_setup(options)
        for name in names:
            getattr(model, name).grad = torch.zeros_like(getattr(model, name))
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        model.max_radii2D = torch.zeros(6, device="cuda")
        model.xyz_gradient_accum = torch.full((6, 1), .00015, device="cuda")
        model.denom = torch.ones((6, 1), device="cuda")
        return model
    one = target_densification_model(GaussianModel, [-1]*3, [1]*3, 1.)
    two = target_densification_model(GaussianModel, [-1]*3, [1]*3, 2.)
    control, identity, candidate = make(GaussianModel), make(one), make(two)
    checks = {"membership_oracle": candidate.target_membership().tolist() == [True, False, True, False, True, False]}
    for model in (control, identity, candidate):
        torch.manual_seed(131)
        with torch.no_grad():
            model.densify_and_prune(.0002, .005, 1., None, torch.zeros(6, device="cuda"))
    checks["identity_multiplier_all_native_tensors_exact"] = all(torch.equal(getattr(control, name), getattr(identity, name)) for name in names)
    checks["native_pruning_keeps_four_control_points"] = len(control.get_xyz) == 4
    features = candidate._features_dc[:, 0, 0].detach().cpu().numpy()
    checks["expected_target_clone_and_split_ids"] = sorted(features.tolist()) == [0., 0., 1., 2., 2., 3.]
    checks["zero_padded_new_clones_not_split_again"] = len(candidate.get_xyz) == 6
    outside_equal = True
    for feature_id in (1., 3.):
        old_index = int(torch.where(control._features_dc[:, 0, 0] == feature_id)[0].item())
        new_index = int(torch.where(candidate._features_dc[:, 0, 0] == feature_id)[0].item())
        outside_equal &= all(torch.equal(getattr(control, name)[old_index], getattr(candidate, name)[new_index]) for name in names)
    checks["all_outside_points_unchanged"] = outside_equal
    checks["all_optimizer_states_resized"] = all(model.optimizer.state[getattr(model, name)]["exp_avg"].shape == getattr(model, name).shape
        and model.optimizer.state[getattr(model, name)]["exp_avg_sq"].shape == getattr(model, name).shape
        for model in (control, identity, candidate) for name in names)
    events = candidate.target_density_receipt()
    checks["extra_outside_zero"] = events["extra_candidates_outside_sum"] == 0
    checks["growth_events_explain_extra_eligibility"] = events["extra_candidates_inside_sum"] == 3
    moved = make(two)
    with torch.no_grad():
        moved._xyz.add_(torch.tensor([10., -2., 5.], device="cuda"))
    shifted = target_densification_model(GaussianModel, [9, -3, 4], [11, -1, 6])(3)
    shifted._xyz = moved._xyz
    checks["translation_membership_invariant"] = shifted.target_membership().tolist() == [True, False, True, False, True, False]
    boundary = two(3)
    boundary._xyz = torch.nn.Parameter(torch.tensor([[-1., -1., -1.], [1., 1., 1.],
        [1.001, 0., 0.], [0., -1.001, 0.], [0., 0., 1.001]], device="cuda"))
    checks["inclusive_boundary_oracle"] = boundary.target_membership().tolist() == [True, True, False, False, False]
    rotated = two(3)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], device="cuda")
    rotated._xyz = torch.nn.Parameter(boundary._xyz.detach() @ rotation.T)
    checks["right_angle_cube_rotation_oracle"] = rotated.target_membership().tolist() == [True, True, False, False, False]
    for key, lo, hi, multiplier in [("reversed_bounds", [1]*3, [-1]*3, 2), ("bad_multiplier", [-1]*3, [1]*3, 3),
            ("nonfinite_bounds", [float("nan")]*3, [1]*3, 2), ("bad_shape", [-1]*2, [1]*3, 2)]:
        try:
            target_densification_model(GaussianModel, lo, hi, multiplier)
        except ValueError:
            checks[key+"_rejected"] = True
        else:
            checks[key+"_rejected"] = False
    roi = json.loads(args.roi_source.read_text())
    lower, upper = np.array(roi["roi_minimum"]), np.array(roi["roi_maximum"])
    rows = []
    for path in (args.dataset/"sparse/0/points3D.ply", args.model_run/"model/point_cloud/iteration_12000/point_cloud.ply"):
        data = PlyData.read(path)["vertex"].data
        xyz = np.stack([data[field] for field in ("x", "y", "z")], axis=1)
        selected = ((xyz >= lower) & (xyz <= upper)).all(axis=1)
        rows.append({"source": data_identity(path), "total": len(xyz), "inside": int(selected.sum()), "fraction": float(selected.mean())})
    checks["actual_both_ROI_fractions_nonempty_under_half"] = all(0 < row["fraction"] < .5 for row in rows)
    checks["resource_bounds"] = time.monotonic()-started < 300 and torch.cuda.max_memory_reserved() < budget and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 < 32*1024**3
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "runtime": metadata,
        "roi_source": data_identity(args.roi_source), "ROI_rows": rows, "synthetic_events": events,
        "adapter": data_identity(Path(__file__).with_name("target_density_adapter.py")),
        "source": data_identity(Path(__file__)), "elapsed_seconds": time.monotonic()-started,
        "counted": False, "training_started": False}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "ROI_rows", "elapsed_seconds")}))
    if result["status"] != "passed":
        raise RuntimeError("Target density numeric or membership admission failed")


if __name__ == "__main__":
    main()
