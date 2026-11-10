"""Train a fresh native reference or an identity-bound accepted execution variant."""

import argparse
from contextlib import ExitStack, contextmanager
import inspect
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
V3 = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005"
sys.path[:0] = [str(V1), str(V2), str(V3)]
from runtime import atomic_json, data_identity
import train_case
import checkpoint_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "optimized"), required=True)
    args, remaining = parser.parse_known_args()
    output = Path(remaining[remaining.index("--output") + 1])
    if output.exists():
        raise FileExistsError("Fresh training requires an empty model output")
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = output.parent / (output.name + "_execution_sources")
    snapshot.mkdir(exist_ok=False)
    source_paths = [Path(__file__), V2 / "camera_execution.py",
                    V2 / "prefetch_camera_images.py", V3 / "native_feature_zero.py"]
    identities = [data_identity(path) for path in source_paths]
    for path in source_paths:
        shutil.copyfile(path, snapshot / path.name)
    external = {"mode": args.mode, "source_sha256": [row["sha256"] for row in identities],
        "camera_prefetch_workers": 4, "camera_prefetch_capacity": 8,
        "native_training_formulas_unchanged": True, "camera_full_K_common_to_both_sides": True}
    atomic_json(output.parent / (output.name + "_execution_contract.json"),
        {"status": "frozen_before_actual_training", "arguments": remaining, "external": external, "sources": identities})
    original_activate = train_case.activate_reference
    original_loader = train_case.camera_loader_adapter
    original_recovery = checkpoint_adapter.RecoveryState
    original_json, original_argv = train_case.atomic_json, sys.argv
    traces = {}

    class BoundRecoveryState(original_recovery):
        def __init__(self, destination, identity, interval, resume=None, crash_after=None):
            if "external_adapters" in identity:
                raise ValueError("Recovery already contains an external execution binding")
            super().__init__(destination, {**identity, "external_adapters": external},
                             interval, resume, crash_after)

    with ExitStack() as stack:
        def activate(*parameters, **keywords):
            result = original_activate(*parameters, **keywords)
            if args.mode == "optimized":
                from scene.gaussian_model import GaussianModel
                import torch
                if torch.get_default_dtype() != torch.float32 or str(torch.get_default_device()) != "cpu":
                    raise ValueError("Exact SH initialization requires the native CPU float32 defaults")
                native = GaussianModel.create_from_pcd
                source = inspect.getsource(native)
                import textwrap
                source = textwrap.dedent(source)
                anchor = "features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()"
                if source.count(anchor) != 1:
                    raise ValueError("The native SH zero initializer changed")
                modified = source.replace(anchor, "features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float32, device='cuda')")
                namespace = dict(native.__globals__)
                exec(compile(modified, "<native_exact_SH_device_zero>", "exec"), namespace)
                actual = namespace["create_from_pcd"]
                trace = {"calls": 0, "known_zero_upload_bytes_avoided": 0, "accepted_work_not_speed_claim": True}
                traces["R157"] = trace

                def initialize(model, pcd, cameras, spatial_lr_scale):
                    trace["calls"] += 1
                    trace["known_zero_upload_bytes_avoided"] += len(pcd.points) * 3 * (model.max_sh_degree + 1)**2 * 4
                    return actual(model, pcd, cameras, spatial_lr_scale)

                GaussianModel.create_from_pcd = initialize
                stack.callback(setattr, GaussianModel, "create_from_pcd", native)
            return result

        @contextmanager
        def loader(dataset, enabled=True):
            with ExitStack() as camera_stack:
                rows = camera_stack.enter_context(original_loader(dataset, enabled))
                if args.mode == "optimized":
                    from camera_execution import camera_execution
                    from prefetch_camera_images import prefetch_camera_images
                    traces["Q58_Q59_Q60"] = camera_stack.enter_context(camera_execution(True, True, True))
                    traces["Q31"] = camera_stack.enter_context(prefetch_camera_images(4, 8))
                yield rows

        def publish(path, value):
            if Path(path) == output / "CONFIG.json":
                value = {**value, "external_adapters": external, "external_adapter_sources": identities}
            original_json(path, value)

        try:
            train_case.activate_reference = activate
            train_case.camera_loader_adapter = loader
            checkpoint_adapter.RecoveryState = BoundRecoveryState
            train_case.atomic_json = publish
            sys.argv = [str(Path(train_case.__file__)), *remaining]
            train_case.main()
        finally:
            train_case.activate_reference = original_activate
            train_case.camera_loader_adapter = original_loader
            checkpoint_adapter.RecoveryState = original_recovery
            train_case.atomic_json = original_json
            sys.argv = original_argv
    trace_values = {key: value.trace if hasattr(value, "trace") else value for key, value in traces.items()}
    atomic_json(output / "EXECUTION_MODULES.json", {"sources": identities, "external": external,
        "traces": trace_values, "status": "actual_training_complete_quality_verification_pending"})


if __name__ == "__main__":
    main()
