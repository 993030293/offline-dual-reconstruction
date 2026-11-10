"""Preserve complete training state around the pinned INRIA training loop."""

from contextlib import contextmanager
import inspect
import os
from pathlib import Path
import random

import numpy as np

from runtime import atomic_json, data_identity


def exact_equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(exact_equal(a, b) for a, b in zip(left, right))
    return left == right


class RecoveryState:
    def __init__(self, output, identity, interval, resume=None, crash_after=None):
        self.output = Path(output)
        self.identity = identity
        self.interval = interval
        self.resume = Path(resume) if resume else None
        self.crash_after = crash_after
        self.restored = None
        self.resumed_iteration = 0
        self.sample_names = []
        self.latest = None

    def prepare(self, gaussians, options, scene):
        import torch
        if self.resume is None:
            return 0
        receipt_path = self.resume.with_suffix(".json")
        import json
        receipt = json.loads(receipt_path.read_text())
        if data_identity(self.resume)["sha256"] != receipt["checkpoint"]["sha256"]:
            raise ValueError("Checkpoint payload identity mismatch")
        state = torch.load(self.resume, weights_only=False)
        if state["schema_version"] != 1 or state["identity"] != self.identity:
            raise ValueError("Checkpoint input, schedule or implementation identity mismatch")
        if not 0 < state["iteration"] < options.iterations:
            raise ValueError("Checkpoint iteration is outside the remaining training schedule")
        legacy_rng = random.Random()
        legacy_rng.setstate(random.getstate())
        legacy_queue = [camera.image_name for camera in scene.getTrainCameras()]
        expected_rng = random.Random()
        expected_rng.setstate(state["rng_python"])
        expected_queue = list(state["remaining_cameras"])
        legacy_names, expected_names = [], []
        for _ in range(32):
            if not legacy_queue:
                legacy_queue = [camera.image_name for camera in scene.getTrainCameras()]
            if not expected_queue:
                expected_queue = list(state["camera_order"])
            legacy_names.append(legacy_queue.pop(legacy_rng.randint(0, len(legacy_queue)-1)))
            expected_names.append(expected_queue.pop(expected_rng.randint(0, len(expected_queue)-1)))
        atomic_json(self.output/"LEGACY_SAMPLER_ORACLE.json", {
            "scope": "native random-pop sampler simulated with actual fresh-scene and saved states",
            "full_native_resumed_training_run": False, "native_next_32": legacy_names,
            "required_next_32": expected_names, "native_preserves_sampling": legacy_names == expected_names})
        gaussians._exposure = state["exposure"]
        gaussians.exposure_mapping = state["exposure_mapping"]
        gaussians.pretrained_exposures = state["pretrained_exposures"]
        gaussians.restore(state["gaussian_capture"], options)
        gaussians.exposure_optimizer.load_state_dict(state["exposure_optimizer"])
        independent = torch.load(self.resume, weights_only=False)
        checks = {
            "model_optimizer_and_densification_exact": exact_equal(gaussians.capture(), independent["gaussian_capture"]),
            "exposure_exact": exact_equal(gaussians._exposure, independent["exposure"]),
            "exposure_optimizer_exact": exact_equal(gaussians.exposure_optimizer.state_dict(), independent["exposure_optimizer"]),
        }
        if not all(checks.values()):
            raise AssertionError("Restored tensors or optimizer state differ from the saved state")
        cameras = scene.getTrainCameras()
        by_name = {camera.image_name: camera for camera in cameras}
        if len(by_name) != len(cameras) or set(by_name) != set(state["camera_order"]):
            raise ValueError("Checkpoint camera identity mismatch")
        cameras[:] = [by_name[name] for name in state["camera_order"]]
        self.restored = state
        self.resumed_iteration = state["iteration"]
        self.sample_names = list(state["sample_names"])
        atomic_json(self.output/"RESTORE_CHECK.json", {"status": "tensor_state_restored_loop_pending", "checks": checks,
                    "iteration": self.resumed_iteration, "checkpoint": data_identity(self.resume)})
        return self.resumed_iteration

    def restore_loop(self, scene, stack, indices, ema_loss, ema_depth):
        import torch
        if self.restored is None:
            return stack, indices, ema_loss, ema_depth
        state = self.restored
        by_name = {camera.image_name: camera for camera in scene.getTrainCameras()}
        stack = [by_name[name] for name in state["remaining_cameras"]]
        indices = list(state["remaining_indices"])
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch_cpu"])
        torch.cuda.set_rng_state_all(state["rng_torch_cuda"])
        checks = {
            "python_rng_exact": exact_equal(random.getstate(), state["rng_python"]),
            "numpy_rng_exact": exact_equal(np.random.get_state(), state["rng_numpy"]),
            "torch_cpu_rng_exact": exact_equal(torch.get_rng_state(), state["rng_torch_cpu"]),
            "torch_cuda_rng_exact": exact_equal(torch.cuda.get_rng_state_all(), state["rng_torch_cuda"]),
            "remaining_camera_order_exact": [camera.image_name for camera in stack] == state["remaining_cameras"],
            "camera_indices_exact": indices == state["remaining_indices"],
        }
        if not all(checks.values()):
            raise AssertionError("Restored random state or camera queue differs")
        import json
        receipt = json.loads((self.output/"RESTORE_CHECK.json").read_text())
        receipt["checks"].update(checks)
        receipt["status"] = "all_saved_state_exact_before_next_training_iteration"
        atomic_json(self.output/"RESTORE_CHECK.json", receipt)
        return stack, indices, state["ema_loss"], state["ema_depth"]

    def step(self, gaussians, iteration, scene, stack, indices, ema_loss, ema_depth, options):
        import torch
        if iteration % self.interval == 0 and iteration < options.iterations:
            state = {"schema_version": 1, "identity": self.identity, "iteration": iteration,
                "gaussian_capture": gaussians.capture(), "exposure": gaussians._exposure,
                "exposure_mapping": gaussians.exposure_mapping, "pretrained_exposures": gaussians.pretrained_exposures,
                "exposure_optimizer": gaussians.exposure_optimizer.state_dict(),
                "camera_order": [camera.image_name for camera in scene.getTrainCameras()],
                "remaining_cameras": [camera.image_name for camera in stack], "remaining_indices": list(indices),
                "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                "rng_torch_cpu": torch.get_rng_state(), "rng_torch_cuda": torch.cuda.get_rng_state_all(),
                "ema_loss": ema_loss, "ema_depth": ema_depth, "sample_names": list(self.sample_names)}
            target = self.output/f"checkpoint_{iteration:06d}.pth"
            temporary = target.with_suffix(".pth.tmp")
            with temporary.open("xb") as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
            atomic_json(target.with_suffix(".json"), {"status": "complete", "iteration": iteration,
                        "checkpoint": data_identity(target), "identity": self.identity})
            self.latest = target
        if self.crash_after == iteration:
            atomic_json(self.output/"CRASH_INJECTION.json", {"iteration": iteration,
                "last_complete_checkpoint": str(self.latest), "mode": "os._exit(86)_without_cleanup"})
            os._exit(86)


@contextmanager
def recoverable_training(reference, state):
    original = reference.training
    source = inspect.getsource(original)
    replacements = [
        ("    if checkpoint:\n        (model_params, first_iter) = torch.load(checkpoint)\n        gaussians.restore(model_params, opt)\n",
         "    first_iter = recovery_state.prepare(gaussians, opt, scene)\n"),
        ("    ema_Ll1depth_for_log = 0.0\n", "    ema_Ll1depth_for_log = 0.0\n"
         "    viewpoint_stack, viewpoint_indices, ema_loss_for_log, ema_Ll1depth_for_log = recovery_state.restore_loop(\n"
         "        scene, viewpoint_stack, viewpoint_indices, ema_loss_for_log, ema_Ll1depth_for_log)\n"),
        ("        viewpoint_cam = viewpoint_stack.pop(rand_idx)\n", "        viewpoint_cam = viewpoint_stack.pop(rand_idx)\n"
         "        recovery_state.sample_names.append(viewpoint_cam.image_name)\n"),
        ("            if (iteration in checkpoint_iterations):\n", "            recovery_state.step(gaussians, iteration, scene, viewpoint_stack, viewpoint_indices,\n"
         "                                ema_loss_for_log, ema_Ll1depth_for_log, opt)\n"
         "            if (iteration in checkpoint_iterations):\n"),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError("Pinned training loop no longer matches the guarded checkpoint insertion")
        source = source.replace(before, after)
    namespace = dict(reference.__dict__)
    namespace["recovery_state"] = state
    exec(compile(source, "<isolated_recoverable_INRIA_training>", "exec"), namespace)
    reference.training = namespace["training"]
    try:
        yield state
    finally:
        reference.training = original
