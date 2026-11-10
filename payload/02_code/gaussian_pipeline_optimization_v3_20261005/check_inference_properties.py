"""Check the owned workspace against the actual native Gaussian getter methods."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import psutil

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(CODE), str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005")]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    atomic_json(args.output / "PROCESS_IDENTITY.json", {"scope": "WSL", "pid": process.pid, "creation_time": process.create_time()})
    with LinuxRSSGuard(args.output, atomic_json, maximum_seconds=300, maximum_bytes=8 * 1024**3):
        metadata = activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import numpy as np
        import torch
        from scene.gaussian_model import GaussianModel
        from inference_property_workspace import FIELDS, PROPERTIES, InferencePropertyWorkspace
        import scene.gaussian_model as native_module
        random.seed(8437)
        np.random.seed(8437)
        torch.manual_seed(8437)
        checks, cases = {}, []

        def bits(value):
            return value.detach().contiguous().cpu().numpy().tobytes()

        def state(model):
            return {name: (bits(getattr(model, name)), tuple(getattr(model, name).shape),
                tuple(getattr(model, name).stride()), str(getattr(model, name).dtype),
                getattr(model, name)._version, getattr(model, name).requires_grad) for name in FIELDS}

        def rng():
            return (repr(random.getstate()), repr(np.random.get_state()), bits(torch.random.get_rng_state()),
                    tuple(bits(value) for value in torch.cuda.get_rng_state_all()))

        def populate(model, n, degree, device, dtype=torch.float32, noncontiguous=False):
            shapes = ((n, 3), (n, 1, 3), (n, (degree + 1)**2 - 1, 3), (n, 1), (n, 3), (n, 4))
            for index, (name, shape) in enumerate(zip(FIELDS, shapes)):
                length = int(np.prod(shape))
                raw = (torch.arange(length, device=device, dtype=dtype) % 17 - 8) / 16
                value = raw.reshape(shape)
                if noncontiguous and length:
                    backing = torch.empty((*shape[:-1], shape[-1] * 2), dtype=dtype, device=device)
                    backing[..., ::2].copy_(value)
                    value = backing[..., ::2]
                setattr(model, name, torch.nn.Parameter(value, requires_grad=True))
            model.active_sh_degree = degree
            return model

        for device in ("cpu", "cuda"):
            for n, degree, noncontiguous in ((0, 0, False), (1, 1, False), (7, 3, True), (129, 3, False)):
                owner = InferencePropertyWorkspace(GaussianModel, True)
                model = populate(owner.model_class(degree), n, degree, device, noncontiguous=noncontiguous)
                before, before_rng = state(model), rng()
                with torch.no_grad():
                    for name in PROPERTIES:
                        expected = owner.descriptors[name].fget(model)
                        first, second = getattr(model, name), getattr(model, name)
                        if bits(expected) != bits(first) or bits(expected) != bits(second) or first is not second:
                            raise AssertionError("A native property bit pattern or cache reuse differed")
                if state(model) != before or rng() != before_rng:
                    raise AssertionError("A complete native parameter or caller RNG changed")
                owner.close()
                cases.append({"device": device, "rows": n, "SH": degree, "noncontiguous": noncontiguous, "trace": owner.trace})
        checks["eight_native_CPU_CUDA_shape_SH_layout_cases_all_bits_inputs_RNG_and_actual_hits"] = len(cases) == 8

        owner = InferencePropertyWorkspace(GaussianModel, True)
        model = populate(owner.model_class(3), 7, 3, "cuda")
        with torch.no_grad():
            first = model.get_opacity
            model._opacity.add_(.25)
            changed = model.get_opacity
            checks["versioned_parameter_mutation_invalidates_and_exactly_recomputes"] = (owner.trace["invalidations"] == 1
                and changed is not first and bits(changed) == bits(owner.descriptors["get_opacity"].fget(model)))
            model._opacity = torch.nn.Parameter(model._opacity.clone())
            replaced = model.get_opacity
            checks["parameter_object_replacement_invalidates"] = owner.trace["invalidations"] == 2 and bits(replaced) == bits(changed)
            replaced.add_(1.)
            repaired = model.get_opacity
            checks["returned_tensor_mutation_invalidates"] = owner.trace["invalidations"] == 3 and bits(repaired) == bits(changed)
            model.opacity_activation = lambda value: torch.sigmoid(value + .1)
            activated = model.get_opacity
            checks["activation_function_replacement_invalidates"] = (owner.trace["invalidations"] == 4
                and bits(activated) == bits(owner.descriptors["get_opacity"].fget(model)))
        owner.close()
        checks["close_twice_drains_and_use_after_close_rejects"] = owner.resident_bytes == 0
        owner.close()
        try:
            with torch.no_grad():
                model.get_opacity
        except RuntimeError as error:
            checks["close_twice_drains_and_use_after_close_rejects"] &= str(error) == "An owned property workspace is closed"
        else:
            checks["close_twice_drains_and_use_after_close_rejects"] = False

        owner = InferencePropertyWorkspace(GaussianModel, True)
        model = populate(owner.model_class(1), 5, 1, "cuda")
        native = populate(GaussianModel(1), 5, 1, "cuda")
        model.get_opacity.sum().backward()
        native.get_opacity.sum().backward()
        checks["grad_enabled_uses_native_graph_and_gradient_bits"] = (bits(model._opacity.grad) == bits(native._opacity.grad)
            and owner.trace["routes"] == {"grad_enabled": 1} and not owner.entries)
        owner.close()

        owner = InferencePropertyWorkspace(GaussianModel, True, maximum_bytes=0)
        model = populate(owner.model_class(1), 5, 1, "cuda")
        with torch.no_grad():
            value = model.get_scaling
            checks["budget_rejected_before_execution_preserves_native_bits"] = (bits(value) == bits(owner.descriptors["get_scaling"].fget(model))
                and owner.trace["routes"] == {"budget_before_execution": 1} and not owner.entries)
        owner.close()

        owner = InferencePropertyWorkspace(GaussianModel, True, maximum_bytes=6000)
        models = [populate(owner.model_class(3), 17, 3, "cuda") for _ in range(2)]
        with torch.no_grad():
            for model, name in ((models[0], "get_opacity"), (models[1], "get_opacity"),
                                (models[0], "get_features"), (models[1], "get_features")):
                if bits(getattr(model, name)) != bits(owner.descriptors[name].fget(model)):
                    raise AssertionError("An interleaved capacity route changed native property bits")
        checks["two_model_interleaving_reserves_complete_entry_before_native_execution"] = (owner.trace["routes"]["budget_before_execution"] == 2
            and owner.reserved_bytes == 3808 and owner.resident_bytes <= owner.reserved_bytes <= 6000)
        owner.close()
        checks["complete_entry_reservation_released_after_close"] = owner.reserved_bytes == 0 and owner.trace["reserved_bytes_after_close"] == 0

        owner = InferencePropertyWorkspace(GaussianModel, True)
        model = populate(owner.model_class(1), 5, 1, "cpu", dtype=torch.float64)
        with torch.no_grad():
            checks["unsupported_float64_uses_native_without_cache"] = all(bits(getattr(model, name)) == bits(owner.descriptors[name].fget(model)) for name in PROPERTIES)
        checks["unsupported_float64_uses_native_without_cache"] &= not owner.entries and owner.trace["routes"] == {"unsupported_format": 4}
        owner.close()

        owner = InferencePropertyWorkspace(GaussianModel, True)
        model = populate(owner.model_class(3), 7, 3, "cuda")
        with torch.no_grad():
            model._scaling[0] = torch.tensor([-float("inf"), 90., -0.], device="cuda")
            model._opacity[0] = float("nan")
            for name in PROPERTIES:
                if bits(getattr(model, name)) != bits(owner.descriptors[name].fget(model)):
                    raise AssertionError("A native nonfinite or negative-zero result differed")
        owner.close()
        checks["native_nonfinite_overflow_and_negative_zero_bits_preserved"] = True

        owner = InferencePropertyWorkspace(GaussianModel, True)
        model = populate(owner.model_class(1), 5, 1, "cuda")
        marker = RuntimeError("REGISTERED_GETTER_ERROR")
        model.opacity_activation = lambda value: (_ for _ in ()).throw(marker)
        try:
            with torch.no_grad():
                model.get_opacity
        except RuntimeError as error:
            checks["same_native_getter_exception_propagates_without_partial_result"] = error is marker and not owner.entries
        else:
            checks["same_native_getter_exception_propagates_without_partial_result"] = False
        owner.close()
        checks["native_base_descriptors_and_class_binding_unchanged"] = (native_module.GaussianModel is GaussianModel
            and all(getattr(GaussianModel, name) is owner.descriptors[name] for name in PROPERTIES))
        torch.cuda.synchronize()
        source_paths = [Path(__file__), CODE / "inference_property_workspace.py", Path(native_module.__file__)]
        result = {"status": "generic_native_getters_passed" if all(checks.values()) else "failed_verification", "checks": checks,
            "cases": cases, "sources": [data_identity(path) for path in source_paths], "runtime": metadata,
            "whole_seconds": time.perf_counter() - STARTED, "CUDA_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "new_optimization_count": 0}
        atomic_json(args.output / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "checks": checks}), flush=True)
        if not all(checks.values()):
            raise AssertionError("An actual native Gaussian property contract differed")


if __name__ == "__main__":
    main()
