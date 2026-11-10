"""Reuse one owned row index during one complete native Gaussian prune."""

from contextlib import contextmanager
import inspect
import textwrap
import time

import torch


GROUP_NAMES = {"xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"}
AUXILIARY_NAMES = ("xyz_gradient_accum", "denom", "max_radii2D", "tmp_radii")


def admitted(model, mask):
    if not isinstance(mask, torch.Tensor) or mask.ndim != 1 or mask.dtype != torch.bool:
        return False
    if mask.device.type != "cuda":
        return False
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None or len(optimizer.param_groups) != len(GROUP_NAMES):
        return False
    names, owners = set(), set()
    for group in optimizer.param_groups:
        if len(group["params"]) != 1 or group.get("name") in names:
            return False
        names.add(group.get("name"))
        parameter = group["params"][0]
        if (not isinstance(parameter, torch.nn.Parameter) or parameter.ndim < 1
                or parameter.shape[0] != mask.numel() or parameter.device != mask.device
                or id(parameter) in owners):
            return False
        owners.add(id(parameter))
        state = optimizer.state.get(parameter)
        if state is not None:
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = state.get(key)
                if (not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape
                        or tensor.device != parameter.device or tensor.dtype != parameter.dtype):
                    return False
    if names != GROUP_NAMES:
        return False
    for name in AUXILIARY_NAMES:
        value = getattr(model, name, None)
        if (not isinstance(value, torch.Tensor) or value.ndim < 1
                or value.shape[0] != mask.numel() or value.device != mask.device):
            return False
    return True


def make_prune(base, trace):
    optimizer_source = inspect.getsource(base._prune_optimizer)
    prune_source = inspect.getsource(base.prune_points)
    if optimizer_source.count("[mask]") != 4 or prune_source.count("[valid_points_mask]") != 4:
        raise ValueError("The complete pinned native prune selections changed")
    optimizer_source = textwrap.dedent(optimizer_source)
    prune_source = textwrap.dedent(prune_source)
    before = "def _prune_optimizer(self, mask):"
    if optimizer_source.count(before) != 1:
        raise ValueError("The native private prune signature changed")
    optimizer_source = optimizer_source.replace(before,
        "def _owned_prune_optimizer(self, mask, owned_indices):")
    optimizer_source = optimizer_source.replace("[mask]", ".index_select(0, owned_indices)")
    replacements = [
        ("def prune_points(self, mask):", "def _owned_prune_points(self, mask):"),
        ("valid_points_mask = ~mask", "valid_points_mask = ~mask\n    owned_indices = valid_points_mask.nonzero(as_tuple=True)[0]\n    _record_index(mask.numel(), owned_indices.numel())\n    if owned_indices.numel() == 0:\n        _record_empty_route()\n        return _native_prune_points(self, mask)"),
        ("self._prune_optimizer(valid_points_mask)",
         "_owned_prune_optimizer(self, valid_points_mask, owned_indices)"),
    ]
    for before, after in replacements:
        if prune_source.count(before) != 1:
            raise ValueError("A native complete prune binding changed")
        prune_source = prune_source.replace(before, after)
    prune_source = prune_source.replace("[valid_points_mask]", ".index_select(0, owned_indices)")
    namespace = dict(base.prune_points.__globals__)

    def record(before, after):
        trace["indices_created"] += 1
        trace["input_points"] += before
        trace["retained_points"] += after

    namespace["_record_index"] = record
    # Native empty advanced indexing preserves its TensorIterator layout.
    def record_empty_route():
        trace["native_empty_route_calls"] += 1

    namespace["_record_empty_route"] = record_empty_route
    namespace["_native_prune_points"] = base.prune_points
    exec(compile(optimizer_source + "\n\n" + prune_source,
                 "<pinned_complete_prune_owned_index>", "exec"), namespace)
    original = base.prune_points
    implementation = namespace["_owned_prune_points"]

    def prune_points(model, mask):
        if not admitted(model, mask):
            trace["native_route_calls"] += 1
            return original(model, mask)
        trace["indexed_route_calls"] += 1
        return implementation(model, mask)

    return prune_points


@contextmanager
def shared_prune_index(base, enabled=True):
    started = time.perf_counter()
    original = base.prune_points
    trace = {"indices_created": 0, "indexed_route_calls": 0, "native_route_calls": 0,
             "native_empty_route_calls": 0,
             "input_points": 0, "retained_points": 0}
    replacement = make_prune(base, trace) if enabled else original
    if enabled:
        base.prune_points = replacement
    trace["setup_seconds"] = time.perf_counter() - started
    try:
        yield trace
    finally:
        started = time.perf_counter()
        base.prune_points = original
        trace["restore_seconds"] = time.perf_counter() - started
        trace["original_method_restored"] = base.prune_points is original
