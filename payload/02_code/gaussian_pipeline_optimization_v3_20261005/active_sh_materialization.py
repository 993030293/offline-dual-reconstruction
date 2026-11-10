"""Materialize only the coefficients consumed by the pinned active SH degree."""

from contextlib import contextmanager
import inspect


def active_features(model):
    import torch

    count = (model.active_sh_degree + 1) ** 2
    return torch.cat((model._features_dc, model._features_rest[:, :count - 1, :]), dim=1)


@contextmanager
def active_sh_adapter(adapter, enabled=True):
    original = adapter.render
    trace = {"enabled": enabled, "calls": 0, "full_coefficients_bytes": 0,
             "consumed_coefficients_bytes": 0, "rest_gradient_remains_materialized": True}
    if not enabled:
        yield trace
        return
    source = inspect.getsource(original)
    anchor = "gaussian.get_features if override_color is None else override_color"
    if source.count(anchor) != 1:
        raise ValueError("The pinned full-K SH renderer boundary changed")

    def materialize(model):
        degree = model.active_sh_degree
        if type(degree) is not int or not 0 <= degree <= model.max_sh_degree:
            raise ValueError("The model has an invalid active SH degree")
        count = (degree + 1) ** 2
        full = (model.max_sh_degree + 1) ** 2
        if (model._features_dc.shape[1:] != (1, 3)
                or model._features_rest.shape != (len(model._features_dc), full - 1, 3)):
            raise ValueError("The declared native SH coefficient shape changed")
        trace["calls"] += 1
        trace["full_coefficients_bytes"] += len(model._features_dc) * full * 3 * model._features_dc.element_size()
        trace["consumed_coefficients_bytes"] += len(model._features_dc) * count * 3 * model._features_dc.element_size()
        return active_features(model)

    namespace = dict(original.__globals__)
    namespace["materialize_active_sh"] = materialize
    source = source.replace(anchor, "materialize_active_sh(gaussian) if override_color is None else override_color")
    exec(compile(source, "<pinned_renderer_active_sh_materialization>", "exec"), namespace)
    adapter.render = namespace["render"]
    try:
        yield trace
    finally:
        adapter.render = original
        trace["original_renderer_restored"] = adapter.render is original
