"""Reduce active SH payloads and unused late density work without changing raster kernels."""

from contextlib import contextmanager
import inspect


@contextmanager
def gsplat_execution_specialization(active_sh=False, late_density=False):
    import torch
    import gsplat_adapter as adapter
    from scene.gaussian_model import GaussianModel
    original_render, original_raster = adapter.render, adapter.rasterize_values
    original_setup, original_update = GaussianModel.training_setup, GaussianModel.update_learning_rate
    trace = {"actual_render_calls": 0, "active_payload_calls": 0, "late_density_calls": 0,
        "SH_color_bytes_baseline_sum": 0, "SH_color_bytes_effective_sum": 0,
        "raster_kernels_and_covariance_projection_unchanged": True,
        "unused_SH_gradient_is_still_materialized_for_native_Adam": True,
        "density_stop_comes_from_native_training_options": True}

    def features(model):
        if not active_sh:
            return model.get_features
        count = (model.active_sh_degree+1)**2
        trace["active_payload_calls"] += 1
        trace["SH_color_bytes_baseline_sum"] += model._features_dc.shape[0]*(model.max_sh_degree+1)**2*3*model._features_dc.element_size()
        trace["SH_color_bytes_effective_sum"] += model._features_dc.shape[0]*count*3*model._features_dc.element_size()
        return torch.cat((model._features_dc, model._features_rest[:, :count-1, :]), dim=1)

    def density(model):
        trace["actual_render_calls"] += 1
        if not late_density:
            return True
        if not hasattr(model, "_execution_iteration") or not hasattr(model, "_execution_density_until"):
            raise ValueError("A late density specialization requires the actual native iteration and densification schedule")
        required = model._execution_iteration < model._execution_density_until
        trace["late_density_calls"] += int(not required)
        return required

    raster = inspect.getsource(original_raster)
    signature = "background, sh_degree=None, screen_proxy=None):"
    boundary = "    with torch.no_grad():\n        conics = info[\"conics\"][0].detach()"
    if raster.count(signature) != 1 or raster.count(boundary) != 1:
        raise ValueError("The pinned raster density bridge changed its guarded boundaries")
    raster = raster.replace(signature, "background, sh_degree=None, screen_proxy=None, density_required=True):")
    raster = raster.replace(boundary,
        "    if not density_required:\n        return rendered[0].permute(2, 0, 1), None, info\n"+boundary)
    raster_namespace = dict(original_raster.__globals__)
    exec(compile(raster, "<native_raster_optional_unused_density>", "exec"), raster_namespace)
    render = inspect.getsource(original_render)
    replacements = [
        ("    proxy = torch.zeros_like(gaussian.get_xyz, requires_grad=torch.is_grad_enabled())",
         "    density_required = execution_density(gaussian)\n    proxy = torch.zeros_like(gaussian.get_xyz, requires_grad=torch.is_grad_enabled()) if density_required else None"),
        ("gaussian.get_features if override_color is None else override_color",
         "execution_features(gaussian) if override_color is None else override_color"),
        ("screen_proxy=proxy)", "screen_proxy=proxy, density_required=density_required)"),
        ('"visibility_filter": (radii > 0).nonzero()', '"visibility_filter": None if radii is None else (radii > 0).nonzero()')]
    for before, after in replacements:
        if render.count(before) != 1:
            raise ValueError("The original dense RGB renderer changed its native execution contract")
        render = render.replace(before, after)
    render_namespace = dict(original_render.__globals__)
    render_namespace.update(execution_density=density, execution_features=features,
        rasterize_values=raster_namespace["rasterize_values"])
    exec(compile(render, "<native_SH_and_late_density_execution>", "exec"), render_namespace)

    def setup(model, options):
        if options.optimizer_type != "default" or not 0 < options.densify_until_iter < options.iterations:
            raise ValueError("The late renderer requires native dense Adam and a declared finite densification schedule")
        result = original_setup(model, options)
        model._execution_density_until = options.densify_until_iter
        model._execution_iteration = 0
        return result

    def update(model, iteration):
        result = original_update(model, iteration)
        model._execution_iteration = iteration
        return result

    adapter.render, adapter.rasterize_values = render_namespace["render"], raster_namespace["rasterize_values"]
    if late_density:
        GaussianModel.training_setup, GaussianModel.update_learning_rate = setup, update
    try:
        yield trace
    finally:
        adapter.render, adapter.rasterize_values = original_render, original_raster
        GaussianModel.training_setup, GaussianModel.update_learning_rate = original_setup, original_update

