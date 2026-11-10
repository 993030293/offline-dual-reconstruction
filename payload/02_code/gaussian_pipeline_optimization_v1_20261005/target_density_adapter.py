"""Restrict extra gradient-driven densification to a fixed metric target volume."""

import numpy as np


def target_densification_model(base, minimum, maximum, multiplier=2.):
    lower, upper = np.asarray(minimum, dtype=np.float64), np.asarray(maximum, dtype=np.float64)
    if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or not (lower < upper).all() or multiplier not in (1., 2.)):
        raise ValueError("A finite nonempty AABB and the registered gradient multiplier are required")

    class TargetDensityModel(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.target_density_events = []

        def target_membership(self, count=None):
            import torch
            xyz = self.get_xyz if count is None else self.get_xyz[:count]
            lo = torch.as_tensor(lower, device=xyz.device, dtype=xyz.dtype)
            hi = torch.as_tensor(upper, device=xyz.device, dtype=xyz.dtype)
            return ((xyz >= lo) & (xyz <= hi)).all(dim=1)

        def _target_grads(self, grads, threshold, kind, extent):
            import torch
            if (grads.ndim != 2 or grads.shape[1] != 1 or grads.shape[0] > self.get_xyz.shape[0]
                    or not torch.isfinite(grads).all().item()):
                raise ValueError("Densification requires one finite gradient per original point")
            mask = self.target_membership(grads.shape[0])
            weighted = grads*torch.where(mask[:, None], multiplier, 1.)
            sizes = self.get_scaling[:grads.shape[0]].max(dim=1).values
            eligible_size = (sizes <= self.percent_dense*extent if kind == "clone"
                             else sizes > self.percent_dense*extent)
            original = (grads.norm(dim=1) >= threshold) & eligible_size
            proposed = (weighted.norm(dim=1) >= threshold) & eligible_size
            self.target_density_events.append({"operation": kind, "original_gradient_count": len(grads),
                "model_count_before": len(self.get_xyz), "target_count": int(mask.sum().item()),
                "original_eligible": int(original.sum().item()), "weighted_eligible": int(proposed.sum().item()),
                "extra_inside": int((proposed & ~original & mask).sum().item()),
                "extra_outside": int((proposed & ~original & ~mask).sum().item())})
            return weighted

        def densify_and_clone(self, grads, grad_threshold, scene_extent):
            weighted = self._target_grads(grads, grad_threshold, "clone", scene_extent)
            return super().densify_and_clone(weighted, grad_threshold, scene_extent)

        def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
            weighted = self._target_grads(grads, grad_threshold, "split", scene_extent)
            return super().densify_and_split(weighted, grad_threshold, scene_extent, N)

        def target_density_receipt(self):
            return {"minimum": lower.tolist(), "maximum": upper.tolist(), "multiplier": multiplier,
                "final_total_gaussians": len(self.get_xyz),
                "final_target_gaussians": int(self.target_membership().sum().item()),
                "count_basis": "post_loop_in_memory_state_native_save_precedes_last_densification",
                "extra_candidates_inside_sum": sum(row["extra_inside"] for row in self.target_density_events),
                "extra_candidates_outside_sum": sum(row["extra_outside"] for row in self.target_density_events),
                "events": self.target_density_events,
                "loss_or_optimizer_gradients_modified": False, "native_pruning_unchanged": True}

    return TargetDensityModel
