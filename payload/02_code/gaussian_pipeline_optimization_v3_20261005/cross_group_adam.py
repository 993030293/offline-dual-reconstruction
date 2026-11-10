"""Batch independent Adam groups with PyTorch2.7.1's native operation order.

This adapter uses the installed PyTorch APIs and Adam definition. It does not
select fused Adam, alter any learning rate, or approximate optimizer state.
"""

import numbers

import torch


class CrossGroupAdam(torch.optim.Adam):
    @torch.no_grad()
    def step(self, closure=None):
        if torch.__version__ != "2.7.1+cu128":
            raise RuntimeError("The private Adam initialization interface is pinned to the existing runtime")
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        parameters, gradients, first, second, counters, rates = [], [], [], [], [], []
        settings, storage = None, set()
        for group in self.param_groups:
            options = (group["betas"], group["eps"])
            if settings is not None and settings != options:
                raise ValueError("Cross-group batching requires common beta and epsilon settings")
            settings = options
            if (group["amsgrad"] or group["maximize"] or group["capturable"] or group["differentiable"]
                    or group["weight_decay"] != 0 or group.get("decoupled_weight_decay", False)
                    or group["foreach"] is False or group["fused"] or not isinstance(group["lr"], numbers.Real)):
                raise ValueError("The explicit native dense non-capturable Adam admission failed")
            for parameter in group["params"]:
                if parameter.device.type != "cuda" or parameter.dtype != torch.float32 or parameter.is_complex():
                    raise ValueError("The admitted cross-group tensor precision or device changed")
                address = parameter.untyped_storage().data_ptr()
                if address in storage:
                    raise ValueError("Aliased parameter storage cannot be reordered across groups")
                storage.add(address)
            before = len(parameters)
            maximum = []
            has_complex = self._init_group(group, parameters, gradients, first, second, maximum, counters)
            if has_complex or maximum:
                raise ValueError("Complex or AMSGrad state is excluded by the explicit admission")
            rates += [float(group["lr"])]*(len(parameters)-before)
        if not parameters:
            return loss
        if any(c.device.type != "cpu" or c.numel() != 1 for c in counters):
            raise ValueError("The original non-capturable CPU step counter contract changed")
        beta1, beta2 = settings[0]
        epsilon = settings[1]
        if not isinstance(beta1, numbers.Real) or not isinstance(beta2, numbers.Real):
            raise ValueError("Tensor beta settings require the original native route")
        # Each elementwise operation matches native foreach Adam's rounding order.
        torch._foreach_add_(counters, torch.tensor(1., device="cpu"), alpha=1.)
        torch._foreach_lerp_(first, gradients, 1-beta1)
        torch._foreach_mul_(second, beta2)
        torch._foreach_addcmul_(second, gradients, gradients, 1-beta2)
        steps = [counter.item() for counter in counters]
        scales = [(rate/(1-beta1**step))*-1 for rate, step in zip(rates, steps)]
        variances = [(1-beta2**step)**.5 for step in steps]
        denominator = torch._foreach_sqrt(second)
        torch._foreach_div_(denominator, variances)
        torch._foreach_add_(denominator, epsilon)
        torch._foreach_addcdiv_(parameters, first, denominator, scales)
        return loss


def cross_group_model(base):
    class BatchedGroupsModel(base):
        def training_setup(self, options):
            super().training_setup(options)
            if self.optimizer_type != "default" or self.optimizer.state:
                raise ValueError("Select the admitted dense batching route before optimizer initialization")
            self.optimizer = CrossGroupAdam([dict(g) for g in self.optimizer.param_groups], lr=0., eps=1e-15)
    return BatchedGroupsModel
