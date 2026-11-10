"""Select PyTorch's fused dense Adam without changing the training objective."""


def fused_adam_model(base_class):
    import torch

    class FusedDenseAdamModel(base_class):
        def training_setup(self, training_args):
            super().training_setup(training_args)
            if self.optimizer_type != "default":
                raise ValueError("Fused dense Adam must be compared separately from sparse Adam")
            if self.optimizer.state:
                raise ValueError("Switch Adam implementation only before initializing optimizer state")
            groups = [dict(group, fused=True, foreach=False) for group in self.optimizer.param_groups]
            self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15, fused=True, foreach=False)
            if not all(group["fused"] and group["foreach"] is False for group in self.optimizer.param_groups):
                raise AssertionError("Dense fused Adam was not activated")

    return FusedDenseAdamModel
