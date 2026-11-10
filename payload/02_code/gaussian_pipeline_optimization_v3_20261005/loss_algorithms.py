"""Keep the native SSIM definition while comparing convolution factorizations."""

import torch
import torch.nn.functional as functional


def moments_value(values, size_average):
    mu1, mu2, second1, second2, cross = values
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2
    sigma1_sq, sigma2_sq, sigma12 = second1-mu1_sq, second2-mu2_sq, cross-mu1_mu2
    result = ((2*mu1_mu2+.01**2)*(2*sigma12+.03**2))/((mu1_sq+mu2_sq+.01**2)*(sigma1_sq+sigma2_sq+.03**2))
    return result.mean() if size_average else result.mean(1).mean(1).mean(1)


class ConvolutionSSIM:
    def __init__(self, losses, mode):
        if mode not in ("batched", "separable"):
            raise ValueError("The registered SSIM convolution algorithm changed")
        self.losses, self.mode, self.entries = losses, mode, {}
        self.trace = {"mode": mode, "calls": 0, "kernel_builds": 0, "maximum_entries": 16,
            "mixed_precision": False, "definition": "native11_gaussian_zero_padding_full_pixels"}

    def __call__(self, a, b, window_size=11, size_average=True):
        if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device or a.size(-3) != 3 or window_size != 11:
            raise ValueError("The native RGB SSIM shape, device or precision contract changed")
        key = a.dtype, a.device, window_size
        if key not in self.entries:
            if len(self.entries) >= 16:
                raise ValueError("SSIM immutable kernel allocation exceeded its explicit bound")
            if self.mode == "batched":
                weights = self.losses.create_window(11, 3).to(a).repeat(5, 1, 1, 1)
            else:
                gaussian = self.losses.gaussian(11, 1.5).to(a)
                weights = (gaussian.reshape(1, 1, 1, 11).expand(3, 1, 1, 11).contiguous(),
                    gaussian.reshape(1, 1, 11, 1).expand(3, 1, 11, 1).contiguous())
            self.entries[key] = weights
            self.trace["kernel_builds"] += 1
        self.trace["calls"] += 1
        if self.mode == "batched":
            value = torch.cat((a, b, a*a, b*b, a*b), dim=-3)
            filtered = functional.conv2d(value, self.entries[key], padding=5, groups=15)
            moments = filtered.chunk(5, dim=-3)
        else:
            horizontal, vertical = self.entries[key]

            def filtered(value):
                partial = functional.conv2d(value, horizontal, padding=(0, 5), groups=3)
                return functional.conv2d(partial, vertical, padding=(5, 0), groups=3)

            moments = [filtered(value) for value in (a, b, a*a, b*b, a*b)]
        return moments_value(moments, size_average)


def native_l1(a, b):
    return torch.abs(a-b).mean()


class CompiledL1:
    def __init__(self):
        self.function = torch.compile(native_l1, fullgraph=True)
        self.trace = {"mode": "compiled_native_l1", "calls": 0, "mixed_precision": False}

    def __call__(self, a, b):
        self.trace["calls"] += 1
        return self.function(a, b)
