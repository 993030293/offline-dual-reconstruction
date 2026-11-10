"""Batch scalar telemetry while preserving chronological Python EMA arithmetic."""

import math


class DeferredTelemetry:
    def __init__(self, capacity=10, initial_ema=0.):
        import torch
        if not 1 <= capacity <= 32 or not isinstance(initial_ema, float) or not math.isfinite(initial_ema):
            raise ValueError("Telemetry requires a finite native EMA and bounded capacity")
        self.capacity, self.ema = capacity, initial_ema
        self.pending, self.values, self.elapsed_ms = [], [], []
        self.events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(capacity)]
        self.trace = {"capacity": capacity, "samples": 0, "batches": 0, "maximum_pending": 0,
            "native_CPU_double_EMA_order_and_coefficients": True, "no_autograd_graph_retained": True,
            "maximum_scalar_payload_bytes": capacity*4}

    def event_pair(self):
        if len(self.pending) >= self.capacity:
            raise RuntimeError("Flush telemetry before reusing an owned CUDA event slot")
        return self.events[len(self.pending)]

    def push(self, loss, start, end, force=False):
        import torch
        if loss.device.type != "cuda" or loss.dtype != torch.float32 or loss.ndim != 0:
            raise ValueError("The registered native telemetry loss must be a scalar CUDA float32 tensor")
        if (start, end) != self.events[len(self.pending)]:
            raise ValueError("The loss does not own the next bounded timing event pair")
        self.pending.append((loss.detach(), start, end))
        self.trace["samples"] += 1
        self.trace["maximum_pending"] = max(self.trace["maximum_pending"], len(self.pending))
        if len(self.pending) == self.capacity or force:
            self.flush()
        return self.ema

    def flush(self):
        import torch
        if not self.pending:
            return self.ema
        # One ordered copy returns the original float32 scalars to Python doubles.
        values = torch.stack([loss for loss, _, _ in self.pending]).cpu().tolist()
        elapsed = [start.elapsed_time(end) for _, start, end in self.pending]
        for value in values:
            self.ema = .4*value+.6*self.ema
        self.values.extend(values)
        self.elapsed_ms.extend(elapsed)
        self.pending.clear()
        self.trace["batches"] += 1
        return self.ema

    def close(self):
        self.flush()
        if len(self.values) != self.trace["samples"] or len(self.elapsed_ms) != len(self.values):
            raise ValueError("The complete scalar or CUDA timing denominator changed")
        self.trace["all_samples_and_events_consumed"] = True
        return self.trace
