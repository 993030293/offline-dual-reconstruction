"""Own one bounded staging slot and overlap exact RGB copying with independent rendering."""

import torch


class CameraCopy:
    def __init__(self, mode):
        if mode not in ("native", "nonblocking", "streamed"):
            raise ValueError("The registered camera transfer algorithm changed")
        self.mode = mode
        self.stream = torch.cuda.Stream() if mode == "streamed" else None
        self.host, self.device, self.ready = None, None, None
        self.trace = {"mode": mode, "copies": 0, "staging_builds": 0, "pinned_bytes": 0, "device_buffer_bytes": 0,
            "maximum_slots": 1, "no_future_camera_or_sampling_access": True}

    def begin(self, source):
        if source.device.type != "cpu" or source.dtype != torch.float32 or source.ndim != 3 or source.shape[0] != 3:
            raise ValueError("Only the native complete CPU RGB tensors may be transferred")
        self.trace["copies"] += 1
        if self.mode == "native":
            self.source = source
            return
        if self.mode == "nonblocking":
            self.device = source.cuda(non_blocking=True)
            return
        if self.host is None:
            self.host = torch.empty_like(source, pin_memory=True, memory_format=torch.preserve_format)
            self.device = torch.empty_like(source, device="cuda", memory_format=torch.preserve_format)
            self.trace.update(staging_builds=1, pinned_bytes=self.host.numel()*self.host.element_size(),
                device_buffer_bytes=self.device.numel()*self.device.element_size())
        if source.shape != self.host.shape or source.stride() != self.host.stride():
            raise ValueError("A single bounded camera staging slot cannot silently change input layout")
        if self.ready is not None:
            self.ready.synchronize()
        self.host.copy_(source)
        # The prior default-stream consumer finishes before overwriting its device slot.
        prior = torch.cuda.current_stream().record_event()
        self.stream.wait_event(prior)
        with torch.cuda.stream(self.stream):
            self.device.copy_(self.host, non_blocking=True)
            self.ready = self.stream.record_event()

    def consume(self):
        if self.mode == "native":
            return self.source.cuda()
        if self.mode == "streamed":
            torch.cuda.current_stream().wait_event(self.ready)
            self.device.record_stream(torch.cuda.current_stream())
        return self.device

    def close(self):
        if self.stream is not None:
            self.stream.synchronize()
        torch.cuda.current_stream().synchronize()
        self.host, self.device, self.ready = None, None, None
