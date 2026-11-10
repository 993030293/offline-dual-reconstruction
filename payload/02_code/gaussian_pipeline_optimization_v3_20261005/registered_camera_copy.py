"""Register owned immutable CPU image storage without a duplicate staging buffer."""

import torch


def layout(source):
    return (source.data_ptr(), source.untyped_storage().data_ptr(),
            source.untyped_storage().nbytes(), tuple(source.shape),
            tuple(source.stride()), source.storage_offset(), source._version)


def validate_source(source):
    if (source.device.type != "cpu" or source.dtype != torch.float32
            or source.ndim != 3 or source.shape[0] != 3
            or source.numel() == 0 or source.requires_grad):
        raise ValueError("Only owned nonempty original CPU float32 RGB inputs are supported")


class HostRegistration:
    def __init__(self, sources, maximum_bytes=8 * 1024**3):
        self.sources = {id(source): (source, layout(source)) for source in sources}
        if not self.sources or maximum_bytes <= 0:
            raise ValueError("A bounded nonempty immutable image bank is required")
        storages = {}
        for source, _ in self.sources.values():
            validate_source(source)
            if source.is_pinned():
                raise ValueError("Foreign page-locked storage must not be registered or unregistered")
            storage = source.untyped_storage()
            storages[(storage.data_ptr(), storage.nbytes())] = storage
        total = sum(size for _, size in storages)
        if total > maximum_bytes:
            raise MemoryError("The declared host registration budget was exceeded")
        self.registered = []
        self.closed = False
        self.trace = {"unique_storages": len(storages), "registered_bytes": total,
                      "duplicate_CPU_staging_bytes": 0, "registrations": 0,
                      "unregistrations": 0, "closed": False}
        try:
            for (pointer, size), storage in storages.items():
                error = int(torch.cuda.cudart().cudaHostRegister(pointer, size, 0))
                if error:
                    raise RuntimeError("cudaHostRegister failed with code " + str(error))
                self.registered.append((pointer, storage))
                self.trace["registrations"] += 1
            if not all(source.is_pinned() for source, _ in self.sources.values()):
                raise RuntimeError("CUDA did not report every registered original source as pinned")
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note("Registration cleanup also failed: " + repr(cleanup_error))
            raise

    def validate(self, source):
        validate_source(source)
        item = self.sources.get(id(source))
        if self.closed or item is None or item[0] is not source or layout(source) != item[1]:
            raise ValueError("The registered input identity, layout or mutation version changed")

    def close(self):
        if self.closed:
            return
        changed = any(layout(source) != expected for source, expected in self.sources.values())
        errors = []
        for pointer, storage in tuple(reversed(self.registered)):
            error = int(torch.cuda.cudart().cudaHostUnregister(pointer))
            if error:
                errors.append(error)
            else:
                self.registered.remove((pointer, storage))
                self.trace["unregistrations"] += 1
        self.closed = not self.registered
        self.trace["closed"] = self.closed
        if errors:
            raise RuntimeError("cudaHostUnregister failed with codes " + repr(errors))
        if changed:
            raise ValueError("An original registered input changed before registration was closed")


class RegisteredCameraCopy:
    def __init__(self, mode, sources):
        if mode not in ("native", "registered"):
            raise ValueError("The registered R82 transfer modes changed")
        self.mode = mode
        self.bank = None
        self.stream = None
        self.device = None
        self.ready = None
        self.source = None
        self.closed = False
        self.trace = {"mode": mode, "copies": 0, "staging_builds": 0,
                      "pinned_bytes": 0, "device_buffer_bytes": 0,
                      "maximum_slots": 1, "no_future_camera_or_sampling_access": True,
                      "duplicate_CPU_staging_bytes": 0, "closed": False}
        if mode == "registered":
            self.bank = HostRegistration(sources)
            self.trace["registration"] = self.bank.trace
            self.trace["pinned_bytes"] = self.bank.trace["registered_bytes"]
            try:
                self.stream = torch.cuda.Stream()
            except BaseException:
                self.bank.close()
                raise

    def begin(self, source):
        validate_source(source)
        if self.closed:
            raise ValueError("A closed image transfer cannot be reused")
        self.trace["copies"] += 1
        if self.mode == "native":
            self.source = source
            return
        self.bank.validate(source)
        if self.device is None:
            self.device = torch.empty_like(source, device="cuda", memory_format=torch.preserve_format)
            self.trace["staging_builds"] += 1
            self.trace["device_buffer_bytes"] = self.device.numel() * self.device.element_size()
        if source.shape != self.device.shape or source.stride() != self.device.stride():
            raise ValueError("The single owned device slot cannot change input layout")
        # The last default-stream consumer must finish before the next copy overwrites its slot.
        self.stream.wait_event(torch.cuda.current_stream().record_event())
        with torch.cuda.stream(self.stream):
            self.device.copy_(source, non_blocking=True)
            self.ready = self.stream.record_event()

    def consume(self):
        if self.closed or (self.source is None if self.mode == "native" else self.ready is None):
            raise ValueError("An image copy must begin before it is consumed")
        if self.mode == "native":
            return self.source.cuda()
        torch.cuda.current_stream().wait_event(self.ready)
        self.device.record_stream(torch.cuda.current_stream())
        return self.device

    def close(self):
        if self.closed:
            return
        if self.stream is not None:
            self.stream.synchronize()
        torch.cuda.current_stream().synchronize()
        try:
            if self.bank is not None:
                self.bank.close()
        finally:
            self.device, self.ready, self.source = None, None, None
            self.closed = self.bank is None or self.bank.closed
            self.trace["closed"] = self.closed
