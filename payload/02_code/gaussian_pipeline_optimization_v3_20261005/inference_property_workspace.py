"""Reuse native property results for bounded, owned Gaussian inference models."""

import weakref

import torch


PROPERTIES = ("get_features", "get_scaling", "get_rotation", "get_opacity")
FIELDS = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")


def tensor_key(value):
    return (id(value), value.untyped_storage().data_ptr(), value.storage_offset(),
            tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device),
            value._version, value.requires_grad)


class InferencePropertyWorkspace:
    def __init__(self, native_class, enabled, maximum_bytes=256 * 1024**2):
        if type(maximum_bytes) is not int or maximum_bytes < 0:
            raise ValueError("The property workspace needs a nonnegative byte budget")
        self.native_class = native_class
        self.descriptors = {name: getattr(native_class, name) for name in PROPERTIES}
        if any(type(value) is not property or value.fget is None for value in self.descriptors.values()):
            raise ValueError("All four native getter descriptors are required")
        self.enabled, self.maximum_bytes = bool(enabled), maximum_bytes
        self.entries, self.resident_bytes, self.reserved_bytes, self.closed = {}, 0, 0, False
        self.trace = {"enabled": self.enabled, "maximum_bytes": maximum_bytes,
            "peak_resident_bytes": 0, "peak_reserved_bytes": 0, "requests": {name: 0 for name in PROPERTIES},
            "native_executions": {name: 0 for name in PROPERTIES},
            "native_CUDA_elements": {name: 0 for name in PROPERTIES},
            "hits": {name: 0 for name in PROPERTIES}, "routes": {}, "invalidations": 0,
            "models_created": 0, "models_released": 0, "closed": False}
        owner = self

        class OwnedInferenceModel(native_class):
            pass

        for name in PROPERTIES:
            def getter(model, name=name):
                return owner.read(model, name)
            setattr(OwnedInferenceModel, name, property(getter))
        self.model_class = OwnedInferenceModel

    def release(self, identity):
        entry = self.entries.pop(identity, None)
        if entry is not None:
            self.resident_bytes -= sum(row["bytes"] for row in entry["values"].values())
            self.reserved_bytes -= entry["reservation"]
            entry["values"].clear()
            self.trace["models_released"] += 1
        if not 0 <= self.resident_bytes <= self.reserved_bytes <= self.maximum_bytes:
            raise AssertionError("The property workspace byte accounting became negative")

    def native(self, model, name, route):
        self.trace["routes"][route] = self.trace["routes"].get(route, 0) + 1
        value = self.descriptors[name].fget(model)
        self.trace["native_executions"][name] += 1
        if value.device.type == "cuda":
            self.trace["native_CUDA_elements"][name] += value.numel()
        return value

    def read(self, model, name):
        if self.closed:
            raise RuntimeError("An owned property workspace is closed")
        self.trace["requests"][name] += 1
        if not self.enabled:
            return self.native(model, name, "control")
        if torch.is_grad_enabled():
            self.release(id(model))
            return self.native(model, name, "grad_enabled")
        fields = tuple(getattr(model, field) for field in FIELDS)
        if (torch.__version__ != "2.7.1+cu128"
                or any(not isinstance(value, torch.Tensor) for value in fields)
                or any(value.dtype != torch.float32 or value.device.type not in ("cpu", "cuda") for value in fields)
                or len({str(value.device) for value in fields}) != 1):
            self.release(id(model))
            return self.native(model, name, "unsupported_format")
        identity = id(model)
        key = (tuple(tensor_key(value) for value in fields),
               tuple(id(getattr(model, field)) for field in ("scaling_activation", "rotation_activation", "opacity_activation")))
        entry = self.entries.get(identity)
        if entry is not None and entry["model"]() is not model:
            raise RuntimeError("An owned property workspace model identity was reused")
        if entry is not None and (entry["key"] != key or any(row["tensor"]._version != row["version"] for row in entry["values"].values())):
            self.trace["invalidations"] += 1
            self.release(identity)
            entry = None
        if (model.scaling_activation is not torch.exp or model.opacity_activation is not torch.sigmoid
                or model.rotation_activation is not torch.nn.functional.normalize):
            self.release(identity)
            return self.native(model, name, "custom_activation_before_execution")
        if entry is None:
            prospective = sum(value.numel() * value.element_size() for value in fields[1:])
            if prospective > self.maximum_bytes - self.reserved_bytes:
                return self.native(model, name, "budget_before_execution")
            entry = {"model": weakref.ref(model, lambda reference, identity=identity: self.release(identity)),
                "key": key, "reservation": prospective, "values": {}}
            self.entries[identity] = entry
            self.reserved_bytes += prospective
            self.trace["peak_reserved_bytes"] = max(self.trace["peak_reserved_bytes"], self.reserved_bytes)
            self.trace["models_created"] += 1
        row = entry["values"].get(name)
        if row is not None:
            self.trace["hits"][name] += 1
            return row["tensor"]
        value = self.native(model, name, "first_native_result")
        size = value.numel() * value.element_size()
        if size > entry["reservation"] - sum(row["bytes"] for row in entry["values"].values()):
            self.release(identity)
            raise AssertionError("The approved property result exceeded its declared budget")
        entry["values"][name] = {"tensor": value, "version": value._version, "bytes": size}
        self.resident_bytes += size
        self.trace["peak_resident_bytes"] = max(self.trace["peak_resident_bytes"], self.resident_bytes)
        return value

    def close(self):
        if not self.closed:
            for identity in list(self.entries):
                self.release(identity)
            self.closed = True
        if self.entries or self.resident_bytes or self.reserved_bytes:
            raise AssertionError("The owned property workspace retained a closed entry")
        self.trace.update(closed=True, resident_bytes_after_close=self.resident_bytes, reserved_bytes_after_close=self.reserved_bytes)
