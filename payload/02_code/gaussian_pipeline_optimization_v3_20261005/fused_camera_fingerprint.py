"""Consume exact prepared digests only within owned immutable camera images."""

import time
import weakref

from fused_camera_digest import DigestLibrary


class ImageFingerprintCache:
    def __init__(self, library, trace):
        self.library = DigestLibrary(library)
        self.trace = trace
        self.entries = {}
        self.trace.update(prepared=0, consumed=0, native_fallback=0, active_entries=0, workers=4,
                          scratch_bytes_per_worker=32768, full_planar_or_Python_bytes_materializations=0,
                          owned_immutable_tensor_scope_not_arbitrary_external_mutation=True,
                          selected_AVX2=self.library.avx2)

    @staticmethod
    def binding(value):
        return (value._version, value.data_ptr(), value.untyped_storage().data_ptr(), value.storage_offset(),
                value.shape, value.stride(), value.dtype, value.device, value.requires_grad)

    def prepare(self, cameras):
        if self.entries or self.trace["prepared"]:
            raise RuntimeError("The owned image batch cannot be prepared twice")
        values = [camera.original_image for camera in cameras]
        if not all(self.library.eligible(value) for value in values):
            raise ValueError("The admitted complete fused camera consumer requires native interleaved float32 views")
        started = time.perf_counter()
        bindings = [self.binding(value) for value in values]
        digests = self.library.images(values, workers=4)
        if bindings != [self.binding(value) for value in values]:
            raise RuntimeError("An owned immutable image binding changed during preparation")
        for value, digest, binding in zip(values, digests, bindings):
            if id(value) in self.entries:
                entry = self.entries[id(value)]
                if entry["digest"] != digest or entry["binding"] != binding:
                    raise RuntimeError("A shared image binding changed its complete digest")
                entry["remaining"] += 1
            else:
                self.entries[id(value)] = {"value": weakref.ref(value), "binding": binding,
                                          "digest": digest, "remaining": 1}
        self.trace.update(prepared=len(values), active_entries=len(self.entries),
                          actual_native_batch_seconds=time.perf_counter()-started)

    def take(self, value):
        entry = self.entries.get(id(value))
        if entry is None or entry["value"]() is not value or entry["binding"] != self.binding(value):
            raise RuntimeError("An unowned, changed or already consumed image cannot reuse a digest")
        self.trace["consumed"] += 1
        entry["remaining"] -= 1
        if entry["remaining"] == 0:
            self.entries.pop(id(value))
        self.trace["active_entries"] = len(self.entries)
        return entry["digest"]

    def close(self, aborted=False):
        if not aborted and (self.entries or self.trace["prepared"] != self.trace["consumed"]):
            raise RuntimeError("The original image consumer did not consume every prepared binding")
        self.entries.clear()
        self.trace.update(active_entries=0, cache_closed=True, aborted=aborted)
