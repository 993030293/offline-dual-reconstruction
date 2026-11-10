"""Use byte-owned read-only views in the pinned immutable SQLite feature consumer."""

from contextlib import contextmanager
import inspect
import textwrap
import time


@contextmanager
def feature_blob_execution(enabled=True):
    import byte_sift_matching as byte
    import localize_with_byte_sift as entry
    import sift_feature_store as store
    old_factory, old_entry = byte.byte_sift_matching, entry.byte_sift_matching
    original_init, original_get, original_close = (getattr(store.SiftFeatureStore, name)
        for name in ("__init__", "__getitem__", "close"))
    if old_factory is not old_entry:
        raise ValueError("The pinned byte-store execution aliases have different ownership")
    started = time.perf_counter()
    trace = {"enabled": enabled, "instances": 0, "get_calls": 0, "view_calls": 0, "native_calls": 0,
        "initialization_seconds": 0., "all_get_seconds": 0., "close_seconds": 0., "decode_seconds": 0.,
        "all_owned_store_closes": 0, "skipped_array_copy_bytes": 0, "get_includes_cache_SQL_decode_and_LRU": True}
    trace.update(decode_binding_setup_seconds=0., decode_binding_restore_seconds=0.)

    def initialize(instance, *args, **kwargs):
        before = time.perf_counter()
        try:
            return original_init(instance, *args, **kwargs)
        finally:
            trace["instances"] += 1
            trace["initialization_seconds"] += time.perf_counter()-before

    def get(instance, image_id):
        before = time.perf_counter()
        try:
            return original_get(instance, image_id)
        finally:
            trace["get_calls"] += 1
            trace["all_get_seconds"] += time.perf_counter()-before

    def close(instance):
        before = time.perf_counter()
        try:
            return original_close(instance)
        finally:
            trace["all_owned_store_closes"] += 1
            trace["close_seconds"] += time.perf_counter()-before

    @contextmanager
    def factory(active=True):
        binding_started = time.perf_counter()
        with old_factory(active) as native_trace:
            original_decode = store.SiftFeatureStore.decode
            implementation = original_decode
            if enabled and active:
                source = textwrap.dedent(inspect.getsource(original_decode))
                for anchor in ("np.frombuffer(kd, np.float32).reshape(kr, kc).copy()",
                        "np.frombuffer(dd, np.uint8).reshape(dr, dc).copy()"):
                    if source.count(anchor) != 1:
                        raise ValueError("The pinned native byte feature copy call site changed")
                    source = source.replace(anchor, anchor.removesuffix(".copy()"))
                namespace = dict(original_decode.__globals__)
                namespace.update(inspect.getclosurevars(original_decode).nonlocals)
                exec(compile(source, "<original_immutable_feature_blob_views>", "exec"), namespace)
                implementation = namespace[original_decode.__name__]

            def decode(instance, image_id, record):
                before = time.perf_counter()
                try:
                    use_view = (enabled and active and len(record) == 7 and record[1] > 0 and record[4] > 0
                        and isinstance(record[3], bytes) and isinstance(record[6], bytes))
                    trace["view_calls" if use_view else "native_calls"] += 1
                    value = implementation(instance, image_id, record) if use_view else original_decode(instance, image_id, record)
                    if use_view:
                        if value[1].flags.writeable or value[2].flags.writeable:
                            raise ValueError("An immutable SQL payload view unexpectedly permits mutation")
                        trace["skipped_array_copy_bytes"] += value[1].nbytes+value[2].nbytes
                    return value
                finally:
                    trace["decode_seconds"] += time.perf_counter()-before

            store.SiftFeatureStore.decode = decode
            trace["decode_binding_setup_seconds"] += time.perf_counter()-binding_started
            try:
                yield native_trace
            finally:
                before = time.perf_counter()
                store.SiftFeatureStore.decode = original_decode
                trace["decode_binding_restore_seconds"] += time.perf_counter()-before

    store.SiftFeatureStore.__init__, store.SiftFeatureStore.__getitem__, store.SiftFeatureStore.close = initialize, get, close
    byte.byte_sift_matching = entry.byte_sift_matching = factory
    trace["setup_seconds"] = time.perf_counter()-started
    try:
        yield trace
    finally:
        started = time.perf_counter()
        store.SiftFeatureStore.__init__, store.SiftFeatureStore.__getitem__, store.SiftFeatureStore.close = original_init, original_get, original_close
        byte.byte_sift_matching, entry.byte_sift_matching = old_factory, old_entry
        trace["restore_seconds"] = time.perf_counter()-started
        trace["all_function_bindings_restored"] = (byte.byte_sift_matching is old_factory and entry.byte_sift_matching is old_entry
            and store.SiftFeatureStore.__init__ is original_init and store.SiftFeatureStore.__getitem__ is original_get
            and store.SiftFeatureStore.close is original_close)
