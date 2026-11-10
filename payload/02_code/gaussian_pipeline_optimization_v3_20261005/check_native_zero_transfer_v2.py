"""Run fresh native V2 cases with the frozen payload metric and source guards."""

import inspect
from pathlib import Path

import check_native_feature_zero as parent
from native_feature_zero_v2 import native_feature_zero


def main():
    source = inspect.getsource(parent.main)
    changed = source.replace("actual_profiler_HtoD_bytes", "actual_native_HtoD_bytes")
    changed = changed.replace("profiler_matches_all_native_uploads", "native_API_matches_all_tensor_uploads")
    changed = changed.replace("all_actual_profiler_bytes_match_native_uploads", "all_native_API_bytes_match_tensor_uploads")
    anchor = "        identities = [data_identity(path) for path in sources]"
    if changed.count(anchor) != 1:
        raise ValueError("The original native test source guard boundary changed")
    changed = changed.replace(anchor, "        sources.extend(_meter_sources)\n" + anchor)
    namespace = dict(parent.main.__globals__)
    namespace.update(native_feature_zero=native_feature_zero,
                     _meter_sources=[Path(__file__), parent.CODE / "native_feature_zero_v2.py",
                                     parent.CODE / "native_transfer_meter.py",
                                     parent.CODE / "native_cuda_transfer_counter.c",
                                     parent.OUT / "build/native_transfer_counter.so"])
    exec(compile(changed, str(Path(__file__)), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
