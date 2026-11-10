"""Restrict the preserved process owner to the registered operand entries."""

import inspect
from pathlib import Path

import native_zero_feature_node as base
import native_zero_checked_node as checked
import scene_transform_cost_node as controller


def main():
    source = inspect.getsource(base.main)
    anchor = '("check_native_feature_zero.py", "profile_native_feature_zero.py",\n               "check_native_zero_transfer_v2.py")'
    if source.count(anchor) != 1:
        raise ValueError("The explicit native entry allowlist changed")
    source = source.replace(anchor, '("check_coordinate_reuse.py", "profile_coordinate_scene.py", "profile_coordinate_failure.py")')
    anchor = '"04_outputs/gaussian_pipeline_optimization_v3_20261005/gzi001"'
    if source.count(anchor) != 1:
        raise ValueError("The registered process output owner changed")
    source = source.replace(anchor, '"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup017/R169"')
    source = source.replace("_ZERO_ADAPTER.py", "_COORDINATE_ADAPTER.py")
    original = controller.main
    guarded = inspect.getsource(original)
    anchor = '    exec(compile(after, str(original), "exec"), namespace)'
    if guarded.count(anchor) != 1:
        raise ValueError("The preserved native actual-exit boundary changed")
    addition = '    _exit_anchor = \'    guard = read(args.node/"guard.json")\'\n'
    addition += '    if after.count(_exit_anchor) != 1:\n        raise ValueError("The actual guard exit boundary changed")\n'
    addition += '    after = after.replace(_exit_anchor, \'    atomic_json(args.node/"ACTUAL_GUARD_EXIT.json", {"returncode": record["returncode"], "owner": owner})\\n\' + _exit_anchor)\n'
    guarded = guarded.replace(anchor, addition + anchor)
    guarded = guarded.replace("size > 12*1024**3", "size > 2*1024**3")
    exec(compile(guarded, str(Path(__file__)), "exec"), controller.__dict__)
    namespace = dict(base.main.__globals__)
    namespace["__file__"] = str(Path(__file__).resolve())
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    try:
        namespace["main"]()
    finally:
        controller.main = original


if __name__ == "__main__":
    main()
