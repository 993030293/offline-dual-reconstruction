"""Publish actual native guard exit before inspecting its final journal."""

import inspect
from pathlib import Path

import native_zero_feature_node as parent
import scene_transform_cost_node as controller


def main():
    source = inspect.getsource(parent.main)
    anchor = '"check_native_zero_transfer_v2.py")'
    if source.count(anchor) != 1:
        raise ValueError("The original explicit entry ownership boundary changed")
    changed = source.replace(anchor, '"check_native_zero_transfer_v2.py", "profile_native_zero_bounded.py", "check_zero_transfer_contracts.py", "profile_native_zero_fault.py")')
    original = controller.main
    guarded = inspect.getsource(original)
    anchor = '    exec(compile(after, str(original), "exec"), namespace)'
    if guarded.count(anchor) != 1:
        raise ValueError("The existing original controller compile boundary changed")
    addition = '    _exit_anchor = \'    guard = read(args.node/"guard.json")\'\n'
    addition += '    if after.count(_exit_anchor) != 1:\n        raise ValueError("The original actual guard exit boundary changed")\n'
    addition += '    after = after.replace(_exit_anchor, \'    atomic_json(args.node/"ACTUAL_GUARD_EXIT.json", {"returncode": record["returncode"], "owner": owner})\\n\' + _exit_anchor)\n'
    exec(compile(guarded.replace(anchor, addition + anchor), "<owned_zero_guard_exit_only>", "exec"), controller.__dict__)
    namespace = dict(parent.main.__globals__)
    exec(compile(changed, str(Path(__file__)), "exec"), namespace)
    try:
        namespace["main"]()
    finally:
        controller.main = original


if __name__ == "__main__":
    main()
