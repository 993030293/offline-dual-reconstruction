"""Keep the native algorithm and tighten the registered Linux resource guard."""

import inspect
from pathlib import Path

import profile_native_feature_zero as parent
from mapped_image_linux_guard import LinuxRSSGuard


class ZeroSceneGuard(LinuxRSSGuard):
    def __init__(self, output, publish):
        super().__init__(output, publish, maximum_seconds=300, maximum_bytes=16 * 1024**3)


def main():
    source = inspect.getsource(parent.main)
    anchor = '    native_activate = namespace["activate_reference"]'
    changed = source
    if changed.count(anchor) != 1:
        raise ValueError("The original complete Linux bound ownership boundary changed")
    changed = changed.replace(anchor, anchor + '\n    namespace["LinuxRSSGuard"] = _registered_guard')
    anchor = '    identities = [data_identity(path) for path in paths]'
    if changed.count(anchor) != 1:
        raise ValueError("The complete private bound source guard changed")
    changed = changed.replace(anchor, '    paths.extend(_bounds_sources)\n' + anchor)
    namespace = dict(parent.main.__globals__)
    namespace.update(_registered_guard=ZeroSceneGuard,
                     _bounds_sources=[Path(__file__), parent.CODE / "native_zero_checked_node.py"])
    exec(compile(changed, str(Path(__file__)), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
