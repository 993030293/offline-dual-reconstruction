"""Bind the existing fault owner to the actual private preparation fault entry."""

import inspect
from pathlib import Path

import open_image_handoff_node as original


def main():
    before = inspect.getsource(original.main)
    anchor = "/profile_open_image_handoff.py"
    if before.count(anchor) != 1:
        raise ValueError("The actual Linux preparation fault admission changed")
    namespace = dict(original.main.__globals__)
    namespace["__file__"] = str(Path(__file__))
    exec(compile(before.replace(anchor, "/profile_faulted_image_handoff.py"), "<owned_native_prepare17_fault_node>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
