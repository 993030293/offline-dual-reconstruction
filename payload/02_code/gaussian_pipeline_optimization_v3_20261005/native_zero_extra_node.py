"""Extend only the explicit owned entry allowlist for contract diagnostics."""

import inspect
from pathlib import Path

import native_zero_feature_node as parent


def main():
    source = inspect.getsource(parent.main)
    anchor = '"check_native_zero_transfer_v2.py")'
    if source.count(anchor) != 1:
        raise ValueError("The existing explicit extra-entry ownership boundary changed")
    changed = source.replace(anchor, '"check_native_zero_transfer_v2.py", "check_zero_transfer_contracts.py", "profile_native_zero_fault.py")')
    namespace = dict(parent.main.__globals__)
    exec(compile(changed, str(Path(__file__)), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
