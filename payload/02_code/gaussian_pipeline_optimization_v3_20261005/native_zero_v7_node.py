"""Keep original ownership checks and allow only one extra SIGINT observer."""

import inspect
from pathlib import Path

import native_zero_checked_node as parent


def main():
    source = inspect.getsource(parent.main)
    before = '"check_zero_transfer_contracts.py", "profile_native_zero_fault.py")'
    if source.count(before) != 1:
        raise ValueError('The preserved explicit Linux owner allowlist changed')
    source = source.replace(before, '"check_zero_transfer_contracts.py", "profile_native_zero_fault.py", "profile_native_zero_sigint_v7.py")')
    namespace = dict(parent.main.__globals__)
    namespace['__file__'] = str(Path(__file__).resolve())
    exec(compile(source, '<same_bounded_owner_plus_SIGINT_entry>', 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
