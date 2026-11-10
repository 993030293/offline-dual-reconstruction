"""Run fresh c1 pairs under tightened bounds with the exact old mechanism."""

import inspect
from pathlib import Path

import run_native_zero_feature_campaign as parent


def main():
    command_source = inspect.getsource(parent.command)
    if command_source.count('"native_zero_feature_node.py"') != 1 or command_source.count('/profile_native_feature_zero.py') != 1:
        raise ValueError("The existing explicit complete command changed")
    command_source = command_source.replace('"native_zero_feature_node.py"', '"native_zero_checked_node.py"')
    command_source = command_source.replace('/profile_native_feature_zero.py', '/profile_native_zero_bounded.py')
    source = inspect.getsource(parent.main)
    if source.count('output = OUT / "c0"') != 1:
        raise ValueError("The fresh complete campaign scope changed")
    changed = source.replace('output = OUT / "c0"', 'output = OUT / "c2"')
    anchor = '    sources = [data_identity(path) for path in paths]'
    if changed.count(anchor) != 1:
        raise ValueError("The complete campaign source guard changed")
    changed = changed.replace(anchor, '    paths.extend(_checked_sources)\n' + anchor)
    namespace = dict(parent.main.__globals__)
    namespace['_checked_sources'] = [Path(__file__), parent.CODE / "profile_native_zero_bounded.py",
                                     parent.CODE / "native_zero_checked_node.py"]
    exec(compile(command_source, str(Path(__file__)), "exec"), namespace)
    exec(compile(changed, str(Path(__file__)), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
