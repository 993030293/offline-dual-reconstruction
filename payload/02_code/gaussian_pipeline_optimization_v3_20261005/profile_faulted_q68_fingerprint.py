"""Inject a native identity17 error after the actual complete Scene consumption."""

import difflib
import functools
import inspect
from pathlib import Path
import sys
import textwrap

import profile_q68_fused_fingerprint as original
from runtime import atomic_json, data_identity


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    enabled = sys.argv[sys.argv.index("--identity-enabled") + 1] == "1"
    state = {"active": False, "prefix": [], "target": None, "paths": []}

    def install(helper):
        previous = helper.OrderedIdentities.__next__

        def observe(owner):
            result = previous(owner)
            if state["active"]:
                state["prefix"].append(result)
            return result

        helper.OrderedIdentities.__next__ = observe

    def fault_native(paths, native, label):
        if label != "inputs_after":
            return native
        if len(paths) != 1036:
            raise ValueError("The actual complete failure denominator changed")
        state.update(active=True, target=str(Path(paths[16])), paths=[str(Path(p)) for p in paths])

        @functools.wraps(native)
        def failing(path):
            if str(Path(path)) == state["target"]:
                raise OSError("Forced original complete input identity failure at17")
            value = native(path)
            if not enabled:
                state["prefix"].append(value)
            return value

        return failing

    before = textwrap.dedent(inspect.getsource(original.main))
    after = before
    replacements = (
        ('    spec.loader.exec_module(helper)', '    spec.loader.exec_module(helper)\n    _fault_install(helper)'),
        ('        row = {"phase": label,', '        native = _fault_native(paths, native, label)\n        row = {"phase": label,'),
    )
    for old, new in replacements:
        if after.count(old) != 1:
            raise ValueError("The original composed identity consumer boundary changed")
        after = after.replace(old, new)
    archive = output.with_name(output.name + "_fault_entry")
    archive.mkdir(exist_ok=False)
    (archive / "BASE_COMPLETE_ENTRY.py").write_text(before, encoding="utf-8")
    (archive / "EXECUTED_COMPLETE_ENTRY.py").write_text(after, encoding="utf-8")
    (archive / "FAULT_ENTRY.diff").write_text("".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="original", tofile="same_native_identity17_fault")), encoding="utf-8")
    (archive / "FAULT_ADAPTER_SOURCE.py").write_bytes(Path(__file__).read_bytes())
    namespace = dict(original.main.__globals__)
    namespace.update(_fault_install=install, _fault_native=fault_native)
    exec(compile(after, "<actual_complete_Q68_identity17_fault>", "exec"), namespace)
    try:
        namespace["main"]()
    except BaseException as error:
        if len(state["prefix"]) != 16 or not state["active"]:
            raise AssertionError("The actual original FIFO failure did not follow sixteen complete identities") from error
        atomic_json(output / "FAULT_PREFIX.json", {"status": "actual_complete_Scene_then_native_identity17_prefix16",
            "attempted_identity": 17, "first16_complete_identities": state["prefix"], "paths": state["paths"],
            "error": repr(error), "source": data_identity(Path(__file__)), "new_optimization_count": 0})
        raise
    raise AssertionError("The explicitly faulted complete entry unexpectedly succeeded")


if __name__ == "__main__":
    main()
