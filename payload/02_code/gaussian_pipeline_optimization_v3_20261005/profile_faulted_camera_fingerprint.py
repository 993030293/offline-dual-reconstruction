"""Inject a real image-consumer error without changing the measured implementation."""

import difflib
import inspect
from pathlib import Path
import sys
import textwrap

import profile_fused_camera_fingerprint as original
from runtime import atomic_json, data_identity


def main():
    before = textwrap.dedent(inspect.getsource(original.main))
    anchor = '    exec(compile(after, str(original), "exec"), namespace)'
    if before.count(anchor) != 1:
        raise ValueError("The original generated full consumer binding changed")

    def fault_source(source, enabled, output, namespace):
        rows, current = [], None
        attempts = 0

        def observe(field, value):
            nonlocal attempts, current
            if field == "before_image":
                attempts += 1
                if attempts == 17:
                    if len(rows) != 16 or any(len(row) != 6 for row in rows):
                        raise AssertionError("A true image-consumer error did not follow 16 complete rows")
                    atomic_json(output / "FAULT_PREFIX.json", {"status": "actual_complete_Scene_image17_error_prefix16",
                        "attempted_image": attempts, "first16_complete_camera_fields": rows,
                        "source": data_identity(Path(__file__)), "new_optimization_count": 0})
                    raise OSError("Forced original image consumption failure at17")
                current = {}
                rows.append(current)
            else:
                current[field] = value

        boundary = "        def rng():"
        if source.count(boundary) != 1:
            raise ValueError("The exact tensor/RNG boundary changed")
        wrapper = '''        _original_field_tensor = tensor
        def tensor(value, _field=None):
            if _field == "original_image":
                _fault_observe("before_image", None)
            _result = _original_field_tensor(value, _field) if _fault_fused else _original_field_tensor(value)
            if _field is not None:
                _fault_observe(_field, _result)
            return _result

'''
        after = source.replace(boundary, wrapper + boundary)
        call = "**{name: tensor(getattr(camera, name)) for name in field_names}"
        if not enabled:
            if after.count(call) != 1:
                raise ValueError("The original field labels were not admitted once")
            after = after.replace(call, call.replace("getattr(camera, name))", "getattr(camera, name), name)"))
        snapshot = output.with_name(output.name + "_fingerprint_sources")
        (snapshot / "BASE_PROFILE_BEFORE_FAULT.py").write_text(source, encoding="utf-8")
        (snapshot / "EXECUTED_FULL_SCENE_PROFILE.py").write_text(after, encoding="utf-8")
        (snapshot / "FAULT_PROFILE.diff").write_text("".join(difflib.unified_diff(source.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile="original_complete_consumer", tofile="same_image17_fault")), encoding="utf-8")
        (snapshot / "FAULT_ADAPTER_SOURCE.py").write_bytes(Path(__file__).read_bytes())
        namespace.update(_fault_observe=observe, _fault_fused=enabled)
        return after

    after = before.replace(anchor, "    after = _fault_source(after, enabled, output, namespace)\n" + anchor)
    namespace = dict(original.main.__globals__)
    namespace["_fault_source"] = fault_source
    exec(compile(after, "<original_full_Scene_image17_fault>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
