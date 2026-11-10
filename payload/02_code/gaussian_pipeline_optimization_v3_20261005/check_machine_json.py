"""Check native JSON data, publication, error and binding contracts with real files."""

import argparse
import json
import math
from pathlib import Path
import sys
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json
from machine_json import compact_atomic_json, exact_json_values, machine_json_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(ROOT / "04_outputs")
    args.output.mkdir(parents=True, exist_ok=False)
    checks = []
    values = [None, True, False, 0, -1, 2**60, "", " spaces stay inside strings ",
              "quote\"backslash\\line\n\t", "中文对象不是特定类别", [], {}, [0], {"a": [1]},
              [0., -0., math.nextafter(0., 1.), math.nextafter(1., 2.), sys.float_info.max],
              {"z": {"y": [[True, None], ["c", 9.3]]}, "a": {"é": "\u0000"}},
              (1, 2.3), {3: "integer key", None: "null key", True: "boolean key"}]
    for index, value in enumerate(values):
        directory = args.output / ("value" + str(index))
        directory.mkdir()
        first, second = directory / "native.json", directory / "compact.json"
        atomic_json(first, value)
        compact_atomic_json(second, value)
        exact_json_values(json.loads(first.read_text(encoding="utf-8")),
                          json.loads(second.read_text(encoding="utf-8")))
        if first.with_suffix(".json.tmp").exists() or second.with_suffix(".json.tmp").exists():
            raise AssertionError("Successful native publication left a temporary file")
        checks.append({"name": "complete_parsed_native_value_" + str(index), "passed": True})
    cycle = []
    cycle.append(cycle)
    failures = [float("nan"), float("inf"), float("-inf"), cycle, {"bad": {1, 2}}, {object(): 1}]
    for index, value in enumerate(failures):
        errors = []
        for name, writer in (("native", atomic_json), ("compact", compact_atomic_json)):
            output = args.output / ("reject" + str(index) + name + ".json")
            output.write_text("original destination", encoding="utf-8")
            try:
                writer(output, value)
            except (ValueError, TypeError) as error:
                errors.append(type(error).__name__)
            else:
                raise AssertionError("A native rejected JSON value was published")
            if output.read_text(encoding="utf-8") != "original destination":
                raise AssertionError("Rejected serialization changed its destination")
        if len(errors) != 2 or errors[0] != errors[1]:
            raise AssertionError("A native JSON rejection type changed")
        checks.append({"name": "native_rejected_value_" + str(index), "passed": True,
                       "original_error_type": errors[0]})
    for operation in ("write_text", "replace"):
        for name, writer in (("native", atomic_json), ("compact", compact_atomic_json)):
            output = args.output / (operation + name + ".json")
            output.write_text("original destination", encoding="utf-8")
            with patch.object(Path, operation, side_effect=OSError("injected publication error")):
                try:
                    writer(output, {"a": 1})
                except OSError:
                    pass
                else:
                    raise AssertionError("An output error was silently accepted")
            if output.read_text(encoding="utf-8") != "original destination":
                raise AssertionError("Failed publication changed its destination")
        checks.append({"name": "native_publication_error_" + operation, "passed": True})
    module = types.SimpleNamespace(atomic_json=atomic_json)
    value = {"a": [1., -0.], "b": "unicode 对象"}
    with machine_json_execution(module, True) as trace:
        module.atomic_json(args.output / "camera_path.json", value)
        module.atomic_json(args.output / "human_report.json", value)
    if (trace["compact_calls"], trace["native_calls"]) != (1, 1) or not trace["original_binding_restored"]:
        raise AssertionError("The explicit machine route or native restoration changed")
    expected = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if (args.output / "human_report.json").read_text(encoding="utf-8") != expected:
        raise AssertionError("A non-machine path changed its original pretty representation")
    checks.append({"name": "explicit_machine_only_and_original_nonmachine_route", "passed": True})
    try:
        with machine_json_execution(module, True) as failed_trace:
            raise OSError("injected caller failure")
    except OSError:
        pass
    if not failed_trace["original_binding_restored"] or module.atomic_json is not atomic_json:
        raise AssertionError("An exceptional caller did not restore the native binding")
    checks.append({"name": "exceptional_caller_binding_restored", "passed": True})
    result = {"status": "all_real_JSON_data_publication_oracles_passed", "checks": checks,
              "passed_checks": len(checks), "native_rejection_message_text_identity_claimed": False,
              "synthetic_fixtures_are_not_new_object_geometry_or_training": True, "counted": False}
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "passed_checks": len(checks)}), flush=True)


if __name__ == "__main__":
    main()
