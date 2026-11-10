"""Reject actual truncated and changed PLY files through the official reader."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pycolmap

import check_sparse_ply_oracle as fixture
from runtime import atomic_json, data_identity
from sparse_ply_oracle import sparse_ply_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    native = fixture.native
    original, saved_module = native.write_initial_ply, native.pycolmap
    model = fixture.make_model(7, .01)
    checks, results = {}, []
    for mode in ("truncated_header", "truncated_body", "changed_color"):
        class Reader:
            def __init__(self):
                self.actual = pycolmap.Reconstruction()

            def import_PLY(self, path):
                content = Path(path).read_bytes()
                if mode == "truncated_header":
                    content = content[:4]
                elif mode == "truncated_body":
                    content = content[:-1]
                else:
                    payload = bytearray(content)
                    payload[-1] ^= 1
                    content = bytes(payload)
                Path(path).write_bytes(content)
                self.actual.import_PLY(path)

            @property
            def points3D(self):
                return self.actual.points3D

        values, contents = [], []
        try:
            native.pycolmap = SimpleNamespace(Reconstruction=Reader)
            for enabled in (False, True):
                path = args.output/f"{mode}_{int(enabled)}.ply"
                with sparse_ply_execution(native, enabled):
                    values.append(fixture.outcome(native.write_initial_ply, model, path))
                contents.append(path.read_bytes())
                if native.write_initial_ply is not original:
                    raise AssertionError("Sparse export did not restore its binding")
        finally:
            native.pycolmap = saved_module
        checks[mode] = values[0] == values[1] and values[0]["status"] == "failed" and contents[0] == contents[1]
        results.append({"case": mode, "native": values[0], "columns": values[1],
                        "actual_official_importer_called": True,
                        "changed_bytes_preserved_after_rejection": True})
    result = {"status": "all_actual_official_bad_PLY_rejections_passed" if all(checks.values()) else "failed",
              "checks": checks, "rows": results, "counted": False,
              "source": [data_identity(Path(__file__)), data_identity(Path(fixture.__file__)),
                         data_identity(Path(__file__).with_name("sparse_ply_oracle.py")),
                         data_identity(Path(native.__file__))]}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": checks}))
    if not all(checks.values()):
        raise AssertionError("The original official bad PLY rejection changed")


if __name__ == "__main__":
    main()
