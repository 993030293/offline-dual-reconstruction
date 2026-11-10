"""Check the registered API-byte unit and preserve the prior ownership audit."""

import inspect
import math
from pathlib import Path

import numpy as np

import review_strict_encoding as preserved


def api_census(trace):
    outputs = [row for call in trace["calls"] for row in call["actual_successful_numpy_outputs"]]
    return all(row["bytes"] == math.prod(row["shape"])*np.dtype(row["dtype"]).itemsize for row in outputs) and sum(
        row["bytes"] for row in outputs) == trace["actual_recorded_successful_output_bytes"]


def main():
    source = inspect.getsource(preserved.main)
    edits = [
        ('folder = OUT / "review0"', 'folder = OUT / "review1"'),
        ('path.name == "REVIEW_LEASE.json"', 'path.name == "REVIEW_V2_LEASE.json"'),
        ('"all_actual_six_API_output_censuses_recomputed_and_owned":',
         '"all_registered_successful_six_API_output_bytes_recomputed":'),
        ('    result = {"status":',
         '    checks["all_formal_original_and_native_outputs_own_their_storage"] = all(\n'
         '        row["owns_data"] for i in range(3) for tag in ("b", "c")\n'
         '        for call in read(OUT/f"c0/t{i}/{tag}/RESULT.json")["encoding"]["calls"]\n'
         '        for row in call["actual_successful_numpy_outputs"])\n'
         '    result = {"status":'),
        ('"checks": checks, "original_raw_array_output_bytes": totals,',
         '"checks": checks, "reviewer_sources": [data_identity(Path(__file__)), data_identity(PRESERVED_SOURCE)],\n'
         '        "ownership_of_subclass_views_is_not_the_registered_API_byte_unit": True,\n'
         '        "original_raw_array_output_bytes": totals,'),
        ('shutil.copyfile(Path(__file__), folder / "REVIEW_SOURCE.py")',
         'shutil.copyfile(Path(__file__), folder / "REVIEW_SOURCE.py")\n'
         '    shutil.copyfile(PRESERVED_SOURCE, folder / "PRESERVED_REVIEW_SOURCE.py")')
    ]
    for before, after in edits:
        if source.count(before) != 1:
            raise ValueError("The original strict-encoding root reviewer boundary changed")
        source = source.replace(before, after)
    namespace = dict(preserved.main.__globals__)
    namespace.update(census=api_census, PRESERVED_SOURCE=Path(preserved.__file__), __file__=str(Path(__file__).resolve()))
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
