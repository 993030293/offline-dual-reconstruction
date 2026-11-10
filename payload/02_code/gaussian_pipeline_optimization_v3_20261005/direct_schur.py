"""Select native CPU sparse Schur without changing the metric BA objective."""

from contextlib import contextmanager
from pathlib import Path
import sys

import pycolmap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
COMMON = ROOT / "02_code/box2_lio_guided_sfm_v1_20261002"
sys.path[:0] = [str(V1), str(COMMON)]
import common
from runtime import atomic_json


class CandidateGuardFailure(RuntimeError):
    """A real candidate contract failure, distinct from oracle construction."""


def plain(value):
    return common.serializable(value)


def differences(a, b, prefix=""):
    result = {}
    for key in sorted(set(a) | set(b)):
        path = prefix + key
        if isinstance(a.get(key), dict) and isinstance(b.get(key), dict):
            result.update(differences(a[key], b[key], path + "."))
        elif a.get(key) != b.get(key):
            result[path] = [a.get(key), b.get(key)]
    return result


def select_direct(options):
    before = plain(options.todict())
    options.ceres.auto_select_solver_type = False
    enum = type(options.ceres.solver_options.linear_solver_type)
    options.ceres.solver_options.linear_solver_type = enum.SPARSE_SCHUR
    after = plain(options.todict())
    diff = differences(before, after)
    allowed = {"ceres.auto_select_solver_type", "ceres.solver_options.linear_solver_type"}
    if set(diff) != allowed or not options.check():
        raise CandidateGuardFailure("Direct selection did not change exactly the two registered fields")
    return {"before": before, "after": after, "differences": diff}


def actual_backend(solution):
    native = solution["native_summary"]["ceres_summary"]
    checks = {
        "usable": solution["usable"],
        "solver_given_sparse_schur": native["linear_solver_type_given"] == "SPARSE_SCHUR",
        "solver_used_sparse_schur": native["linear_solver_type_used"] == "SPARSE_SCHUR",
        "sparse_backend_suite_sparse": native["sparse_linear_algebra_library_type"] == "SUITE_SPARSE",
        "actual_linear_solves_positive": native["num_linear_solves"] > 0,
        "actual_linear_solve_time_positive": native["linear_solver_time_in_seconds"] > 0,
    }
    return checks


@contextmanager
def metric_binding(mode, output):
    """Restore native bindings even when the first original guard rejects."""
    saved = common.ba_options, common.solver_receipt
    option_records, solutions = [], []

    def options(iterations):
        value = saved[0](iterations)
        record = select_direct(value) if mode == "direct" else {"original": plain(value.todict())}
        option_records.append(record)
        atomic_json(output / "OPTION_SELECTION.json", option_records)
        return value

    def receipt(summary):
        value = plain(saved[1](summary))
        checks = actual_backend(value) if mode == "direct" else {"original_solver_usable": value["usable"]}
        solutions.append({"solution": value, "backend_checks": checks})
        atomic_json(output / "ACTUAL_SOLVES.json", solutions)
        if not all(checks.values()):
            raise CandidateGuardFailure("Actual native solver/backend/usable guard failed; no fallback")
        return value

    common.ba_options, common.solver_receipt = options, receipt
    try:
        yield option_records, solutions
    finally:
        common.ba_options, common.solver_receipt = saved
        restored = common.ba_options is saved[0] and common.solver_receipt is saved[1]
        atomic_json(output / "BINDING_RESTORE.json", {"restored": restored})
        if not restored:
            raise AssertionError("Original common metric bindings were not restored")
