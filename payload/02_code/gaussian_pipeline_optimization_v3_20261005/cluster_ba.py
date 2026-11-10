"""Select visibility cluster preconditioning while keeping the native BA objective."""

from contextlib import contextmanager

import pycolmap

from direct_schur import COMMON, ROOT, V1, CandidateGuardFailure, differences, plain
import common
from runtime import atomic_json


def select_cluster(options):
    before = plain(options.todict())
    options.ceres.auto_select_solver_type = False
    solver = options.ceres.solver_options
    solver.linear_solver_type = type(solver.linear_solver_type).ITERATIVE_SCHUR
    solver.preconditioner_type = type(solver.preconditioner_type).CLUSTER_JACOBI
    after = plain(options.todict())
    diff = differences(before, after)
    allowed = {"ceres.auto_select_solver_type", "ceres.solver_options.linear_solver_type",
               "ceres.solver_options.preconditioner_type"}
    if set(diff) != allowed or not options.check():
        raise CandidateGuardFailure("Cluster selection did not change exactly the three registered fields")
    return {"before": before, "after": after, "differences": diff}


def actual_backend(solution):
    native = solution["native_summary"]["ceres_summary"]
    return {
        "usable": solution["usable"],
        "solver_given_iterative_schur": native["linear_solver_type_given"] == "ITERATIVE_SCHUR",
        "solver_used_iterative_schur": native["linear_solver_type_used"] == "ITERATIVE_SCHUR",
        "preconditioner_given_cluster_jacobi": native["preconditioner_type_given"] == "CLUSTER_JACOBI",
        "preconditioner_used_cluster_jacobi": native["preconditioner_type_used"] == "CLUSTER_JACOBI",
        "original_visibility_canonical_views": native["visibility_clustering_type"] == "CANONICAL_VIEWS",
        "original_sparse_backend_suite_sparse": native["sparse_linear_algebra_library_type"] == "SUITE_SPARSE",
        "actual_linear_solves_positive": native["num_linear_solves"] > 0,
        "actual_linear_solve_time_positive": native["linear_solver_time_in_seconds"] > 0,
    }


@contextmanager
def metric_binding(mode, output):
    """Restore private native bindings after either success or original rejection."""
    if mode not in ("auto", "cluster"):
        raise ValueError("An explicitly registered solver route is required")
    saved = common.ba_options, common.solver_receipt
    option_records, solutions = [], []

    def options(iterations):
        value = saved[0](iterations)
        record = select_cluster(value) if mode == "cluster" else {"original": plain(value.todict())}
        option_records.append(record)
        atomic_json(output / "OPTION_SELECTION.json", option_records)
        return value

    def receipt(summary):
        value = plain(saved[1](summary))
        if mode == "cluster":
            checks = actual_backend(value)
        else:
            native = value["native_summary"]["ceres_summary"]
            checks = {"original_solver_usable": value["usable"],
                      "original_iterative_schur": native["linear_solver_type_used"] == "ITERATIVE_SCHUR",
                      "original_schur_jacobi": native["preconditioner_type_used"] == "SCHUR_JACOBI"}
        solutions.append({"solution": value, "backend_checks": checks})
        atomic_json(output / "ACTUAL_SOLVES.json", solutions)
        if not all(checks.values()):
            raise CandidateGuardFailure("Actual native solver or preconditioner guard failed; no fallback")
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

