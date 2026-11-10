"""Keep the complete RAW128 consumer and change only its candidate priority."""

import difflib
import json
from pathlib import Path
import sys

import profile_vector_gray_stats as parent
from lazy_corner_execution import lazy_corner_execution

ROOT = parent.ROOT
CODE = Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/lco001"


def main():
    index = sys.argv.index("--order-library")
    library = Path(sys.argv[index + 1]).resolve(strict=True)
    sys.argv = sys.argv[:index] + sys.argv[index + 2:]
    library.relative_to(OUT / "g0/build")
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    output.relative_to(OUT)
    if output.exists():
        raise FileExistsError("A complete corner ordering screen needs a fresh directory")
    generic = parent.reference.load(OUT / "g0/RESULT.json")
    if (not all(generic["checks"].values()) or generic["component_raw_reduction_not_whole_acceptance"] < .05
            or parent.reference.data_identity(library) != generic["library"]):
        raise ValueError("The current exact corner ordering component is not admitted")
    for row in generic["sources"]:
        if parent.reference.data_identity(Path(row["path"])) != row:
            raise ValueError("A checked corner priority source changed before the pilot")
    original = Path(parent.__file__)
    before = original.read_text(encoding="utf-8")
    before = before[:before.rfind('if __name__ == "__main__":')]
    replacements = (
        ('output.relative_to(OUT / "vector_gray_stats_001")', 'output.relative_to(OUT / "lco001")'),
        ('    sources = [reference.data_identity(path) for path in sources_paths]',
         '    sources_paths += _owned_paths\n    sources = [reference.data_identity(path) for path in sources_paths]'),
        ('with planar_corner_execution(common) as corner_trace:',
         'with lazy_corner_execution(_lazy_library, mode == "vector") as lazy_trace, planar_corner_execution(common) as corner_trace:'),
        ('with (vector_gray_stats_execution(args.library) if mode == "vector" else nullcontext()) as vector_trace:',
         'with vector_gray_stats_execution(args.library) as vector_trace:'),
        ('if mode == "vector" and not (vector_trace["original_factory_restored"]',
         'if not (vector_trace["original_factory_restored"]'),
        ('"corner_execution": corner_trace, "vector_execution": vector_trace, "histogram": histogram_trace,',
         '"corner_execution": corner_trace, "vector_execution": vector_trace, "histogram": histogram_trace, "lazy_order_execution": lazy_trace,'),
    )
    after = before
    for anchor, replacement in replacements:
        if after.count(anchor) != 1:
            raise ValueError("An original full RAW128 priority consumer boundary changed: " + anchor)
        after = after.replace(anchor, replacement)
    owned_paths = [Path(__file__), original, CODE / "lazy_corner_execution.py", CODE / "lazy_corner_order.py",
        CODE / "native_lazy_corner_order.cpp", CODE / "check_lazy_corner_order.py", library]
    identities = [parent.reference.data_identity(path) for path in owned_paths]
    namespace = dict(vars(parent))
    namespace.update(__file__=str(Path(__file__)), __name__="original_complete_RAW128_same_vector_lazy_priority",
        _owned_paths=owned_paths, _lazy_library=library, lazy_corner_execution=lazy_corner_execution)
    exec(compile(after, "<original_full_RAW128_lazy_priority_only>", "exec"), namespace)
    try:
        namespace["main"]()
    finally:
        if output.is_dir():
            (output / "EXECUTED_RAW128_PARENT.py").write_text(after, encoding="utf-8")
            (output / "RAW128_PARENT.diff").write_text("".join(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="original_vector_statistics_pilot", tofile="same_statistics_lazy_priority")), encoding="utf-8")
    result = parent.reference.load(output / "RESULT.json")
    traces = [pair["modes"][mode]["lazy_order_execution"] for pair in result["pairs"] for mode in pair["order"]]
    checks = {
        "own_scientific_sources_and_checked_library_unchanged": identities == [parent.reference.data_identity(path) for path in owned_paths],
        "three_complete128_pairs_original_outputs_and_cache_bytes_exact": len(result["pairs"]) == 3 and all(all(pair["checks"].values()) for pair in result["pairs"]),
        "all_private_factories_restored": len(traces) == 6 and all(trace["original_factory_restored"] for trace in traces),
        "only_candidate_priority_enabled_on_three_sides": sum(trace["enabled"] for trace in traces) == 3,
        "all_original128_corner_calls_and_owned_planes_DDL_drained": all(
            len(trace["workspaces"]) == 1 and trace["workspaces"][0]["native_calls"] == 128
            and trace["workspaces"][0]["closed"] and trace["workspaces"][0]["priority_DLL_unloaded"]
            and not trace["workspaces"][0]["active_calls"] and not trace["workspaces"][0]["retained_plane_bytes"]
            if trace["enabled"] else not trace["workspaces"] for trace in traces),
        "current_vector_statistics_enabled_on_all_six_sides": all(pair["modes"][mode]["vector_execution"]["engine"]["native_calls"] == 128 for pair in result["pairs"] for mode in pair["order"]),
    }
    parent.reference.atomic_json(output / "PILOT_ORDER_ADAPTER.json", {"checks": checks, "sources": identities,
        "legacy_current_vector_names_mean_native_priority_lazy_priority": True,
        "both_sides_original_current_vector_statistics": True, "new_optimization_count": 0})
    if not all(checks.values()):
        raise AssertionError("An actual complete corner ordering screen changed an original owner")


if __name__ == "__main__":
    main()
