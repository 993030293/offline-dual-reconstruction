"""Measure a complete current evaluator with one optional prediction-tail graph."""

import argparse
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]
import evaluate_interior_ssim as entry
from evaluate_interior_conv_work import composed_main, replace_once, finish_coupled_view
from runtime import atomic_json, data_identity


def measurement_source():
    source = composed_main()
    source = replace_once(source, 'choices=("native", "interior")', 'choices=("incumbent", "graph")')
    source = replace_once(source,
        'inner = ReferenceSSIM(loss_utils.create_window) if args.mode == "native" else InteriorSSIM(loss_utils.create_window)',
        'inner = GraphInteriorSSIM(loss_utils.create_window) if args.mode == "graph" else InteriorSSIM(loss_utils.create_window)')
    source = replace_once(source, 'if args.mode == "interior":', 'if args.mode in ("incumbent", "graph"):')
    source = replace_once(source, 'finish_coupled_view(inner, finished_views)', 'finish_graph_view(inner, finished_views)')
    source = replace_once(source, '"native_conv_calls_before_return": len(_conv_trace["rows"]), ', '')
    source = replace_once(source, 'CODE / "interior_ssim_reuse.py", CODE / "evaluate_interior_ssim.py",',
                          'CODE / "graph_reference_ssim.py", CODE / "evaluate_interior_conv_work.py", CODE / "interior_ssim_reuse.py", CODE / "evaluate_interior_ssim.py",')
    if source.count("inner.clear()") != 2:
        raise ValueError("The original two complete-cache closure boundaries changed")
    return source.replace("inner.clear()", "_clear_inner(inner)")


def run():
    pair = Path(sys.argv[sys.argv.index("--pair") + 1])
    pair.resolve().relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/eg001")
    source = measurement_source()
    path = pair.parent / "ACTUAL_MEASUREMENT_SOURCE.py"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(source)
    registry, close_rows = [], []
    dependency = {"all_generic_tested_Linux_and_workspace_sources_exact": False, "seconds": 0.0}

    def clear(inner):
        before = time.perf_counter()
        try:
            inner.clear()
        finally:
            close_rows.append(time.perf_counter() - before)

    def finish(inner, trace):
        finish_coupled_view(inner, trace)
        if hasattr(inner, "finish_reference"):
            inner.finish_reference()

    original_activate = entry.activate_reference
    error = None

    def activate(runtime):
        metadata = original_activate(runtime)
        before = time.perf_counter()
        import json
        generic = json.loads((ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/eg001/g0/RESULT.json").read_text())
        for row in generic["sources"]:
            if data_identity(Path(row["path"]))["sha256"] != row["sha256"]:
                raise ValueError("An actual generic-tested Linux or workspace source changed")
        dependency.update(all_generic_tested_Linux_and_workspace_sources_exact=True,
                          seconds=time.perf_counter() - before)
        from graph_reference_ssim import GraphInteriorSSIM
        namespace["GraphInteriorSSIM"] = GraphInteriorSSIM
        return metadata

    namespace = dict(vars(entry))
    namespace.update(__file__=str(Path(__file__)), STARTED=STARTED,
        activate_reference=activate, compose_view_major=__import__("view_major_evaluation").compose_view_major,
        finish_coupled_view=finish_coupled_view, finish_graph_view=finish,
        _register_inner=registry.append, _fault_before_SSIM=lambda ordinal: None, _clear_inner=clear)
    try:
        exec(compile(source, "<complete-current-Q65-evaluation-graph>", "exec"), namespace)
        namespace["main"]()
    except BaseException as failure:
        error = failure
    finally:
        for inner in registry:
            clear(inner)
    checks = {"all_owned_reference_caches_closed": bool(registry) and all(
                  inner.receipt()["resident_accounted_bytes"] == 0
                  and inner.receipt().get("retained_last_map_bytes", 0) == 0 for inner in registry),
              "all_owned_graphs_closed": all(
                  inner.receipt().get("owned_graph", {}).get("closed", True)
                  and inner.receipt().get("owned_graph", {}).get("live_graph_entries", 0) == 0 for inner in registry),
              "original_runtime_activation_binding_unchanged": entry.activate_reference is original_activate}
    checks.update(all_actual_generic_tested_native_dependencies_exact=
                  dependency["all_generic_tested_Linux_and_workspace_sources_exact"])
    atomic_json(pair.parent / "GRAPH_RESULT.json", {"error": None if error is None else {
        "type": type(error).__name__, "message": str(error)}, "checks": checks,
        "close_seconds_all_actual_cache_calls": sum(close_rows), "close_calls": len(close_rows),
        "actual_native_dependency_validation": dependency,
        "final_owned_states": [inner.receipt() for inner in registry],
        "measurement_source": data_identity(path), "complete_child_seconds": time.perf_counter() - STARTED,
        "new_optimization_count": 0})
    if error is not None:
        raise error
    if not all(checks.values()):
        raise AssertionError("A full graph lifetime guard failed")


if __name__ == "__main__":
    run()
