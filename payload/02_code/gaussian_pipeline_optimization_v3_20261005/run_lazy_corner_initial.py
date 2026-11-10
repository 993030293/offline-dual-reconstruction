"""Keep the complete accepted RAW entry and replace only candidate ordering."""

import difflib
from contextlib import contextmanager
from pathlib import Path
import sys

import run_vector_gray_stats_input as parent
from lazy_corner_execution import lazy_corner_execution

ROOT = parent.ROOT
CODE = Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/lco001"


def main():
    index = sys.argv.index("--lazy-enabled")
    value = sys.argv[index + 1]
    if value not in ("0", "1"):
        raise ValueError("The full priority route must be explicit")
    sys.argv = sys.argv[:index] + sys.argv[index + 2:]
    index = sys.argv.index("--order-library")
    library = Path(sys.argv[index + 1]).resolve(strict=True)
    sys.argv = sys.argv[:index] + sys.argv[index + 2:]
    library.relative_to(OUT / "g0/build")
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    output.relative_to(OUT)
    if output.exists() or sys.argv[sys.argv.index("--mode") + 1] != "vector":
        raise ValueError("Both fresh complete sides require the original current vector statistics")
    generic = parent.accepted.atomic_json
    identity = parent.accepted.data_identity
    from review_scene_batch_negative import read
    oracle = read(OUT / "g0/RESULT.json")
    pilot = read(OUT / "p0/RESULT.json")
    if (not all(oracle["checks"].values()) or oracle["library"] != identity(library)
            or pilot["status"] != "complete128_screen_positive_complete7338_pending"
            or not all(read(OUT / "p0/PILOT_ORDER_ADAPTER.json")["checks"].values())):
        raise ValueError("The actually executed ordering prerequisites did not pass")
    for row in oracle["sources"]:
        if identity(Path(row["path"])) != row:
            raise ValueError("A checked priority source changed before a full consumer")
    original = Path(parent.__file__)
    before = original.read_text(encoding="utf-8")
    before = before[:before.rfind('if __name__ == "__main__":')]
    anchor = "with planar_corner_execution(args.corner_library) as corner_trace:"
    if before.count(anchor) != 1:
        raise ValueError("The original complete corner owner boundary changed")
    after = before.replace(anchor, "with lazy_corner_execution(_order_library, _lazy_enabled) as order_trace, planar_corner_execution(args.corner_library) as corner_trace:")
    traces = []

    @contextmanager
    def actual_order_owner(*arguments):
        with lazy_corner_execution(*arguments) as recorded:
            traces.append(recorded)
            yield recorded

    namespace = dict(vars(parent))
    namespace.update(__file__=str(original), __name__="original_complete_RAW_current_statistics_lazy_order",
        _order_library=library, _lazy_enabled=value == "1", lazy_corner_execution=actual_order_owner)
    owned_paths = [Path(__file__), CODE / "lazy_corner_execution.py", CODE / "lazy_corner_order.py",
        CODE / "native_lazy_corner_order.cpp", original, library]
    identities = [identity(path) for path in owned_paths]
    exec(compile(after, "<original_complete_RAW_lazy_priority_only>", "exec"), namespace)
    try:
        namespace["main"]()
    finally:
        if output.is_dir():
            snapshot = output / "priority_source_snapshot"
            snapshot.mkdir(exist_ok=False)
            for index, path in enumerate(owned_paths):
                if path.suffix != ".dll":
                    (snapshot / (f"{index:03d}_" + path.name)).write_bytes(path.read_bytes())
            (snapshot / "EXECUTED_FULL_INPUT_PARENT.py").write_text(after, encoding="utf-8")
            (snapshot / "FULL_INPUT_PARENT.diff").write_text("".join(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="original_current_vector_input", tofile="same_vector_lazy_priority")), encoding="utf-8")
    if not traces:
        raise AssertionError("The actual full candidate priority owner was not recorded")
    trace = traces[0]
    checks = {
        "own_full_entry_sources_and_checked_library_unchanged": identities == [identity(path) for path in owned_paths],
        "original_priority_factory_restored": trace["original_factory_restored"],
        "one_complete_priority_owner_and_all7338_calls_closed": len(traces) == 1 and (
            len(trace["workspaces"]) == 1 and trace["workspaces"][0]["native_calls"] == 7338
            and trace["workspaces"][0]["DLL_unloaded"] and trace["workspaces"][0]["priority_DLL_unloaded"]
            and not trace["workspaces"][0]["active_calls"] and not trace["workspaces"][0]["retained_plane_bytes"]
            if value == "1" else not trace["workspaces"]),
        "current_vector_statistics_route_preserved": read(output / "VECTOR_INPUT.json")["trace"]["engine"]["native_calls"] == 7338,
    }
    generic(output / "PRIORITY_INPUT.json", {"checks": checks, "trace": trace, "sources": identities,
        "new_optimization_count": 0})
    if not all(checks.values()):
        raise AssertionError("The actual complete priority entry changed an original owner")


if __name__ == "__main__":
    main()
