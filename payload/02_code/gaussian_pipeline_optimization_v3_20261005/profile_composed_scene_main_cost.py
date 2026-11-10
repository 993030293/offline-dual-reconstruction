"""Extend the exact full preparation diagnostic with original main-call costs."""

from pathlib import Path
import difflib
import shutil
import sys

from profile_composed_scene_cost import ROOT, CODE
from runtime import atomic_json, data_identity

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sclm002"


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    output.relative_to(OUT.resolve())
    if output.exists():
        raise FileExistsError("The full main-call diagnostic requires a fresh output")
    original = CODE / "profile_composed_scene_cost.py"
    before = original.read_text(encoding="utf-8")
    after = before
    replacements = (
        ("gaussian_pipeline_optimization_v3_20261005/sclc001", "gaussian_pipeline_optimization_v3_20261005/sclm002"),
        ("from timed_camera_preparation import timed_camera_preparation'", "from timed_camera_preparation import timed_camera_preparation\\nfrom timed_camera_main_calls import timed_camera_main_calls'"),
        ("mapping as store, timed_camera_preparation(args.output) as preparation_cost, prefetch_camera_images(4, 8) as prefetch'",
         "mapping as store, timed_camera_preparation(args.output) as preparation_cost, timed_camera_main_calls(args.output), prefetch_camera_images(4, 8) as prefetch'"),
    )
    for old, new in replacements:
        expected = 2 if old.startswith("gaussian_pipeline") else 1
        if after.count(old) != expected:
            raise ValueError("A preserved preparation diagnostic source boundary changed")
        after = after.replace(old, new)
    paths = [Path(__file__), original, CODE / "timed_camera_main_calls.py", CODE / "composed_scene_main_cost_node.py", CODE / "run_composed_scene_main_cost.py"]
    identities = [data_identity(path) for path in paths]
    archive = output.with_name(output.name + "_main_call_sources")
    archive.mkdir(parents=True, exist_ok=False)
    for path in paths:
        shutil.copyfile(path, archive / path.name)
    (archive / "EXECUTED_PREPARATION_PARENT.py").write_text(after, encoding="utf-8")
    (archive / "PREPARATION_PARENT.diff").write_text("".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="accepted_diagnostic", tofile="additional_original_calls_timed")), encoding="utf-8")
    namespace = {"__file__": str(original), "__name__": "owned_full_original_main_call_cost"}
    exec(compile(after, str(original), "exec"), namespace)
    namespace["main"]()
    if identities != [data_identity(path) for path in paths]:
        raise AssertionError("An actual main-call diagnostic source changed")
    atomic_json(output / "MAIN_ADAPTER_SOURCES.json", {"sources": identities, "actual_sources_unchanged": True, "new_optimization_count": 0})


if __name__ == "__main__":
    main()
