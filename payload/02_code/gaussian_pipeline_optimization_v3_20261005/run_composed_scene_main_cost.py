"""Run and compare the full unchanged Scene after timing original main calls."""

import difflib
from pathlib import Path

from review_scene_batch_negative import read
from scene_transform_cost_node import ROOT, CODE
from runtime import atomic_json, data_identity

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sclm002"


def main():
    original = CODE / "run_composed_scene_cost.py"
    before = original.read_text(encoding="utf-8")
    after = before
    for old, new, expected in (
        ("gaussian_pipeline_optimization_v3_20261005/sclc001", "gaussian_pipeline_optimization_v3_20261005/sclm002", 2),
        ("composed_scene_cost_node.py", "composed_scene_main_cost_node.py", 2),
        ("profile_composed_scene_cost.py", "profile_composed_scene_main_cost.py", 2),
    ):
        if after.count(old) != expected:
            raise ValueError("The original complete cost runner source changed")
        after = after.replace(old, new)
    source = data_identity(original)
    (OUT / "EXECUTED_ORIGINAL_COST_RUNNER.py").write_text(after, encoding="utf-8")
    (OUT / "ORIGINAL_COST_RUNNER.diff").write_text("".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="preparation_cost_runner", tofile="main_call_cost_runner")), encoding="utf-8")
    namespace = {"__file__": str(original), "__name__": "owned_original_main_cost_runner"}
    exec(compile(after, str(original), "exec"), namespace)
    namespace["main"]()
    result = read(OUT / "p0/MAIN_CALL_COST.json")
    if not all(result["checks"].values()) or source != data_identity(original):
        raise AssertionError("The full main-call diagnostic did not preserve original calls")
    atomic_json(OUT / "MAIN_SUMMARY.json", {"checks": result["checks"], "groups": result["groups"],
        "source": data_identity(Path(__file__)), "preserved_original_runner": source, "new_optimization_count": 0})
    print({"original_calls": result["groups"], "new": 0})


if __name__ == "__main__":
    main()
