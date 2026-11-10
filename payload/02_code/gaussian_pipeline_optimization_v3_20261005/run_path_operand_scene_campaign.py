"""Reuse the full guarded pair protocol for one distinct path consumer."""

import difflib
from pathlib import Path

from scene_transform_cost_node import ROOT, CODE
from runtime import atomic_json, data_identity
from review_scene_batch_negative import read

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/spr004"


def main():
    contract = read(OUT.parent / "spr003/g0/RESULT.json")
    prefix = "/mnt/c/" + ROOT.relative_to(Path("C:/")).as_posix() + "/"
    for row in contract["sources"]:
        if not row["path"].startswith(prefix):
            raise ValueError("The native contract source belongs to a different workspace")
        current = data_identity(ROOT / row["path"][len(prefix):])
        if any(current[key] != row[key] for key in ("bytes", "sha256")):
            raise ValueError("An actual native contract implementation changed")
    if not all(contract["checks"].values()):
        raise ValueError("The original native path and exception contract did not pass")
    original = CODE / "run_phase_isolated_scene_campaign.py"
    before = original.read_text(encoding="utf-8")
    after = before
    replacements = (
        ("gaussian_pipeline_optimization_v3_20261005/sip001", "gaussian_pipeline_optimization_v3_20261005/spr004", 1),
        ("profile_phase_isolated_scene.py", "profile_path_operand_scene.py", 2),
        ("phase_isolated_camera_prepare.py", "prefetch_path_operand.py", 1),
        ("phase_isolated_scene_node.py", "path_operand_scene_node.py", 2),
        ("phase_isolated_scene_preflight.py", "path_operand_scene_preflight.py", 1),
        ("PHASE_ADAPTER_CHECKS.json", "PATH_ADAPTER_CHECKS.json", 1),
        ("--phase-enabled", "--operand-enabled", 1),
        ('"isolated"', '"operand"', 9),
        ("first_complete_phase_positive_validation_pending0", "first_complete_path_positive_validation_pending0", 2),
        ("three_complete_phase_pairs_final_validation_and_cost_pending0", "three_complete_path_pairs_final_validation_and_cost_pending0", 1),
        ("closed_negative_complete_phase_screen0", "closed_negative_complete_path_screen0", 1),
        ("both_actual_phase5_ownership_checks_and_source_versions_exact", "both_actual_path5_binding_checks_and_source_versions_exact", 1),
    )
    for old, new, count in replacements:
        if after.count(old) != count:
            raise ValueError("The complete pair protocol source boundary changed: " + old)
        after = after.replace(old, new)
    source = data_identity(original)
    (OUT / "EXECUTED_ORIGINAL_PAIR_PROTOCOL.py").write_text(after, encoding="utf-8")
    (OUT / "ORIGINAL_PAIR_PROTOCOL.diff").write_text("".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="original_guarded_pair_protocol", tofile="path_operand_pair_protocol")), encoding="utf-8")
    namespace = {"__file__": str(original), "__name__": "owned_complete_path_operand_campaign"}
    exec(compile(after, str(original), "exec"), namespace)
    namespace["main"]()
    if source != data_identity(original):
        raise AssertionError("The previous negative experiment source changed")
    atomic_json(OUT / "PAIR_PROTOCOL_SOURCE_CHECK.json", {"original_source": source,
        "actual_wrapper_source": data_identity(Path(__file__)), "unchanged": True, "new_optimization_count": 0})


if __name__ == "__main__":
    main()
