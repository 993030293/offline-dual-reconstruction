"""Change only new augmentation proposals in the frozen native recording stage."""

import inspect
import linecache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "02_code/gaussian_full_raw_ab_20261008"
sys.path.insert(0, str(OLD))
import run_native_stage
from atomic_json_windows import atomic_json, read_json
from augmentation_graph import proposal_pairs, POLICY


def validate_registration_guards(result):
    """Keep original registration and pose-gap recovery as separate contracts."""
    if result['purpose'] == 'pose_gap':
        if not result['pose_gap_registration_guard_passed']:
            raise RuntimeError('Sparse matching failed to register every selected pose-gap image')
        best = max(result['models'], key=lambda row: row['registered_count'])
        if (best['original_registered_frames_lost']
                or len(set(best['additions_registered'])) != result['added_count']
                or best['registered_count'] < best['original_registered_count'] + result['added_count']):
            raise RuntimeError('Pose-gap recovery lost a registered camera or a selected additional image')
        return 'pose_gap_only_with_existing_registered_cameras_preserved'
    if not result['original_registration_guard_passed']:
        raise RuntimeError('Sparse matching failed the complete original image registration gate')
    return 'complete_original_registration'


def main():
    if "--kind" not in sys.argv or sys.argv[sys.argv.index("--kind")+1] != "temporal":
        raise ValueError("This adapter owns only augmentation proposals")
    import run_temporal_augmentation as native
    original = native.main
    source = inspect.getsource(original)
    before = 'pairs = [(a, b) for i, a in enumerate(all_names) for b in all_names[i+1:] if a in additions or b in additions]'
    after = 'pairs = proposal_pairs(all_names, additions, by_name, previous, args.output)'
    if source.count(before) != 1:
        raise ValueError("The frozen augmentation proposal anchor changed")
    source = source.replace(before, after)
    before_contract = '"pair_policy": "new image with every old/new image; unchanged pairs remain cached"'
    if source.count(before_contract) != 1:
        raise ValueError("The frozen augmentation contract anchor changed")
    source = source.replace(before_contract, '"pair_policy": "new image with temporal and covisibility neighbors; old pairs remain cached; all new-new pairs retained"')
    namespace = {**original.__globals__, "proposal_pairs": proposal_pairs, "atomic_json": atomic_json}
    filename = "<native_sparse_incremental_augmentation>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), namespace)
    output = Path(sys.argv[sys.argv.index("--output")+1])
    atomic_json(output.parent / (output.name+"_SPARSE_ADAPTER.json"), {
        "status": "frozen_before_execution", "policy": POLICY,
        "native_source": str(Path(native.__file__).resolve()),
        "adapter_source": run_native_stage.data_identity(Path(__file__)),
        "graph_source": run_native_stage.data_identity(Path(__file__).with_name("augmentation_graph.py")),
        "only_changes": ["new_pair_proposals", "pair_policy_receipt", "Windows_atomic_publisher",
            "purpose_scoped_registration_evaluation"],
        "feature_and_geometry_thresholds_unchanged": True,
        "fixed_existing_camera_poses_unchanged": True,
        "all_initial_and_additional_image_denominators_preserved": True})
    native.main = namespace["main"]
    try:
        run_native_stage.main()
        result = read_json(output / "RESULT.json")
        scope = validate_registration_guards(result)
        result["incremental_pair_policy"] = POLICY
        result["sparse_graph_registration_verified"] = scope in (
            'complete_original_registration', 'pose_gap_only_with_existing_registered_cameras_preserved')
        result['sparse_graph_registration_guard_scope'] = scope
        result['original_full_registration_failure_preserved'] = not result['original_registration_guard_passed']
        atomic_json(output / "RESULT.json", result)
        (output / "executed_native_body.py").write_text(source, encoding="utf-8")
    finally:
        native.main = original


if __name__ == "__main__":
    main()
