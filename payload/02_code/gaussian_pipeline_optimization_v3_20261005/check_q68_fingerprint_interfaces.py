"""Verify existing queue ownership and full public file identity compatibility."""

import argparse
import importlib.util
from pathlib import Path
import shutil
import threading

from profile_scene_fingerprint_cost import ROOT, CODE
from profile_scene_transform_cost import BoundedSceneGuard
from runtime import atomic_json, data_identity

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scfq001"


def expect_error(action, kind):
    try:
        action()
    except kind:
        return True
    raise AssertionError("An explicit owned identity error was not propagated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OUT.resolve())
    output.mkdir(exist_ok=False)
    helper_path = CODE / "parallel_b_005/ordered_identity.py"
    sources = [data_identity(path) for path in (Path(__file__), helper_path, CODE / "profile_q68_fused_fingerprint.py")]
    spec = importlib.util.spec_from_file_location("checked_owned_Q68", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with BoundedSceneGuard(output, atomic_json) as guard:
        fixtures = output / "fixtures"
        fixtures.mkdir()
        paths = []
        for index, size in enumerate((0, 257, 4 * 1024**2 + 17)):
            path = fixtures / (str(index) + ".bin")
            path.write_bytes(bytes((index * 47 + 19) % 256 for _ in range(size)))
            paths.append(path)
        trace = {}
        expected = [data_identity(p) for p in (paths[0], paths[2], paths[0], paths[1])]
        actual = helper.collect_identities([paths[0], paths[2], paths[0], paths[1]], data_identity, trace)
        empty = helper.collect_identities([], data_identity, {})
        owner = helper.OrderedIdentities(paths, data_identity)
        other = []
        with owner:
            repeated = expect_error(owner.__enter__, RuntimeError)
            def wrong_consumer():
                other.append(expect_error(lambda: next(owner), RuntimeError))
            thread = threading.Thread(target=wrong_consumer)
            thread.start()
            thread.join()
            owned = list(owner)
        closed = expect_error(lambda: next(owner), RuntimeError)
        oversized = expect_error(lambda: helper.OrderedIdentities([paths[0]] * 65537, data_identity), MemoryError)
        public_root = ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/images"
        public_paths = sorted(p for p in public_root.iterdir() if p.is_file())
        if len(public_paths) != 360:
            raise ValueError("The existing full public development file pool changed")
        public_before = [data_identity(p) for p in public_paths]
        public_trace = {}
        public_actual = helper.collect_identities(public_paths, data_identity, public_trace)
        public_after = [data_identity(p) for p in public_paths]
        copies = output / "source_snapshot"
        copies.mkdir()
        for index, row in enumerate(sources):
            copied = copies / (f"{index:03d}_" + Path(row["path"]).name)
            shutil.copyfile(row["path"], copied)
            if data_identity(copied)["sha256"] != row["sha256"]:
                raise ValueError("An owned queue interface source copy changed")
        checks = {
            "empty_original_ordered_list_supported": empty == [],
            "duplicate_paths_empty_file_and_original4MiB_boundary_identities_exact": actual == expected,
            "same_owner_double_enter_explicit_error": repeated,
            "other_consumer_explicit_error_original_consumer_preserved": other == [True] and owned == [data_identity(p) for p in paths],
            "closed_owner_explicit_error": closed,
            "oversized_path_ownership_rejected_before_pool": oversized,
            "all360_public_full_file_identities_order_schema_exact": public_before == public_actual == public_after,
            "all_selected_public_and_own_sources_unchanged": sources == [data_identity(row["path"]) for row in sources],
            "all_actual_native_identity_workers_drained_with4_and8_bounds": all(t["closed"] and t["all_started_joined"] and t["workers"] == 4 and t["maximum_pending"] <= 8 for t in (trace, public_trace)),
            "no_identity_threads_retained": not any(t.name.startswith("b005-native-identity") for t in threading.enumerate()),
        }
        guard.sample()
        atomic_json(output / "RESULT.json", {"status": "owned_identity_queue10_and360_public_full_file_checks_passed0" if all(checks.values()) else "failed_verification",
            "checks": checks, "sources": sources, "traces": {"duplicates": trace, "public360": public_trace},
            "public_full_file_inputs": public_before, "public_already_used_development_not_GT_blind_or_whole_generalization": True,
            "new_optimization_count": 0})
        if not all(checks.values()):
            raise AssertionError("An existing queue ownership or public complete file identity check failed")
    print("Existing queue ownership and all360 public full file identities passed: 10 checks, combination new 0")


if __name__ == "__main__":
    main()
