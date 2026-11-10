"""Repeat native mapping from completed same-campaign features after monitor failure."""

import argparse
import inspect
from pathlib import Path
import shutil
import sqlite3
import time

from atomic_json_windows import atomic_json, read_json
import run_sfm
from sqlite_verified_backup import verified_private_backup, complete_bytes_equal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", required=True, type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--mapper-seconds", type=int, default=3600)
    parser.add_argument("--restore-only", action="store_true")
    args = parser.parse_args()
    campaign = run_sfm.ROOT / "04_outputs/gaussian_full_raw_ab_20261008"
    prior = args.prior.resolve()
    prior.relative_to(campaign)
    if prior.name != "sfm" or not prior.parent.name.startswith("failed_sfm_"):
        raise ValueError("A resume requires an explicitly archived failed same-campaign SfM")
    archive = read_json(prior.parent / "ARCHIVE.json")
    if not archive["all_exact_owned_processes_dead"] or archive["stage"] != "sfm":
        raise ValueError("The old native process tree has not been verified dead")
    failure = read_json(prior / "FAILURE.json")["traceback"]
    if "PermissionError" not in failure or "STATE.json" not in failure:
        raise ValueError("This adapter only admits the evidenced monitor-publication interruption")
    source = inspect.getsource(run_sfm.run)
    start = "    features = args.output / \"features.db\"\n"
    finish = "    models_root = guided / \"models\"\n"
    if source.count(start) != 1 or source.count(finish) != 1:
        raise ValueError("The native feature/matching/mapping boundaries changed")
    first, last = source.index(start), source.index(finish)
    replacement = '''    _restore_completed_matching(args, state, names, params, binary)
    if args.restore_only:
        return
    features = args.output / "features.db"
    guided = args.output / "guided"
    database = guided / "database.db"
    matching_seconds, partition, merge = 0.0, None, None
'''
    modified = source[:first] + replacement + source[last:]
    namespace = dict(run_sfm.run.__globals__)

    def restore(arguments, state, names, params, binary):
        started = time.monotonic()
        contract = read_json(prior / "CONTRACT.json")
        old = read_json(prior / "STATE.json")
        if (contract["initial"] != run_sfm.data_identity(arguments.initial / "INITIAL_FRAME_MANIFEST.json")
            or contract["binary"] != run_sfm.data_identity(binary) or contract["mode"] != arguments.mode
            or contract["camera_params"] != list(map(float, params))):
            raise ValueError("The raw derivative, runtime, calibration, or matching mode changed")
        completed = [stage for stage in old["stages"] if stage["name"] in ("features", "guided_matching")]
        if [stage["name"] for stage in completed] != ["features", "guided_matching"] or any(
            stage["status"] != "completed" or any(code != 0 for code in stage["returncodes"]) for stage in completed):
            raise ValueError("Only fully completed native features and matches may be restored")
        for name in ("image_list.txt", "guided_pairs.txt"):
            if (prior / name).read_bytes() != (arguments.output / name).read_bytes():
                raise ValueError("The complete same-campaign frame or pair graph changed")
        paths = ("features.db", "FEATURES.json", "guided/database.db", "guided/MATCHING.json")
        identities = [run_sfm.data_identity(prior / name) for name in paths]
        (arguments.output / "guided").mkdir()
        backup_checks = []
        for name in paths:
            origin, target = prior / name, arguments.output / name
            if target.suffix == ".db":
                check = verified_private_backup(origin, target,
                    arguments.output / "private_SQLite_sources" / origin.stem, run_sfm.data_identity)
                backup_checks.append(check)
                with sqlite3.connect(target.as_uri() + "?mode=ro", uri=True) as connection:
                    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise ValueError("The completed restored database failed its complete integrity check")
                    saved_names = sorted(row[0] for row in connection.execute("SELECT name FROM images"))
                    if saved_names != sorted(names):
                        raise ValueError("The restored native database image membership changed")
            else:
                shutil.copyfile(origin, target)
                if not complete_bytes_equal(origin, target):
                    raise AssertionError("The complete restored metadata bytes changed")
        if identities != [run_sfm.data_identity(prior / name) for name in paths]:
            raise ValueError("The archived original database changed during restoration")
        for stage in completed:
            state["stages"].append({**stage, "same_campaign_completed_stage_reuse": True,
                                   "elapsed_seconds_charged_to_original_failed_attempt": True})
        atomic_json(arguments.output / "SAME_CAMPAIGN_RESTORE.json", {"status": "completed_exact_restore",
            "prior": str(prior), "source_identities": identities, "full_native_database_values_equal": True,
            "database_container_bytes_not_required_equal_after_SQLite_backup": True,
            "complete_metadata_file_bytes_equal": True, "SQLite_bundle_verification": backup_checks,
            "SQLite_integrity_and_full_frame_membership_verified": True, "restore_seconds": time.monotonic() - started,
            "historical_cache_or_partial_map_reused": False, "original_failure_cost_preserved": archive["elapsed_seconds"]})

    namespace["_restore_completed_matching"] = restore
    exec(compile(modified, "<native_mapping_after_owned_monitor_failure>", "exec"), namespace)
    try:
        namespace["run"](args)
    except BaseException:
        import traceback
        if args.output.exists():
            atomic_json(args.output / "FAILURE.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
