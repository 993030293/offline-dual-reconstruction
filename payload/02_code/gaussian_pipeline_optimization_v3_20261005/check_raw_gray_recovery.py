"""Exercise real initial-entry interruption and original quality-cache recovery."""

import argparse
import ctypes
import importlib
import json
from pathlib import Path
import sys
import time

import cv2
import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2), str(ROOT / "02_code/box2_initial_pose_quality_v2_20261002")]
from runtime import atomic_json, data_identity
import quality_engine
import quality_histogram
import run_initial_pipeline as entry
import run_exact_raw_gray_input as full_entry
from native_quality_workspace import native_quality_workspace_execution
from exact_bayer_gray import raw_gray_quality_execution
from single_copy_raw_reader import iter_single_copy_messages


class InjectedStop(RuntimeError):
    pass


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "pilot", "library"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/bg001")
    output.mkdir(parents=True, exist_ok=False)
    pilot = load(args.pilot / "RESULT.json")
    if pilot["status"] != "component_screen_positive_complete_stage_pending":
        raise ValueError("The exact RAW recovery test requires the original positive pilot")
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr
    contract = load(args.pilot / "CONTRACT.json")
    selected = contract["selected_ids"]
    cache_contract = contract["cache_contract"]
    paths = [Path(__file__), Path(full_entry.__file__), Path(__file__).with_name("exact_bayer_gray.py"),
             Path(__file__).with_name("native_quality_workspace.py"), Path(quality_histogram.__file__), V1 / "quality_engine.py",
             V1 / "run_initial_pipeline.py", baseline.TOOLS / "camera_motion.py", baseline.TOOLS / "color_contract.py", args.library.resolve(strict=True)]
    sources = [data_identity(path) for path in paths]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for position, path in enumerate(paths[:-1]):
        (snapshot / (str(position)+"_"+path.name)).write_bytes(path.read_bytes())
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid":process.pid,"creation_time":process.create_time(),"scope":"Windows"})
    index = McapIndex(baseline.RAW)
    raw_stat = baseline.RAW.stat()
    original_score, original_entry_score = quality_engine.score_messages, entry.score_messages
    original_builder = quality_histogram.make_quality
    original_quality, original_entry_quality = quality_engine.compact_quality, entry.compact_quality
    old_argv, old_threads = sys.argv, cv2.getNumThreads()
    fields = ("camera_frame","camera_header_ns","optical_frame","width","height","quality")
    expected_rows = [{name:row[name] for name in fields} for row in load(args.pilot / "t0/native/SCORES.json")["rows"]]
    checks, traces = {}, {}
    module_handle = ctypes.windll.kernel32.GetModuleHandleW
    module_handle.argtypes, module_handle.restype = [ctypes.c_wchar_p], ctypes.c_void_p
    state = {"status":"running","accepted_count_contribution":0}

    def aliases_restored():
        return (quality_engine.score_messages is original_score and entry.score_messages is original_entry_score
                and quality_histogram.make_quality is original_builder and quality_engine.compact_quality is original_quality
                and entry.compact_quality is original_entry_quality)

    def score(mode, directory, numbers, corrupt=False):
        context = (native_quality_workspace_execution(True) if mode == "native" else
                   raw_gray_quality_execution(args.library, demosaic_rggb8_to_bgr))
        failure = None
        with context as execution:
            with quality_histogram.histogram_quality_execution(True) as histogram_trace:
                cache = quality_engine.ScoreCache(directory, cache_contract)
                converter = demosaic_rggb8_to_bgr if mode == "native" else (lambda raw:raw)
                try:
                    rows, detail = quality_engine.score_messages(iter_single_copy_messages(index, baseline.CAMERA_TOPIC, numbers),
                        decode_bayer, converter, quality_engine.compact_quality, workers=4, capacity=16, cache=cache, maximum_seconds=30)
                except ValueError as error:
                    if not corrupt or not str(error).startswith("Invalid quality cache identity/schema:"):
                        raise
                    failure = str(error)
                    rows, detail = [], {}
        if corrupt and failure is None:
            raise ValueError("The original corrupted quality cache was silently accepted")
        if not aliases_restored() or not histogram_trace["aliases_restored"]:
            raise ValueError("An interrupted original score alias did not restore")
        receipt = {"execution":execution,"histogram":histogram_trace,"detail":detail,"failure":failure,
                   "DLL_module_not_loaded_after_close":not module_handle(str(args.library.resolve()))}
        if not receipt["DLL_module_not_loaded_after_close"]:
            raise ValueError("An exact RAW DLL remained loaded after the owned pool drained")
        return rows, receipt

    try:
        cv2.setNumThreads(1)
        for mode in ("native","fused"):
            folder = output / mode
            folder.mkdir()
            counts = []

            def interrupted(*parameters, **keywords):
                original_progress = keywords.get("progress")

                def progress(count):
                    if original_progress is not None:
                        original_progress(count)
                    counts.append(count)
                    if count == 17:
                        raise InjectedStop("registered_original_initial_score_progress_17")

                keywords["progress"] = progress
                return original_score(*parameters, **keywords)

            quality_engine.score_messages, entry.score_messages = interrupted, interrupted
            sys.argv = [str(Path(full_entry.__file__)),"--mode",mode,"--library",str(args.library),
                        "--output",str(folder/"failed_entry"),"--quality-cache",str(folder/"quality")]
            caught = None
            try:
                full_entry.main()
            except InjectedStop as error:
                caught = str(error)
                if entry.score_messages is not interrupted or quality_engine.score_messages is not interrupted:
                    raise ValueError("The actual failed initial entry did not restore its original owner")
            finally:
                quality_engine.score_messages, entry.score_messages = original_score, original_entry_score
                sys.argv = old_argv
            cache_files = sorted(path.name for path in (folder/"quality").glob("*.json"))
            checks[mode+"_actual_full_entry_interrupted_at17_drained32_cache_items_and_restored"] = (
                caught == "registered_original_initial_score_progress_17" and counts == list(range(1,18))
                and len(cache_files) == 32 and aliases_restored() and not module_handle(str(args.library.resolve())))
            traces[mode] = {"full_entry_failure":caught,"progress_prefix":counts,"drained_partial_cache_names":cache_files}
            rows, recovery = score(mode,folder/"quality",selected)
            traces[mode]["recovery"] = recovery
            checks[mode+"_actual_partial_cache_recovered_all128_metadata_and_quality_exact"] = (
                [{name:row[name] for name in fields} for row in rows] == expected_rows and recovery["detail"]["cache_hits"] > 0)
            warm_rows, warm = score(mode,folder/"quality",selected)
            traces[mode]["warm"] = warm
            checks[mode+"_all128_warm_cache_hits_and_no_gray_conversion"] = warm["detail"]["cache_hits"] == 128
            if mode == "fused":
                checks[mode+"_all128_warm_cache_hits_and_no_gray_conversion"] &= warm["execution"]["engine"]["calls"] == 0
            else:
                checks[mode+"_all128_warm_cache_hits_and_no_gray_conversion"] &= warm["execution"]["workspace"]["calls"] == 0
            checks[mode+"_warm_metadata_and_quality_exact"] = [{name:row[name] for name in fields} for row in warm_rows] == expected_rows
            iterator = iter_single_copy_messages(index,baseline.CAMERA_TOPIC,[selected[0]])
            try:
                frame, message = next(iterator)
            finally:
                iterator.close()
            cache = quality_engine.ScoreCache(folder/"quality",cache_contract)
            bad_file = folder/"quality"/(cache.key(message["payload"])+".json")
            original_bytes = bad_file.read_bytes()
            damaged = json.loads(original_bytes)
            damaged["quality"].pop("sharpness")
            corrupted_bytes = json.dumps(damaged,allow_nan=False).encode("utf-8")
            (folder/"CORRUPTED_CACHE_FAILURE.json").write_bytes(corrupted_bytes)
            bad_file.write_bytes(corrupted_bytes)
            try:
                _, corrupted = score(mode,folder/"quality",[frame],True)
                traces[mode]["corrupted"] = corrupted
            finally:
                bad_file.write_bytes(original_bytes)
            repaired_rows, repaired = score(mode,folder/"quality",[frame])
            traces[mode]["repaired"] = repaired
            checks[mode+"_original_cache_corruption_error_then_exact_repair"] = (
                repaired["detail"]["cache_hits"] == 1 and [{name:row[name] for name in fields} for row in repaired_rows] == expected_rows[:1])
        left,right = output/"native/quality",output/"fused/quality"
        names = sorted(path.name for path in left.glob("*.json"))
        checks["all_partial_recovered_and_repaired_native_cache_bytes_exact"] = names == sorted(path.name for path in right.glob("*.json")) and all(
            (left/name).read_bytes() == (right/name).read_bytes() for name in names)
        checks["all_original_aliases_sources_and_raw_unchanged"] = aliases_restored() and sources == [data_identity(path) for path in paths] and (
            raw_stat.st_size,raw_stat.st_mtime_ns) == (baseline.RAW.stat().st_size,baseline.RAW.stat().st_mtime_ns)
        if not all(checks.values()):
            raise ValueError("A real RAW interruption, cold-to-warm recovery or cache corruption contract failed")
        state.update(status="actual_full_entry_interruption_and_cache_recovery_passed_complete_review_pending",checks=checks,
            traces=traces,sources=sources,all_recovered_cache_files=len(names),node_seconds=time.perf_counter()-started,
            no_new_training_mapper_LIO_or_GT_metric=True,counted=False)
        atomic_json(output/"RESULT.json",state)
        print({"status":state["status"],"checks":len(checks),"seconds":state["node_seconds"],"counted":False},flush=True)
    except BaseException as error:
        state.update(status="failed",checks=checks,traces=traces,error_type=type(error).__name__,error=str(error),seconds=time.perf_counter()-started)
        atomic_json(output/"FAILURE.json",state)
        raise
    finally:
        quality_engine.score_messages,entry.score_messages=original_score,original_entry_score
        sys.argv=old_argv
        cv2.setNumThreads(old_threads)
        atomic_json(output/"RESOURCE_CLOSE.json",{"all_aliases_restored":aliases_restored(),"original_threads_restored":cv2.getNumThreads()==old_threads,
            "DLL_module_not_loaded":not module_handle(str(args.library.resolve())),"node_seconds":time.perf_counter()-started,"counted":False})


if __name__=="__main__":
    main()
