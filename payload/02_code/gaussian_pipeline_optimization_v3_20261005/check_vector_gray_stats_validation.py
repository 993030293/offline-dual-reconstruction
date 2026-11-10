"""Check real initial-entry interruption, exact cache recovery and external RGB inputs."""

import argparse
from contextlib import nullcontext
import ctypes
import importlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import psutil

import profile_quality_corner_cost as reference
import exact_bayer_gray
import run_vector_gray_stats_input as full_entry
import run_initial_pipeline as entry
from planar_corner import planar_corner_execution, ORIGINAL_CORNER
from vector_gray_stats import vector_gray_stats_execution

ROOT = reference.ROOT
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"


class InjectedStop(RuntimeError):
    pass


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "library", "pilot"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OUT / "vector_gray_stats_001")
    output.mkdir(parents=True, exist_ok=False)
    load, atomic_json, identity = reference.load, reference.atomic_json, reference.data_identity
    pilot = load(args.pilot / "RESULT.json")
    if pilot["status"] != "complete128_screen_positive_complete7338_pending" or pilot["reduction"] < .05:
        raise ValueError("The actual original RAW128 prerequisite failed")
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr
    from mcap_audit import McapIndex
    gray_library = OUT / "bg001/b0/native_bayer_gray.dll"
    corner_library = OUT / "qc001/b0/native_planar_corner.dll"
    paths = [Path(__file__), Path(__file__).with_name("vector_gray_stats.py"), Path(__file__).with_name("native_vector_gray_stats.cpp"),
        Path(exact_bayer_gray.__file__), Path(full_entry.__file__), Path(full_entry.accepted.__file__),
        Path(__file__).with_name("planar_corner.py"), Path(reference.quality_histogram.__file__), reference.V1 / "quality_engine.py",
        reference.V1 / "run_initial_pipeline.py", reference.V2 / "single_copy_raw_reader.py", Path(baseline.__file__),
        baseline.TOOLS / "camera_motion.py", baseline.TOOLS / "color_contract.py", baseline.TOOLS / "mcap_audit.py",
        args.library.resolve(strict=True), gray_library, corner_library]
    sources = [identity(path) for path in paths]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for position, path in enumerate(paths[:-3]):
        (snapshot / (str(position) + "_" + path.name)).write_bytes(path.read_bytes())
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time()})
    contract = load(args.pilot / "CONTRACT.json")
    selected = contract["selected_ids"]
    fields = ("camera_frame", "camera_header_ns", "optical_frame", "width", "height", "quality")
    expected = [{name: row[name] for name in fields} for row in load(args.pilot / "t0/current/SCORES.json")["rows"]]
    old_argv, old_threads, old_ipp = sys.argv, cv2.getNumThreads(), cv2.ipp.useIPP()
    original_score, original_entry_score = reference.quality_engine.score_messages, entry.score_messages
    original_quality, original_entry_quality = reference.quality_engine.compact_quality, entry.compact_quality
    original_builder, original_factory = reference.quality_histogram.make_quality, exact_bayer_gray.compile_quality
    index = McapIndex(baseline.RAW)
    raw_stat = baseline.RAW.stat()
    module_handle = ctypes.windll.kernel32.GetModuleHandleW
    module_handle.argtypes, module_handle.restype = [ctypes.c_wchar_p], ctypes.c_void_p
    libraries = (args.library.resolve(), gray_library.resolve(), corner_library.resolve())
    checks, traces = {}, {}
    state = {"status": "running", "accepted_count_contribution": 0}
    engine = gray_workspace = None

    def restored():
        return (reference.quality_engine.score_messages is original_score and entry.score_messages is original_entry_score
            and reference.quality_engine.compact_quality is original_quality and entry.compact_quality is original_entry_quality
            and reference.quality_histogram.make_quality is original_builder and exact_bayer_gray.compile_quality is original_factory
            and cv2.goodFeaturesToTrack is ORIGINAL_CORNER)

    def score(mode, directory, numbers, corrupt=False):
        failure = None
        with planar_corner_execution(corner_library) as corner_trace:
            with (vector_gray_stats_execution(args.library) if mode == "vector" else nullcontext()) as vector_trace:
                with exact_bayer_gray.raw_gray_quality_execution(gray_library, demosaic_rggb8_to_bgr) as gray_trace:
                    with reference.quality_histogram.histogram_quality_execution(True) as aliases:
                        cache = reference.quality_engine.ScoreCache(directory, contract["cache_contract"])
                        try:
                            rows, detail = original_score(reference.iter_single_copy_messages(index, baseline.CAMERA_TOPIC, numbers),
                                decode_bayer, lambda raw: raw, reference.quality_engine.compact_quality,
                                workers=4, capacity=16, cache=cache, maximum_seconds=30)
                        except ValueError as error:
                            if not corrupt or not str(error).startswith("Invalid quality cache identity/schema:"):
                                raise
                            failure = str(error)
                            rows, detail = [], {}
        if corrupt and failure is None:
            raise ValueError("The original invalid quality cache was silently accepted")
        closed = restored() and aliases["aliases_restored"] and all(not module_handle(str(path)) for path in libraries)
        if not closed:
            raise ValueError("The original recovered score aliases or private DLL did not close")
        return rows, {"detail": detail, "failure": failure, "gray": gray_trace,
                      "corner": corner_trace, "vector": vector_trace, "closed": closed}

    try:
        cv2.setNumThreads(1)
        for mode in ("current", "vector"):
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

            reference.quality_engine.score_messages, entry.score_messages = interrupted, interrupted
            sys.argv = [str(Path(full_entry.__file__)), "--mode", mode, "--library", str(args.library),
                "--gray-library", str(gray_library), "--corner-library", str(corner_library),
                "--output", str(folder / "failed_entry"), "--quality-cache", str(folder / "quality")]
            caught = None
            try:
                full_entry.main()
            except InjectedStop as error:
                caught = str(error)
                if entry.score_messages is not interrupted or reference.quality_engine.score_messages is not interrupted:
                    raise ValueError("The failed complete initial entry did not restore its original owner")
            finally:
                reference.quality_engine.score_messages, entry.score_messages = original_score, original_entry_score
                sys.argv = old_argv
            names = sorted(path.name for path in (folder / "quality").glob("*.json"))
            checks[mode + "_actual_full_entry_stop17_drained32_partial_cache_and_restored"] = (
                caught == "registered_original_initial_score_progress_17" and counts == list(range(1, 18))
                and len(names) == 32 and restored() and all(not module_handle(str(path)) for path in libraries))
            traces[mode] = {"actual_entry_exception": caught, "progress_prefix": counts, "partial_cache_files": names}
            rows, recovered = score(mode, folder / "quality", selected)
            traces[mode]["recovered"] = recovered
            checks[mode + "_partial_cache128_recovery_original_metadata_six_fields_exact"] = (
                [{name: row[name] for name in fields} for row in rows] == expected and recovered["detail"]["cache_hits"] > 0)
            warm_rows, warm = score(mode, folder / "quality", selected)
            traces[mode]["warm"] = warm
            checks[mode + "_all128_warm_hits_zero_gray_corner_and_vector_stats"] = warm["detail"]["cache_hits"] == 128 and (
                warm["gray"]["engine"]["calls"] == warm["corner"]["workspace"]["native_calls"] == 0)
            if mode == "vector":
                checks[mode + "_all128_warm_hits_zero_gray_corner_and_vector_stats"] &= warm["vector"]["engine"]["native_calls"] == 0
            checks[mode + "_warm_original_metadata_six_fields_exact"] = [{name: row[name] for name in fields} for row in warm_rows] == expected
            iterator = reference.iter_single_copy_messages(index, baseline.CAMERA_TOPIC, [selected[0]])
            try:
                frame, message = next(iterator)
            finally:
                iterator.close()
            cache = reference.quality_engine.ScoreCache(folder / "quality", contract["cache_contract"])
            damaged_path = folder / "quality" / (cache.key(message["payload"]) + ".json")
            original_bytes = damaged_path.read_bytes()
            damaged = json.loads(original_bytes)
            damaged["quality"].pop("sharpness")
            damaged_bytes = json.dumps(damaged, allow_nan=False).encode("utf-8")
            (folder / "CORRUPTED_CACHE_FAILURE.json").write_bytes(damaged_bytes)
            damaged_path.write_bytes(damaged_bytes)
            try:
                _, failure = score(mode, folder / "quality", [frame], True)
                traces[mode]["corrupted"] = failure
            finally:
                damaged_path.write_bytes(original_bytes)
            repaired, repair = score(mode, folder / "quality", [frame])
            traces[mode]["repaired"] = repair
            checks[mode + "_original_schema_error_exact_byte_repair"] = repair["detail"]["cache_hits"] == 1 and (
                [{name: row[name] for name in fields} for row in repaired] == expected[:1])
        names = sorted(path.name for path in (output / "current/quality").glob("*.json"))
        checks["all_partial_recovered_repaired_cache_files_bytes_exact"] = names == sorted(
            path.name for path in (output / "vector/quality").glob("*.json")) and all(
                (output / "current/quality" / name).read_bytes() == (output / "vector/quality" / name).read_bytes() for name in names)
        image_root = ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/images"
        images = sorted(image_root.glob("*.png"))
        if len(images) != 360:
            raise ValueError("The original public RGB denominator changed")
        image_files = [images[position] for position in np.linspace(0, 359, 32, dtype=int)]
        image_sources = [identity(path) for path in image_files]
        engine = reference.BayerGrayEngine(gray_library)
        gray_workspace = reference.BayerGrayWorkspace(engine)
        external = []
        with planar_corner_execution(corner_library) as external_corner:
            current = exact_bayer_gray.compile_quality(original_quality, gray_workspace, reference.quality_histogram)
            with vector_gray_stats_execution(args.library) as external_vector:
                candidate = exact_bayer_gray.compile_quality(original_quality, gray_workspace, reference.quality_histogram)
                for path in image_files:
                    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                        raise ValueError("An original external RGB image could not be decoded")
                    raw = np.empty(image.shape[:2], np.uint8)
                    for y, x, channel in ((0, 0, 2), (0, 1, 1), (1, 0, 1), (1, 1, 0)):
                        raw[y::2, x::2] = image[y::2, x::2, channel]
                    immutable = raw.copy()
                    before, after = current(raw), candidate(raw)
                    exact = before == after and np.array_equal(np.array(list(before.values()), np.float64).view(np.uint64),
                        np.array(list(after.values()), np.float64).view(np.uint64)) and np.array_equal(raw, immutable)
                    external.append({"image": path.name, "shape": list(raw.shape), "six_fields_and_float_bits_exact": exact,
                                     "current": before, "vector": after})
                    if not exact:
                        np.savez(output / "FIRST_EXTERNAL_FAILURE.npz", raw=raw)
                        raise ValueError("An external RGB-derived original vector quality result changed")
        gray_workspace.close()
        engine.close()
        traces["external"] = {"corner": external_corner, "vector": external_vector}
        atomic_json(output / "EXTERNAL.json", {"rows": external, "input_images": image_sources, "original_RGB_count": 360,
            "derived_RGGB_not_sensor_RAW_or_GT": True, "development_inputs_not_blind": True, "counted": False})
        checks["all32_public_RGB_derived_original_six_fields_float_bits_exact"] = len(external) == 32 and all(
            row["six_fields_and_float_bits_exact"] for row in external) and external_vector["engine"]["native_calls"] == 32
        checks["all_public_inputs_and_scientific_sources_unchanged"] = image_sources == [identity(path) for path in image_files] and (
            sources == [identity(path) for path in paths])
        checks["original_RAW_size_mtime_unchanged"] = (raw_stat.st_size, raw_stat.st_mtime_ns) == (
            baseline.RAW.stat().st_size, baseline.RAW.stat().st_mtime_ns)
        checks["all_original_aliases_IPP_private_DLLs_closed"] = restored() and cv2.ipp.useIPP() == old_ipp and all(
            not module_handle(str(path)) for path in libraries)
        if not all(checks.values()):
            raise ValueError("A registered actual recovery or external exactness check failed")
        state.update(status="real_entry_interruption_score_recovery_and_external_derived_checks_passed", checks=checks,
            traces=traces, sources=sources, input_images=image_sources, hard_crash_full1002_resume_GT_unseen_whole_training_not_verified=True,
            node_seconds=time.perf_counter() - started, counted=False)
        atomic_json(output / "RESULT.json", state)
        print({"status": state["status"], "checks": len(checks), "seconds": state["node_seconds"], "counted": False}, flush=True)
    except BaseException as error:
        state.update(status="failed", checks=checks, traces=traces, error_type=type(error).__name__, error=str(error))
        atomic_json(output / "FAILURE.json", state)
        raise
    finally:
        if gray_workspace is not None:
            gray_workspace.close()
        if engine is not None:
            engine.close()
        reference.quality_engine.score_messages, entry.score_messages = original_score, original_entry_score
        sys.argv = old_argv
        cv2.setNumThreads(old_threads)
        atomic_json(output / "RESOURCE_CLOSE.json", {"original_aliases_factory_and_threads_restored": restored()
            and cv2.getNumThreads() == old_threads, "original_IPP_unchanged": cv2.ipp.useIPP() == old_ipp,
            "private_DLLs_not_loaded": all(not module_handle(str(path)) for path in libraries), "owned_pools_drained": True,
            "node_seconds": time.perf_counter() - started, "counted": False})


if __name__ == "__main__":
    main()
