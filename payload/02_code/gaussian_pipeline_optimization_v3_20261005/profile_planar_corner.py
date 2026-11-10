"""Measure three complete cold RAW scoring pairs with the current exact gray input."""

import argparse
from contextlib import nullcontext
import importlib
from pathlib import Path
import statistics
import sys
import time

import cv2
import psutil

import profile_quality_corner_cost as reference
import exact_bayer_gray
from planar_corner import planar_corner_execution,ORIGINAL_CORNER

ROOT=reference.ROOT
OUT=ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005"


def main():
    started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("output","generic","library"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    output.relative_to(OUT/"qc001")
    generic=reference.load(args.generic/"RESULT.json")
    if generic["status"]!="component_passed_complete_stage_pending" or not all(generic["checks"].values()) or generic["component_reduction"]<.05:
        raise ValueError("The preregistered planar generic and component checks failed")
    output.mkdir(parents=True,exist_ok=False)
    atomic_json=reference.atomic_json
    identity=reference.data_identity
    baseline=importlib.import_module("initial_selection")
    sys.path.insert(0,str(baseline.TOOLS))
    from mcap_audit import McapIndex
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr
    gray_library=OUT/"bg001/b0/native_bayer_gray.dll"
    paths=[Path(__file__),Path(__file__).with_name("planar_corner.py"),Path(__file__).with_name("native_planar_corner.cpp"),
        Path(__file__).with_name("exact_bayer_gray.py"),Path(__file__).with_name("native_quality_workspace.py"),
        Path(reference.quality_histogram.__file__),reference.V1/"quality_engine.py",reference.V2/"single_copy_raw_reader.py",
        Path(baseline.__file__),baseline.TOOLS/"camera_motion.py",baseline.TOOLS/"color_contract.py",baseline.TOOLS/"mcap_audit.py",
        args.library.resolve(strict=True),gray_library]
    sources=[identity(path) for path in paths]
    if sources[1:]!=generic["sources"][1:]:
        raise ValueError("An actual planar source differs from its executed generic verification")
    snapshot=output/"source_snapshot"
    snapshot.mkdir()
    for position,path in enumerate(paths[:-2]):
        (snapshot/(str(position)+"_"+path.name)).write_bytes(path.read_bytes())
    contract=reference.load(OUT/"bg001/s0/CONTRACT.json")
    selected=contract["selected_ids"]
    expected=reference.load(OUT/"bg001/s0/t0/fused/SCORES.json")["rows"]
    atomic_json(output/"CONTRACT.json",contract)
    process=psutil.Process()
    atomic_json(output/"PROCESS_IDENTITY.json",{"pid":process.pid,"creation_time":process.create_time()})
    index=McapIndex(baseline.RAW)
    raw_stat=baseline.RAW.stat()
    old_threads=cv2.getNumThreads()
    original_factory=exact_bayer_gray.compile_quality
    fields=("camera_frame","camera_header_ns","optical_frame","width","height","quality","cache_hit")
    state={"status":"running","pairs":[],"accepted_count_contribution":0}
    try:
        cv2.setNumThreads(1)
        for trial,order in enumerate((("current","planar"),("planar","current"),("current","planar"))):
            pair={"trial":trial,"order":order,"modes":{}}
            rows_by_mode={}
            for mode in order:
                folder=output/("t"+str(trial))/mode
                folder.mkdir(parents=True,exist_ok=False)
                before=time.perf_counter()
                context=planar_corner_execution(args.library) if mode=="planar" else nullcontext()
                with context as corner_trace:
                    with reference.raw_gray_quality_execution(gray_library,demosaic_rggb8_to_bgr) as gray_trace:
                        with reference.quality_histogram.histogram_quality_execution(True) as histogram_trace:
                            cache=reference.quality_engine.ScoreCache(folder/"quality",contract["cache_contract"])
                            rows,detail=reference.quality_engine.score_messages(
                                reference.iter_single_copy_messages(index,baseline.CAMERA_TOPIC,selected),decode_bayer,
                                lambda raw:raw,reference.quality_engine.compact_quality,workers=4,capacity=16,cache=cache,maximum_seconds=60)
                whole=time.perf_counter()-before
                rows_by_mode[mode]=[{name:row[name] for name in fields} for row in rows]
                if detail["submitted"]!=128 or detail["cache_hits"]!=0 or rows_by_mode[mode]!=expected or not gray_trace["builder_restored"] or not histogram_trace["aliases_restored"] or not gray_trace["engine"]["DLL_unloaded"]:
                    raise ValueError("The actual original RAW128 consumer, six fields or lifecycle changed")
                if mode=="planar" and (not corner_trace["original_factory_restored"] or not corner_trace["original_cv2_corner_unchanged"] or not corner_trace["workspace"]["DLL_unloaded"] or corner_trace["workspace"]["native_calls"]!=128 or corner_trace["workspace"]["fallback_calls"] or corner_trace["workspace"]["retained_plane_bytes"]):
                    raise ValueError("The actual four-worker planar consumer failed its original input or close contract")
                pair["modes"][mode]={"complete_phase_seconds":whole,"detail":detail,"gray_execution":gray_trace,
                    "corner_execution":corner_trace,"histogram":histogram_trace,"peak_RSS_bytes":process.memory_info().peak_wset}
                atomic_json(folder/"SCORES.json",{"rows":rows_by_mode[mode],"counted":False})
                atomic_json(folder/"TRACE.json",pair["modes"][mode])
            before=time.perf_counter()
            left=output/("t"+str(trial))/"current/quality"
            right=output/("t"+str(trial))/"planar/quality"
            names=sorted(path.name for path in left.iterdir())
            pair["checks"]={"all128_original_metadata_six_fields_order_exact":rows_by_mode["current"]==rows_by_mode["planar"]==expected,
                "all128_original_cold_cache_file_bytes_exact":len(names)==128 and names==sorted(path.name for path in right.iterdir()) and all((left/name).read_bytes()==(right/name).read_bytes() for name in names),
                "original_scoring_compiler_and_cv2_api_restored":exact_bayer_gray.compile_quality is original_factory and cv2.goodFeaturesToTrack is ORIGINAL_CORNER}
            pair["comparison_seconds_charged_each_side"]=time.perf_counter()-before
            pair["phase_seconds"]={mode:pair["modes"][mode]["complete_phase_seconds"]+pair["comparison_seconds_charged_each_side"] for mode in order}
            state["pairs"].append(pair)
            atomic_json(output/"STATE.json",state)
            print({"trial":trial,"phase_seconds":pair["phase_seconds"],"checks":pair["checks"]},flush=True)
            if not all(pair["checks"].values()):
                raise ValueError("An actual original full RAW128 scoring comparison failed")
        medians={mode:statistics.median(pair["phase_seconds"][mode] for pair in state["pairs"]) for mode in ("current","planar")}
        reduction=1-medians["planar"]/medians["current"]
        if sources!=[identity(path) for path in paths] or (raw_stat.st_size,raw_stat.st_mtime_ns)!=(baseline.RAW.stat().st_size,baseline.RAW.stat().st_mtime_ns):
            raise ValueError("An original planar source or RAW identity changed")
        state.update(status="complete128_screen_positive_complete7338_pending" if reduction>=.05 else "closed_negative_complete128_below5percent",
            medians=medians,reduction=reduction,sources=sources,three_same_process128_pairs_not_formal=True,
            full7338_recovery_external_and_training_not_run=True,node_seconds=time.perf_counter()-started,counted=False)
        atomic_json(output/"RESULT.json",state)
        print({key:state[key] for key in ("status","medians","reduction","accepted_count_contribution")},flush=True)
    except BaseException as error:
        state.update(status="failed",error_type=type(error).__name__,error=str(error))
        atomic_json(output/"FAILURE.json",state)
        raise
    finally:
        cv2.setNumThreads(old_threads)
        atomic_json(output/"RESOURCE_CLOSE.json",{"original_threads_restored":cv2.getNumThreads()==old_threads,
            "original_compiler_restored":exact_bayer_gray.compile_quality is original_factory,
            "original_cv2_corner_unchanged":cv2.goodFeaturesToTrack is ORIGINAL_CORNER,"all_owned_pool_lifetimes_completed":True,
            "node_seconds":time.perf_counter()-started,"counted":False})


if __name__=="__main__":
    main()
