"""Profile the current exact RAW quality consumer without altering image scores."""

import argparse
import importlib
import json
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[2]
V1=ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2=ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0]=[str(V1),str(V2),str(ROOT/"02_code/box2_initial_pose_quality_v2_20261002")]
from runtime import atomic_json,data_identity
import quality_engine
import quality_histogram
from native_quality_workspace import NativeQualityWorkspace
from exact_bayer_gray import BayerGrayEngine,BayerGrayWorkspace,raw_gray_quality_execution
from single_copy_raw_reader import iter_single_copy_messages


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    output.relative_to(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/qc001")
    output.mkdir(parents=True,exist_ok=False)
    baseline=importlib.import_module("initial_selection")
    sys.path.insert(0,str(baseline.TOOLS))
    from mcap_audit import McapIndex
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr
    pilot_root=ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/bg001/s0"
    contract=load(pilot_root/"CONTRACT.json")
    selected=contract["selected_ids"]
    library=ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/bg001/b0/native_bayer_gray.dll"
    paths=[Path(__file__),Path(__file__).with_name("exact_bayer_gray.py"),Path(__file__).with_name("native_quality_workspace.py"),
        Path(quality_histogram.__file__),V1/"quality_engine.py",V2/"single_copy_raw_reader.py",Path(baseline.__file__),
        baseline.TOOLS/"camera_motion.py",baseline.TOOLS/"color_contract.py",baseline.TOOLS/"mcap_audit.py",library]
    sources=[data_identity(path) for path in paths]
    snapshot=output/"source_snapshot"
    snapshot.mkdir()
    for position,path in enumerate(paths[:-1]):
        (snapshot/(str(position)+"_"+path.name)).write_bytes(path.read_bytes())
    process=psutil.Process()
    atomic_json(output/"PROCESS_IDENTITY.json",{"pid":process.pid,"creation_time":process.create_time(),"scope":"Windows"})
    index=McapIndex(baseline.RAW)
    raw_stat=baseline.RAW.stat()
    if index.message_count(baseline.CAMERA_TOPIC)!=7338 or len(selected)!=128:
        raise ValueError("The original fixed RAW quality denominator changed")
    methods={"exact_gray":(BayerGrayWorkspace,"gray"),"derivative":(NativeQualityWorkspace,"derivative"),
        "resize":(NativeQualityWorkspace,"reduced"),"gray_moments":(quality_histogram,"gray_moments"),
        "meanStdDev":(cv2,"meanStdDev"),"goodFeaturesToTrack":(cv2,"goodFeaturesToTrack")}
    originals={name:getattr(owner,attribute) for name,(owner,attribute) in methods.items()}
    original_builder=quality_histogram.make_quality
    old_threads=cv2.getNumThreads()
    totals={name:{"calls":0,"worker_seconds_sum":0.} for name in methods}
    lock=threading.Lock()
    state={"status":"running","accepted_count_contribution":0}

    def wrapped(name):
        native=originals[name]

        def measured(*parameters,**keywords):
            before=time.perf_counter()
            value=native(*parameters,**keywords)
            elapsed=time.perf_counter()-before
            with lock:
                totals[name]["calls"]+=1
                totals[name]["worker_seconds_sum"]+=elapsed
            return value

        return measured

    engine=None
    workspace=None
    try:
        cv2.setNumThreads(1)
        for name,(owner,attribute) in methods.items():
            setattr(owner,attribute,wrapped(name))
        before=time.perf_counter()
        with raw_gray_quality_execution(library,demosaic_rggb8_to_bgr) as execution:
            with quality_histogram.histogram_quality_execution(True) as histogram_trace:
                cache=quality_engine.ScoreCache(output/"quality",contract["cache_contract"])
                rows,detail=quality_engine.score_messages(iter_single_copy_messages(index,baseline.CAMERA_TOPIC,selected),decode_bayer,
                    lambda raw:raw,quality_engine.compact_quality,workers=4,capacity=16,cache=cache,maximum_seconds=60)
        whole=time.perf_counter()-before
        for name,(owner,attribute) in methods.items():
            setattr(owner,attribute,originals[name])
        fields=("camera_frame","camera_header_ns","optical_frame","width","height","quality","cache_hit")
        current=[{name:row[name] for name in fields} for row in rows]
        expected=load(pilot_root/"t0/fused/SCORES.json")["rows"]
        if current!=expected or detail["cache_hits"]!=0 or detail["submitted"]!=128:
            raise ValueError("The actual complete RAW metadata, six fields, order or cold cache changed")
        atomic_json(output/"SCORES.json",{"rows":current,"counted":False})
        selection=[selected[position] for position in np.linspace(0,127,32,dtype=int)]
        components=[]
        corner_arrays={}
        engine=BayerGrayEngine(library)
        workspace=BayerGrayWorkspace(engine)
        iterator=iter_single_copy_messages(index,baseline.CAMERA_TOPIC,selection)
        try:
            for frame,message in iterator:
                _,_,raw=decode_bayer(message["payload"])
                gray=workspace.gray(raw)
                reduced=cv2.resize(gray,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
                before=time.perf_counter()
                corners=cv2.goodFeaturesToTrack(reduced,maxCorners=800,qualityLevel=.01,minDistance=5,blockSize=5)
                complete=time.perf_counter()-before
                timings={}
                before=time.perf_counter()
                eigen=cv2.cornerMinEigenVal(reduced,blockSize=5,ksize=3)
                timings["native_eigen_response_seconds"]=time.perf_counter()-before
                before=time.perf_counter()
                _,maximum,_,_=cv2.minMaxLoc(eigen)
                timings["native_minmax_seconds"]=time.perf_counter()-before
                before=time.perf_counter()
                _,thresholded=cv2.threshold(eigen,maximum*.01,0,cv2.THRESH_TOZERO,dst=eigen)
                timings["native_threshold_seconds"]=time.perf_counter()-before
                before=time.perf_counter()
                dilated=cv2.dilate(thresholded,None)
                timings["native_dilation_seconds"]=time.perf_counter()-before
                candidates=int(np.count_nonzero((thresholded[1:-1,1:-1]!=0)&(thresholded[1:-1,1:-1]==dilated[1:-1,1:-1])))
                count=0 if corners is None else len(corners)
                corner_arrays["frame_"+str(frame)]=np.empty((0,1,2),dtype=np.float32) if corners is None else corners
                components.append({"camera_frame":frame,"reduced_shape":list(reduced.shape),"original_corner_count":count,
                    "original_upper800_reached":count==800,"original_local_maximum_candidates":candidates,
                    "actual_native_goodFeaturesToTrack_seconds":complete,"standalone_native_preprocessing":timings,
                    "separate_call_time_difference_estimate_not_measured_sort_seconds":complete-sum(timings.values())})
        finally:
            iterator.close()
            workspace.close()
            engine.close()
        np.savez(output/"ORIGINAL_CORNERS.npz",**corner_arrays)
        checks={"all128_original_metadata_six_quality_fields_and_order_exact":current==expected,
            "all128_each_timed_original_function":all(row["calls"]==128 for row in totals.values()),
            "accepted_original_builder_aliases_and_DLL_closed":histogram_trace["aliases_restored"] and execution["builder_restored"] and execution["engine"]["DLL_unloaded"],
            "all_timed_functions_restored":all(getattr(owner,attribute) is originals[name] for name,(owner,attribute) in methods.items()),
            "all_original_scientific_sources_unchanged":sources==[data_identity(path) for path in paths],
            "original_RAW_identity_size_mtime_unchanged":(raw_stat.st_size,raw_stat.st_mtime_ns)==(baseline.RAW.stat().st_size,baseline.RAW.stat().st_mtime_ns),
            "full32_original_corner_coordinate_arrays_saved":len(components)==len(corner_arrays)==32}
        if not all(checks.values()):
            raise ValueError("A current complete native corner cost diagnostic contract failed")
        state.update(status="current_complete_RAW_quality_and_native_corner_cost_diagnostic_finished",checks=checks,sources=sources,
            complete_RAW128_phase_seconds=whole,detail=detail,original_function_worker_costs=totals,worker_sums_not_wall_clock_share=True,
            selected_component_RAW_ids=selection,components=components,upper800_count=sum(row["original_upper800_reached"] for row in components),
            component_native_GFTT_seconds_sum=sum(row["actual_native_goodFeaturesToTrack_seconds"] for row in components),
            component_separate_response_seconds_sum=sum(row["standalone_native_preprocessing"]["native_eigen_response_seconds"] for row in components),
            component_time_difference_not_native_sort_measurement=True,raw_identity={"path":str(baseline.RAW),"bytes":raw_stat.st_size,"mtime_ns":raw_stat.st_mtime_ns},
            generic_new_algorithm_full7338_formal_and_training_not_run=True,node_seconds=time.perf_counter()-started,counted=False)
        atomic_json(output/"RESULT.json",state)
        print({key:state[key] for key in ("status","complete_RAW128_phase_seconds","original_function_worker_costs","upper800_count",
            "component_native_GFTT_seconds_sum","component_separate_response_seconds_sum","accepted_count_contribution")},flush=True)
    except BaseException as error:
        state.update(status="failed",error_type=type(error).__name__,error=str(error),seconds=time.perf_counter()-started)
        atomic_json(output/"FAILURE.json",state)
        raise
    finally:
        for name,(owner,attribute) in methods.items():
            setattr(owner,attribute,originals[name])
        if workspace is not None:
            workspace.close()
        if engine is not None:
            engine.close()
        cv2.setNumThreads(old_threads)
        atomic_json(output/"RESOURCE_CLOSE.json",{"original_OpenCV_threads_restored":cv2.getNumThreads()==old_threads,
            "all_original_timed_functions_restored":all(getattr(owner,attribute) is originals[name] for name,(owner,attribute) in methods.items()),
            "original_builder_restored":quality_histogram.make_quality is original_builder,"all_owned_scoring_pool_lifetimes_completed":True,
            "node_seconds":time.perf_counter()-started,"counted":False})


if __name__=="__main__":
    main()
