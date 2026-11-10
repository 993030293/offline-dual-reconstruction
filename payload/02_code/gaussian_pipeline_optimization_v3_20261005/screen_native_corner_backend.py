"""Screen an unchanged native corner consumer under the two existing IPP backends."""

import argparse
import importlib
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np
import psutil

import profile_quality_corner_cost as reference

ROOT=reference.ROOT
OUT=ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005"


def corner(gray):
    return cv2.goodFeaturesToTrack(gray,maxCorners=800,qualityLevel=.01,minDistance=5,blockSize=5)


def same(left,right):
    return left is None and right is None or (left is not None and right is not None and np.array_equal(left,right))


def pair(gray):
    values=[]
    for enabled in (True,False):
        cv2.ipp.setUseIPP(enabled)
        if cv2.ipp.useIPP()!=enabled:
            raise ValueError("The requested existing IPP backend was not enabled")
        values.append(corner(gray))
    return values


def main():
    started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    output=parser.parse_args().output.resolve()
    output.relative_to(OUT/"qc001")
    output.mkdir(parents=True,exist_ok=False)
    atomic_json=reference.atomic_json
    identities=reference.data_identity
    baseline=importlib.import_module("initial_selection")
    sys.path.insert(0,str(baseline.TOOLS))
    from mcap_audit import McapIndex
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr
    library=OUT/"bg001/b0/native_bayer_gray.dll"
    paths=[Path(__file__),Path(reference.__file__),Path(__file__).with_name("exact_bayer_gray.py"),
        Path(__file__).with_name("native_quality_workspace.py"),Path(reference.quality_histogram.__file__),
        reference.V1/"quality_engine.py",reference.V2/"single_copy_raw_reader.py",Path(baseline.__file__),
        baseline.TOOLS/"camera_motion.py",baseline.TOOLS/"color_contract.py",baseline.TOOLS/"mcap_audit.py",library]
    sources=[identities(path) for path in paths]
    snapshot=output/"source_snapshot"
    snapshot.mkdir()
    for position,path in enumerate(paths[:-1]):
        (snapshot/(str(position)+"_"+path.name)).write_bytes(path.read_bytes())
    process=psutil.Process()
    atomic_json(output/"PROCESS_IDENTITY.json",{"pid":process.pid,"creation_time":process.create_time()})
    old_threads,old_ipp,old_optimized=cv2.getNumThreads(),cv2.ipp.useIPP(),cv2.useOptimized()
    builder=reference.quality_histogram.make_quality
    index=McapIndex(baseline.RAW)
    raw_stat=baseline.RAW.stat()
    selected=reference.load(OUT/"qc001/p0/RESULT.json")["selected_component_RAW_ids"]
    contract=reference.load(OUT/"bg001/s0/CONTRACT.json")
    atomic_json(output/"PROTOCOL.json",{"sources":sources,"selected_ids":selected,"generic_cases":48,
        "IPP_before":old_ipp,"optimized_before":old_optimized,"original_threads_before":old_threads,
        "component_not_whole_stage":True,"counted":False})
    engine=workspace=None
    iterator=None
    checks={}
    state={"status":"running","accepted_count_contribution":0}
    try:
        if cv2.__version__!="4.13.0" or not old_ipp or not old_optimized:
            raise ValueError("The registered original native dispatch backend changed")
        cv2.setNumThreads(1)
        rng=np.random.default_rng(148)
        generic=[]
        for shape in ((1,1),(2,7),(7,2),(31,37),(64,79),(120,160),(620,812),(621,813)):
            y,x=np.indices(shape)
            generic.extend([np.zeros(shape,np.uint8),np.full(shape,255,np.uint8),
                ((x//4+y//4)%2*255).astype(np.uint8),((x*3+y*5)%256).astype(np.uint8),
                rng.integers(0,256,size=shape,dtype=np.uint8),rng.choice(np.array([0,1,127,254,255],np.uint8),size=shape)])
        generic_rows=[]
        for number,gray in enumerate(generic):
            native,other=pair(gray)
            exact=same(native,other)
            generic_rows.append({"case":number,"shape":list(gray.shape),"exact":exact,
                "native_count":0 if native is None else len(native),"IPP_off_count":0 if other is None else len(other)})
            if not exact:
                np.savez(output/"FIRST_GENERIC_FAILURE.npz",input=gray,
                    IPP_on=np.empty((0,1,2),np.float32) if native is None else native,
                    IPP_off=np.empty((0,1,2),np.float32) if other is None else other)
                atomic_json(output/"GENERIC.json",{"rows":generic_rows,"counted":False})
                state.update(status="closed_negative_generic_corner_coordinates_or_order_changed",generic=generic_rows)
                return
        atomic_json(output/"GENERIC.json",{"rows":generic_rows,"counted":False})
        checks["all48_generic_corner_coordinates_and_order_exact"]=True
        engine=reference.BayerGrayEngine(library)
        workspace=reference.BayerGrayWorkspace(engine)
        reduced=[]
        real_rows=[]
        iterator=reference.iter_single_copy_messages(index,baseline.CAMERA_TOPIC,selected)
        with reference.raw_gray_quality_execution(library,demosaic_rggb8_to_bgr) as execution:
            with reference.quality_histogram.histogram_quality_execution(True) as aliases:
                for frame,message in iterator:
                    _,_,raw=decode_bayer(message["payload"])
                    small=cv2.resize(workspace.gray(raw),None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
                    native,other=pair(small)
                    quality=[]
                    for enabled in (True,False):
                        cv2.ipp.setUseIPP(enabled)
                        quality.append(reference.quality_engine.compact_quality(raw))
                    exact=same(native,other) and quality[0]==quality[1]
                    real_rows.append({"camera_frame":frame,"exact":exact,"quality_IPP_on":quality[0],"quality_IPP_off":quality[1]})
                    if not exact:
                        np.savez(output/"FIRST_REAL_FAILURE.npz",input=small,
                            IPP_on=np.empty((0,1,2),np.float32) if native is None else native,
                            IPP_off=np.empty((0,1,2),np.float32) if other is None else other)
                        break
                    reduced.append(small)
        atomic_json(output/"REAL.json",{"rows":real_rows,"execution":execution,"aliases":aliases,"counted":False})
        if len(real_rows)!=32 or not all(row["exact"] for row in real_rows):
            state.update(status="closed_negative_real_corner_or_six_quality_fields_changed",checks=checks,real_rows=real_rows)
            return
        checks["all32_original_corners_and_six_quality_fields_exact"]=True
        trials=[]
        for trial,order in enumerate(((True,False),(False,True),(True,False))):
            row={"trial":trial,"order":list(order)}
            for enabled in order:
                cv2.ipp.setUseIPP(enabled)
                before=time.perf_counter()
                values=[corner(gray) for gray in reduced]
                row["IPP_on" if enabled else "IPP_off"]=time.perf_counter()-before
                if len(values)!=32:
                    raise ValueError("The complete corner component denominator changed")
            trials.append(row)
        component_reduction=1-statistics.median(row["IPP_off"] for row in trials)/statistics.median(row["IPP_on"] for row in trials)
        checks["original_scientific_sources_and_DLL_unchanged"]=sources==[identities(path) for path in paths]
        checks["RAW_identity_size_mtime_unchanged"]=(raw_stat.st_size,raw_stat.st_mtime_ns)==(baseline.RAW.stat().st_size,baseline.RAW.stat().st_mtime_ns)
        checks["actual_original_builder_aliases_and_DLL_closed"]=aliases["aliases_restored"] and execution["builder_restored"] and execution["engine"]["DLL_unloaded"]
        if not all(checks.values()):
            raise ValueError("The native backend screen source or restoration contract failed")
        state.update(status="component_passed_complete_stage_pending" if component_reduction>=.05 else "closed_negative_component_speed_insufficient",
            checks=checks,trials=trials,component_reduction=component_reduction,component_not_whole_stage=True,
            generic_new_algorithm_full7338_and_training_not_run=True)
    except BaseException as error:
        state.update(status="failed",error_type=type(error).__name__,error=str(error))
        atomic_json(output/"FAILURE.json",state)
        raise
    finally:
        if iterator is not None:
            iterator.close()
        if workspace is not None:
            workspace.close()
        if engine is not None:
            engine.close()
        cv2.ipp.setUseIPP(old_ipp)
        cv2.setNumThreads(old_threads)
        closure={"original_IPP_restored":cv2.ipp.useIPP()==old_ipp,"original_threads_restored":cv2.getNumThreads()==old_threads,
            "optimized_state_unchanged":cv2.useOptimized()==old_optimized,"builder_restored":reference.quality_histogram.make_quality is builder,
            "owned_DLL_unloaded":engine is None or engine.receipt()["DLL_unloaded"],"counted":False}
        atomic_json(output/"RESOURCE_CLOSE.json",closure)
        state.update(sources=sources,node_seconds=time.perf_counter()-started,closure=closure,counted=False)
        atomic_json(output/"RESULT.json",state)
        print({key:state[key] for key in ("status","accepted_count_contribution","node_seconds")},flush=True)


if __name__=="__main__":
    main()
