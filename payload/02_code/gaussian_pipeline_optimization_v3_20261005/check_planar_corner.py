"""Compare the private planar response and corner arrays against the current native path."""

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
from exact_bayer_gray import compile_quality
from planar_corner import PlanarCornerWorkspace,OwnedCVProxy,ORIGINAL_CORNER,PARAMETERS

ROOT=reference.ROOT
OUT=ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005"


def arrays(value):
    return np.empty((0,1,2),np.float32) if value is None else value


def main():
    started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--library",type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    output.relative_to(OUT/"qc001")
    output.mkdir(parents=True,exist_ok=False)
    atomic_json=reference.atomic_json
    identity=reference.data_identity
    baseline=importlib.import_module("initial_selection")
    sys.path.insert(0,str(baseline.TOOLS))
    from mcap_audit import McapIndex
    from camera_motion import decode_bayer
    library=args.library.resolve(strict=True)
    built=reference.load(library.parent/"RESULT.json")
    if built["status"]!="built_numerical_verification_pending" or built["DLL"]!=identity(library):
        raise ValueError("The actual private compiled DLL identity changed")
    gray_library=OUT/"bg001/b0/native_bayer_gray.dll"
    paths=[Path(__file__),Path(__file__).with_name("planar_corner.py"),Path(__file__).with_name("native_planar_corner.cpp"),
        Path(__file__).with_name("exact_bayer_gray.py"),Path(__file__).with_name("native_quality_workspace.py"),
        Path(reference.quality_histogram.__file__),reference.V1/"quality_engine.py",reference.V2/"single_copy_raw_reader.py",
        Path(baseline.__file__),baseline.TOOLS/"camera_motion.py",baseline.TOOLS/"color_contract.py",baseline.TOOLS/"mcap_audit.py",library,gray_library]
    sources=[identity(path) for path in paths]
    if sources[2]!=built["sources"][1]:
        raise ValueError("The native planar source differs from the compiled input")
    snapshot=output/"source_snapshot"
    snapshot.mkdir()
    for position,path in enumerate(paths[:-2]):
        (snapshot/(str(position)+"_"+path.name)).write_bytes(path.read_bytes())
    process=psutil.Process()
    atomic_json(output/"PROCESS_IDENTITY.json",{"pid":process.pid,"creation_time":process.create_time()})
    old_threads,old_ipp=cv2.getNumThreads(),cv2.ipp.useIPP()
    selected=reference.load(OUT/"qc001/p0/RESULT.json")["selected_component_RAW_ids"]
    raw_stat=baseline.RAW.stat()
    state={"status":"running","accepted_count_contribution":0}
    native=engine=gray_workspace=None
    iterator=None
    checks={}
    verified=[]

    def compare(label,gray):
        original_response=cv2.cornerMinEigenVal(gray,blockSize=5,ksize=3)
        response=native.response(gray).copy()
        before=ORIGINAL_CORNER(gray,**PARAMETERS)
        after=native.goodFeaturesToTrack(gray,**PARAMETERS)
        response_exact=np.array_equal(original_response.view(np.uint32),response.view(np.uint32))
        corner_exact=np.array_equal(arrays(before),arrays(after)) and (before is None)==(after is None)
        row={"input":label,"shape":list(gray.shape),"response_bits_exact":response_exact,"coordinates_and_order_exact":corner_exact,
            "native_count":len(arrays(before)),"planar_count":len(arrays(after)),
            "different_response_pixels":int(np.count_nonzero(original_response.view(np.uint32)!=response.view(np.uint32)))}
        verified.append(row)
        if not response_exact or not corner_exact:
            np.savez(output/"FIRST_EQUIVALENCE_FAILURE.npz",input=gray,native_response=original_response,planar_response=response,
                native_corners=arrays(before),planar_corners=arrays(after))
            atomic_json(output/"EQUIVALENCE.json",{"rows":verified,"counted":False})
            return False
        return True

    try:
        cv2.setNumThreads(1)
        if cv2.__version__!="4.13.0" or not old_ipp or not cv2.useOptimized():
            raise ValueError("The registered original OpenCV backend changed")
        native=PlanarCornerWorkspace(library)
        rng=np.random.default_rng(149)
        cases=[]
        for shape in ((1,1),(2,7),(7,2),(31,37),(64,79),(120,160),(620,812),(621,813)):
            y,x=np.indices(shape)
            cases.extend([np.zeros(shape,np.uint8),np.full(shape,255,np.uint8),
                ((x//4+y//4)%2*255).astype(np.uint8),((x*3+y*5)%256).astype(np.uint8),
                rng.integers(0,256,size=shape,dtype=np.uint8),rng.choice(np.array([0,1,127,254,255],np.uint8),size=shape)])
        for number,gray in enumerate(cases):
            if not compare("generic_"+str(number),gray):
                state.update(status="closed_negative_generic_response_or_corner_changed",verified=verified)
                return
        checks["all48_generic_response_bits_and_original_corner_arrays_exact"]=True
        engine=reference.BayerGrayEngine(gray_library)
        gray_workspace=reference.BayerGrayWorkspace(engine)
        current=compile_quality(reference.quality_engine.compact_quality,gray_workspace,reference.quality_histogram)
        candidate=compile_quality(reference.quality_engine.compact_quality,gray_workspace,reference.quality_histogram)
        candidate.__globals__["cv2"]=OwnedCVProxy(native)
        index=McapIndex(baseline.RAW)
        iterator=reference.iter_single_copy_messages(index,baseline.CAMERA_TOPIC,selected)
        real_rows=[]
        reduced=[]
        for frame,message in iterator:
            _,_,raw=decode_bayer(message["payload"])
            small=cv2.resize(gray_workspace.gray(raw),None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
            if not compare("RAW_"+str(frame),small):
                state.update(status="closed_negative_real_response_or_corner_changed",verified=verified)
                return
            before,after=current(raw),candidate(raw)
            exact=before==after
            real_rows.append({"camera_frame":frame,"quality_exact":exact,"native":before,"planar":after})
            if not exact:
                atomic_json(output/"FIRST_QUALITY_FAILURE.json",real_rows[-1])
                state.update(status="closed_negative_six_quality_fields_changed",verified=verified)
                return
            reduced.append(small)
        checks["all32_actual_responses_coordinates_order_and_six_fields_exact"]=len(real_rows)==32 and all(row["quality_exact"] for row in real_rows)
        atomic_json(output/"REAL.json",{"rows":real_rows,"counted":False})
        trials=[]
        for trial,order in enumerate((("native","planar"),("planar","native"),("native","planar"))):
            row={"trial":trial,"order":order}
            for name in order:
                function=ORIGINAL_CORNER if name=="native" else native.goodFeaturesToTrack
                before=time.perf_counter()
                values=[function(gray,**PARAMETERS) for gray in reduced]
                row[name]=time.perf_counter()-before
                if len(values)!=32:
                    raise ValueError("The complete corner component denominator changed")
            trials.append(row)
        reduction=1-statistics.median(row["planar"] for row in trials)/statistics.median(row["native"] for row in trials)
        checks["all_original_source_and_DLL_inputs_unchanged"]=sources==[identity(path) for path in paths]
        checks["RAW_size_mtime_unchanged"]=(raw_stat.st_size,raw_stat.st_mtime_ns)==(baseline.RAW.stat().st_size,baseline.RAW.stat().st_mtime_ns)
        checks["original_cv2_corner_not_mutated"]=cv2.goodFeaturesToTrack is ORIGINAL_CORNER
        if not all(checks.values()):
            raise ValueError("A planar corner generic or real source contract failed")
        atomic_json(output/"EQUIVALENCE.json",{"rows":verified,"counted":False})
        state.update(status="component_passed_complete_stage_pending" if reduction>=.05 else "closed_negative_component_cost_insufficient",
            checks=checks,component_reduction=reduction,trials=trials,component_not_whole_stage=True,
            complete128_7338_recovery_external_training_not_run=True,workspace_before_close=native.receipt())
    except BaseException as error:
        state.update(status="failed",error_type=type(error).__name__,error=str(error))
        atomic_json(output/"FAILURE.json",state)
        raise
    finally:
        if iterator is not None:
            iterator.close()
        if native is not None:
            native.close()
        if gray_workspace is not None:
            gray_workspace.close()
        if engine is not None:
            engine.close()
        cv2.setNumThreads(old_threads)
        closure={"original_threads_restored":cv2.getNumThreads()==old_threads,"original_IPP_unchanged":cv2.ipp.useIPP()==old_ipp,
            "original_cv2_corner_not_mutated":cv2.goodFeaturesToTrack is ORIGINAL_CORNER,
            "planar_workspace":None if native is None else native.receipt(),"gray_engine":None if engine is None else engine.receipt(),"counted":False}
        atomic_json(output/"RESOURCE_CLOSE.json",closure)
        state.update(sources=sources,node_seconds=time.perf_counter()-started,closure=closure,counted=False)
        atomic_json(output/"RESULT.json",state)
        print({key:state[key] for key in ("status","accepted_count_contribution","node_seconds")},flush=True)


if __name__=="__main__":
    main()
