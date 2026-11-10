"""Check frozen RGGB fusion on Bayer mosaics derived from public RGB textures."""

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path[:0] = [str(V1),str(ROOT / "02_code/box2_initial_pose_quality_v2_20261002")]
from runtime import atomic_json, data_identity
import quality_engine
import quality_histogram
from native_quality_workspace import NativeQualityWorkspace
from exact_bayer_gray import BayerGrayEngine, BayerGrayWorkspace, compile_quality


def main():
    started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("output","image-root","library"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/bg001")
    output.mkdir(parents=True,exist_ok=False)
    baseline=importlib.import_module("initial_selection")
    sys.path.insert(0,str(baseline.TOOLS))
    from color_contract import demosaic_rggb8_to_bgr
    images=sorted(args.image_root.glob("*.png"))
    if len(images)!=360:
        raise ValueError("The complete original public RGB denominator changed")
    selected=[images[position] for position in np.linspace(0,359,32,dtype=int)]
    identities=[data_identity(path) for path in selected]
    paths=[Path(__file__),Path(__file__).with_name("exact_bayer_gray.py"),Path(__file__).with_name("native_bayer_gray.cpp"),
        Path(__file__).with_name("native_quality_workspace.py"),Path(quality_histogram.__file__),V1/"quality_engine.py",
        baseline.TOOLS/"color_contract.py",args.library.resolve(strict=True)]
    sources=[data_identity(path) for path in paths]
    snapshot=output/"source_snapshot"
    snapshot.mkdir()
    for position,path in enumerate(paths[:-1]):
        (snapshot/(str(position)+"_"+path.name)).write_bytes(path.read_bytes())
    process=psutil.Process()
    atomic_json(output/"PROCESS_IDENTITY.json",{"pid":process.pid,"creation_time":process.create_time(),"scope":"Windows"})
    atomic_json(output/"CONTRACT.json",{"input_images":identities,"complete_input_RGB_count":360,"selected_RGB_count":32,
        "fixed_filename_uniform_selection_no_quality_ranking":True,"derived_RGGB_from_actual_RGB_not_sensor_RAW":True,
        "no_GT_TLS_training_pose_or_external_performance_claim":True,"counted":False})
    old_threads=cv2.getNumThreads()
    cv2.setNumThreads(1)
    engine=BayerGrayEngine(args.library)
    fused,reference=BayerGrayWorkspace(engine),NativeQualityWorkspace()
    rows=[]
    state={"status":"running","accepted_count_contribution":0}
    try:
        fused_quality=compile_quality(quality_engine.compact_quality,fused,quality_histogram)
        reference_quality=compile_quality(quality_engine.compact_quality,reference,quality_histogram)
        for path in selected:
            image=cv2.imdecode(np.frombuffer(path.read_bytes(),dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
                raise ValueError("An original public RGB image could not be decoded")
            raw=np.empty(image.shape[:2],dtype=np.uint8)
            for y,x,channel in ((0,0,2),(0,1,1),(1,0,1),(1,1,0)):
                raw[y::2,x::2]=image[y::2,x::2,channel]
            immutable=raw.copy()
            bgr=demosaic_rggb8_to_bgr(raw)
            original_gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            actual_gray=fused.gray(raw)
            original_fields=reference_quality(bgr)
            actual_fields=fused_quality(raw)
            row={"image_name":path.name,"shape":list(raw.shape),"gray_bits_exact":np.array_equal(original_gray,actual_gray),
                "original_six_fields":original_fields,"candidate_six_fields":actual_fields,"six_fields_exact":original_fields==actual_fields,
                "derived_input_unchanged":np.array_equal(raw,immutable)}
            rows.append(row)
            if not all(row[name] for name in ("gray_bits_exact","six_fields_exact","derived_input_unchanged")):
                np.savez(output/"FIRST_FAILURE.npz",original_gray=original_gray,candidate_gray=actual_gray,raw=raw)
                raise ValueError("A complete public-texture derived Bayer gray or quality field changed")
        fused.close()
        reference.close()
        engine.close()
        checks={"all32_complete_derived_Bayer_gray_bits_and_six_fields_exact":len(rows)==32,
            "all_public_RGB_inputs_and_original_sources_unchanged":identities==[data_identity(path) for path in selected] and sources==[data_identity(path) for path in paths],
            "all_private_planes_and_DLL_closed":fused.receipt()["owned_plane_bytes_after_close"]==reference.receipt()["owned_plane_bytes_after_close"]==0
                and engine.receipt()["DLL_unloaded"] and engine.receipt()["active_calls"]==0}
        if not all(checks.values()):
            raise ValueError("A registered external derived-input contract failed")
        state.update(status="public_RGB_derived_Bayer_operator_check_passed_not_RAW_GT_or_whole_generalization",checks=checks,
            rows=rows,sources=sources,input_images=identities,engine=engine.receipt(),fused_workspace=fused.receipt(),reference_workspace=reference.receipt(),
            node_seconds=time.perf_counter()-started,counted=False)
        atomic_json(output/"RESULT.json",state)
        print({"status":state["status"],"cases":len(rows),"seconds":state["node_seconds"],"counted":False},flush=True)
    except BaseException as error:
        state.update(status="failed",rows=rows,error_type=type(error).__name__,error=str(error),seconds=time.perf_counter()-started)
        atomic_json(output/"FAILURE.json",state)
        raise
    finally:
        fused.close()
        reference.close()
        engine.close()
        cv2.setNumThreads(old_threads)
        atomic_json(output/"RESOURCE_CLOSE.json",{"original_threads_restored":cv2.getNumThreads()==old_threads,
            "DLL_and_planes_closed":engine.closed and fused.closed and reference.closed,"node_seconds":time.perf_counter()-started,"counted":False})


if __name__=="__main__":
    main()
