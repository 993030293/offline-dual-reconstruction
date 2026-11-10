"""Inject a caller failure after two real query deliveries and retain cleanup."""

import argparse
from contextlib import contextmanager
from pathlib import Path
import shutil
import sys

import localize_geometry_sift_fusion as entry
import feature_localization
import native_observation_index
import native_sift_process
from fusion_owned_json_write import ORIGINAL
from runtime import data_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--error",choices=("runtime","interrupt"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.resolve().relative_to(entry.OWNED.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    actor=data_identity(Path(__file__))
    shutil.copyfile(Path(__file__),args.output.parent/(args.output.name+"_DRIVER_SOURCE.py"))
    ORIGINAL(args.output.parent/(args.output.name+"_DRIVER_CONTRACT.json"),{"source":actor,"error":args.error,
        "registered_query_deliveries_before_failure":2,"new_optimization_count":0})
    originals=(native_sift_process.NativeSiftProcess.extract,entry.native_sift_process,entry.array_match_geometry,
        feature_localization.FeatureBank.__dict__.get("__setattr__"),native_observation_index.indexed_match,sys.argv)
    traces={}

    def fail(owner,gray):
        if owner.next_deliver==2:
            if args.error=="interrupt":
                raise KeyboardInterrupt("owned_complete_fusion_caller_failure")
            raise RuntimeError("owned_complete_fusion_caller_failure")
        return originals[0](owner,gray)

    @contextmanager
    def observe_sift(*positional,**keywords):
        with originals[1](*positional,**keywords) as trace:
            traces["sift"]=trace
            yield trace

    @contextmanager
    def observe_geometry(*positional,**keywords):
        with originals[2](*positional,**keywords) as trace:
            traces["geometry"]=trace
            yield trace

    expected=KeyboardInterrupt if args.error=="interrupt" else RuntimeError
    caught=False
    try:
        native_sift_process.NativeSiftProcess.extract=fail
        entry.native_sift_process=observe_sift
        entry.array_match_geometry=observe_geometry
        sys.argv=[str(Path(entry.__file__)),"--mode","process","--output",str(args.output)]
        try:
            entry.main()
        except expected as error:
            if str(error)!="owned_complete_fusion_caller_failure":
                raise
            caught=True
    finally:
        native_sift_process.NativeSiftProcess.extract=originals[0]
        entry.native_sift_process=originals[1]
        entry.array_match_geometry=originals[2]
        sys.argv=originals[5]
        checks={"registered_error_propagated":caught,"two_actual_query_results_retained":
            traces.get("sift",{}).get("broker",{}).get("delivered")==2,
            "actual_cpu_pool_joined":traces.get("sift",{}).get("broker",{}).get("closed",False),
            "both_original_hooks_restored":traces.get("sift",{}).get("binding_restored",False)
                and traces.get("geometry",{}).get("binding_restored",False),
            "all_private_injection_bindings_restored":native_sift_process.NativeSiftProcess.extract is originals[0]
                and entry.native_sift_process is originals[1] and entry.array_match_geometry is originals[2]
                and feature_localization.FeatureBank.__dict__.get("__setattr__") is originals[3]
                and native_observation_index.indexed_match is originals[4] and sys.argv is originals[5],
            "partial_entry_not_published_as_completed":not (args.output/"RESULT.json").exists(),
            "actual_injection_actor_source_unchanged":actor==data_identity(Path(__file__))}
        ORIGINAL(args.output/"FAULT_RESULT.json",{"status":"expected_failure_preserved" if all(checks.values()) else "failed_verification",
            "checks":checks,"error":args.error,"traces":traces,"new_optimization_count":0,
            "KeyboardInterrupt_is_injected_exception_not_OS_SIGINT":args.error=="interrupt"})
    if not all(checks.values()):
        raise AssertionError("A complete coupled failure or binding cleanup guard failed")
    raise SystemExit(86)


if __name__=="__main__":
    main()
