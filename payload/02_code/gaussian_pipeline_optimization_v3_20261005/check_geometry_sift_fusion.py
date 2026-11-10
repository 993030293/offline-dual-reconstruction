"""Check coupled cache bytes, original geometry, RNG, and exception cleanup."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWNED = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/fusion_geometry_sift_001"
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from array_match_geometry import array_match_geometry
from native_sift_process import native_sift_process
import fusion_owned_json_write
from check_native_sift_process_guards import bank, arrays_equal
from check_array_match_geometry import prepare, invoke, equal, fixture
from observation_index import indexed_match
from query_feature_cache import QueryFeatureCache, file_identity
import feature_localization
import native_observation_index


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OWNED.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), CODE/"localize_geometry_sift_fusion.py", CODE/"geometry_sift_fusion_node.py",
        CODE/"array_match_geometry.py", CODE/"native_sift_process.py", CODE/"check_array_match_geometry.py",
        CODE/"check_native_sift_process_guards.py", CODE/"fusion_owned_json_write.py", CODE/"owned_json_write.py",
        CODE/"check_owned_json_write.py", Path(feature_localization.__file__),
        ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005/query_feature_cache.py"]
    identities = [data_identity(p) for p in paths]
    for i, path in enumerate(paths):
        shutil.copyfile(path, args.output/f"SOURCE_{i:02d}_{path.name}")
    cv2.setNumThreads(8)
    generator = np.random.default_rng(20261006)
    photos = []
    for i, (width, height, kind) in enumerate(((91,73,"constant"),(320,257,"thin"),
            (640,360,"random"),(257,511,"translated"))):
        pixels = generator.integers(0,256,(height,width,3),dtype=np.uint8)
        if kind == "constant":
            pixels[:] = 127
        elif kind == "thin":
            pixels[:] = 0
            x = np.arange(width)
            y = np.rint(height*.5+height*.3*np.sin(x*.047)).astype(int)
            pixels[y,x] = 255
        elif kind == "translated":
            pixels = np.roll(pixels,(7,-13),(0,1))
        photo = args.output/f"image{i}.png"
        Image.fromarray(pixels).save(photo)
        photos.append(photo)
    extra = feature_localization.FeatureBank.extra
    setattr_original = feature_localization.FeatureBank.__dict__.get("__setattr__")
    geometry_original = native_observation_index.indexed_match
    baseline_bank = bank(args.output/"b")
    baseline_cache = QueryFeatureCache(args.output/"bc", extra)
    expected = [baseline_cache(baseline_bank,p) for p in photos]
    records, checks, geometry_rows = [], {}, []
    with array_match_geometry() as geometry_trace, native_sift_process(photos) as sift_trace:
        current_bank = bank(args.output/"c")
        current_cache = QueryFeatureCache(args.output/"cc", feature_localization.FeatureBank.extra)
        current = []
        for ordinal, photo in enumerate(photos):
            current.append(current_cache(current_bank,photo))
            values = list(fixture((0,7,15,89)[ordinal],(0.01,1,100,1)[ordinal],1,
                ("curved","thin","irregular","curved")[ordinal],3))
            payload = prepare(values)
            old = invoke(indexed_match,values,True,payload)
            new = invoke(native_observation_index.indexed_match,values,True,payload)
            geometry_rows.append({"ordinal":ordinal,"complete_outputs_and_RNG_exact":equal(old,new)})
        checks["four_native_feature_arrays_exact"] = all(arrays_equal(a,b) for a,b in zip(expected,current))
        checks["four_original_NPZ_bytes_exact"] = all(file_identity(baseline_bank.cache/(p.stem+".npz"))
            == file_identity(current_bank.cache/(p.stem+".npz")) for p in photos)
        checks["original_cache_extractor_identity_exact"] = baseline_cache.extractor_identity == current_cache.extractor_identity
        checks["four_original_cache_metadata_exact"] = sorted(p.name for p in baseline_cache.root.iterdir()) == sorted(
            p.name for p in current_cache.root.iterdir()) and all(json.loads((p/"ENTRY.json").read_text())
            == json.loads((current_cache.root/p.name/"ENTRY.json").read_text()) for p in baseline_cache.root.iterdir())
        checks["geometry_with_live_SIFT_pool_all_four_oracles_exact"] = all(r["complete_outputs_and_RNG_exact"] for r in geometry_rows)
        checks["all_four_actual_prepared_array_routes"] = geometry_trace["routes"] == ["owned_exact_array"]*4
    records.append({"sift":sift_trace,"geometry":geometry_trace})
    checks["normal_coupled_hooks_restored_and_pool_joined"] = geometry_trace["binding_restored"] and sift_trace["binding_restored"] and sift_trace["broker"]["closed"]
    for error_type in (RuntimeError,KeyboardInterrupt):
        try:
            with array_match_geometry() as failure_geometry, native_sift_process(photos) as failure_sift:
                failure_bank = bank(args.output/error_type.__name__)
                first = failure_bank.extractor.extract(cv2.imread(str(photos[0]),cv2.IMREAD_GRAYSCALE))
                if not arrays_equal(first,expected[0]):
                    raise AssertionError("The original completed feature prefix changed")
                raise error_type("owned_coupled_caller_failure")
        except error_type as error:
            checks[error_type.__name__+"_propagates_both_hooks_restored_pool_joined"] = (
                str(error)=="owned_coupled_caller_failure" and failure_geometry["binding_restored"]
                and failure_sift["binding_restored"] and failure_sift["broker"]["closed"])
        records.append({"sift":failure_sift,"geometry":failure_geometry})
    checks["original_feature_callback_and_both_bindings_restored"] = (
        feature_localization.FeatureBank.extra is extra and feature_localization.FeatureBank.__dict__.get("__setattr__") is setattr_original
        and native_observation_index.indexed_match is geometry_original)
    for rejected in [ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sp001",ROOT/"01_raw_data"]:
        try:
            fusion_owned_json_write.OwnedAtomicJSON(rejected)
            raise AssertionError("An unrelated write root was admitted")
        except ValueError:
            pass
    checks["original_and_raw_write_roots_rejected_by_fusion_constructor"] = True
    checker_path=CODE/"check_owned_json_write.py"
    checker=checker_path.read_text(encoding="utf-8")
    if checker.count(fusion_owned_json_write.OLD)!=1:
        raise ValueError("The original complete write regression scope changed")
    composed=checker.replace(fusion_owned_json_write.OLD,fusion_owned_json_write.NEW)
    (args.output/"COMPOSED_WRITE_CHECK_SOURCE.py").write_text(composed,encoding="utf-8")
    (args.output/"COMPOSED_OWNED_WRITE_SOURCE.py").write_text(fusion_owned_json_write.COMPOSED_SOURCE,encoding="utf-8")
    namespace={"__file__":str(checker_path),"__name__":"fusion_private_write_check"}
    exec(compile(composed,str(checker_path),"exec"),namespace)
    namespace.update(OwnedAtomicJSON=fusion_owned_json_write.OwnedAtomicJSON,
        owned_atomic_json=fusion_owned_json_write.owned_atomic_json,ORIGINAL=fusion_owned_json_write.ORIGINAL)
    argv=sys.argv
    try:
        sys.argv=[str(checker_path),"--output",str(args.output/"w")]
        namespace["main"]()
    finally:
        sys.argv=argv
    writer=json.loads((args.output/"w/RESULT.json").read_text(encoding="utf-8"))
    checks["all_eight_original_write_recovery_scope_denial_and_restore_guards"] = (
        writer["status"]=="passed" and len(writer["checks"])==8 and all(writer["checks"].values()))
    checks["all_sources_unchanged"] = identities == [data_identity(p) for p in paths]
    result = {"status":"passed" if all(checks.values()) else "failed_verification", "checks":checks,
        "sources":identities,"geometry_rows":geometry_rows,"records":records,"writer_checks":writer["checks"],
        "complete_child_seconds":time.perf_counter()-started,"new_optimization_count":0,
        "scope":"CPU_feature_and_native_BF_prepared_geometry_component_not_live_GPU_or_GT"}
    atomic_json(args.output/"RESULT.json",result)
    print(json.dumps({"status":result["status"],"checks":checks}),flush=True)
    if not all(checks.values()):
        raise AssertionError("A coupled feature or geometry contract failed")


if __name__ == "__main__":
    main()
