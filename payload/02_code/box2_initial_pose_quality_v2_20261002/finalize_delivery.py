"""Consolidate verified stage receipts and snapshot the versioned implementation."""

import json
import shutil
import time
from pathlib import Path


RUN=Path("C:/lidar-camera-fusion-mask-robust-downsampling/04_outputs/box2_initial_pose_quality_v2_20261002T193102")
CODE=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def update(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


selection=read(RUN/"01_initial_frames/SELECTION_RECEIPT.json")
bundle=read(RUN/"01_initial_frames/BUNDLE_VALIDATION.json")
browser=read(RUN/"viewer/BROWSER_VALIDATION.json")
visual=read(RUN/"viewer/VISUAL_REVIEW.json")
active=read(RUN/"viewer/ACTIVE_ENDPOINT_VALIDATION.json")
gt=read(RUN/"02_public_gt/verified/GT_TRAJECTORY_RECEIPT.json")
preflight=read(RUN/"02_public_gt/verified/GT_FRAME_TIME_PREFLIGHT.json")
if bundle["status"]!="passed_engineering_bundle_verification" or browser["status"]!="passed" or visual["status"]!="reviewed" or active["status"]!="passed_active_endpoint":
    raise ValueError("A required delivery check has not completed")
public=read(RUN/"02_public_gt/verified/LOCAL_GT_AUDIT.json")
public.update(status="confirmed_geometry_trajectory_frame_time_inputs_not_accuracy_scored",
    trajectory_receipt=str(RUN/"02_public_gt/verified/GT_TRAJECTORY_RECEIPT.json"),
    frame_time_receipt=str(RUN/"02_public_gt/verified/GT_FRAME_TIME_PREFLIGHT.json"),
    trajectory_and_frame_time_verified=True)
update(RUN/"02_public_gt/verified/LOCAL_GT_AUDIT.json",public)
viewer=read(RUN/"viewer/DISPLAY_EXPORT_RECEIPT.json")
for item in viewer["cases"]:
    item.update(read(RUN/"viewer/cases"/item["case"]/"display_manifest.json"))
viewer.update(status="model_assets_verified_browser_and_visual_review_passed", active_endpoint=active["url"], model_retrained=False)
update(RUN/"viewer/DISPLAY_EXPORT_RECEIPT.json",viewer)
handoff=read(RUN/"viewer/SERVICE_HANDOFF.json")
server=read(RUN/"viewer/SERVER_STATE_8195.json")
handoff.update(status="HTTP_and_active_browser_verified", active_server_pid=server["pid"], active_endpoint=active["url"])
update(RUN/"viewer/SERVICE_HANDOFF.json",handoff)
repair=read(RUN/"01_initial_frames/INTERPOLATION_COMPARISON_REPAIR.json")
repair["status"]="comparison_reference_corrected_bundle_recheck_passed"
update(RUN/"01_initial_frames/INTERPOLATION_COMPARISON_REPAIR.json",repair)
snapshot=RUN/"source_snapshot"
snapshot.mkdir(exist_ok=False)
for path in CODE.iterdir():
    if path.is_file() and path.suffix in {".py",".js",".html",".ps1",".md"}:
        shutil.copy2(path,snapshot/path.name)
receipt={"status":"requested_preview_initial_frame_bundle_and_public_GT_intake_delivered",
    "selected_frame_count":selection["selected_count"],"raw_frame_count":selection["raw_frame_count"],
    "preview_url":active["url"],"preview_server_pid":server["pid"],"displayed_model_frame_count":212,
    "new_frame_bundle_verified":True,"new_frame_bundle_SfM_run":False,"new_Gaussian_training_run":False,
    "public_GT_rows":gt["rows"],"public_GT_pose_frame_time_preflight":preflight["status"],
    "public_GT_accuracy_scored":False,"Box2_independent_GT":False,
    "quality_time_gaps_preserved":selection["unresolved_quality_supported_time_gaps"],
    "source_snapshot":str(snapshot),"review_type":"self_review", "completed_unix":time.time(),
    "claim_boundary":"Completed the requested controls, initial frame selection and GT availability check; precision experiment pending"}
with (RUN/"DELIVERY_RECEIPT.json").open("x",encoding="utf-8") as stream:
    json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
print(json.dumps(receipt,ensure_ascii=False))
