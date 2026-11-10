"""Verify exported images, pose association, calibration, and matching candidates."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from initial_selection import TOOLS, RAW, CAMERA_TOPIC


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(folder):
    manifest=json.loads((folder/"INITIAL_FRAME_MANIFEST.json").read_text(encoding="utf-8"))
    rows=manifest["selected_frames"]
    contract=json.loads((folder/"RAW_SENSOR_CONTRACT.json").read_text(encoding="utf-8"))
    exclusions=json.loads((folder/"EXCLUDED_EVALUATION_IDS.json").read_text(encoding="utf-8"))["frames"]
    images=folder/"images"
    names={row["image_name"] for row in rows}
    ids={row["camera_frame"] for row in rows}
    if len(names)!=len(rows) or len(ids)!=len(rows) or names!={path.name for path in images.iterdir()}:
        raise ValueError("Exported frame identities, count, or file set differs")
    if ids & {int(value) for value in exclusions}:
        raise ValueError("A development evaluation image entered the initial set")
    for row in rows:
        path=images/row["image_name"]
        if digest(path)!=row["image_sha256"]:
            raise ValueError("Exported image data identity differs")
        image=cv2.imread(str(path))
        if image is None or image.shape[:2]!=(1240,1624):
            raise ValueError("An exported image cannot be decoded at raw resolution")
        if row["image_name"]!=f"frame_{row['camera_frame']:05d}_{row['camera_header_ns']}.jpg":
            raise ValueError("An image name does not match its sensor timestamp")
    pair_set=set()
    adjacency={name:set() for name in names}
    for line in (folder/"initial_matching_pairs.txt").read_text(encoding="utf-8").splitlines():
        a,b=line.split()
        if a not in names or b not in names or a==b or (a,b) in pair_set or (b,a) in pair_set:
            raise ValueError("An initial matching pair is invalid or duplicated")
        pair_set.add((a,b)); adjacency[a].add(b); adjacency[b].add(a)
    reached={rows[0]["image_name"]}
    queue=list(reached)
    for name in queue:
        for neighbor in adjacency[name]-reached:
            reached.add(neighbor); queue.append(neighbor)
    if reached!=names:
        raise ValueError("The proposed matching graph is disconnected")
    sys.path.insert(0,str(TOOLS))
    from mcap_audit import McapIndex, decode_camera_info
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr
    index=McapIndex(RAW)
    first=contract["camera_info"]
    info_count=0
    for message in index.iter_messages("/sensing/camera/camera_info"):
        info=decode_camera_info(message["payload"])
        for key in ("frame_id","height","width","distortion_model","distortion","camera_matrix"):
            if info[key]!=first[key]:
                raise ValueError("Camera calibration changes during the recording")
        info_count+=1
    selected_samples={rows[int(i)]["camera_frame"]:rows[int(i)] for i in np.linspace(0,len(rows)-1,16,dtype=int)}
    for frame,message in index.iter_selected_messages(CAMERA_TOPIC,selected_samples):
        stamp,_,raw=decode_bayer(message["payload"])
        row=selected_samples[frame]
        if stamp!=row["camera_header_ns"]:
            raise ValueError("Sampled raw frame timestamp differs")
        ok,encoded=cv2.imencode(".jpg",demosaic_rggb8_to_bgr(raw),[cv2.IMWRITE_JPEG_QUALITY,97])
        if not ok or hashlib.sha256(encoded.tobytes()).hexdigest()!=row["image_sha256"]:
            raise ValueError("Sampled export differs from the raw Bayer colour contract")
    # Use the closed-form spherical interpolation, not the legacy small-angle NLERP approximation.
    raw_poses=np.load(folder/"raw_body_odometry.npz")
    stamps=raw_poses["timestamp_ns"]
    poses=raw_poses["pose_xyz_xyzw"]
    transform=np.asarray(contract["T_body_camera"])
    position_errors=[]; angle_errors=[]
    for row in rows:
        stamp=row["camera_header_ns"]
        right=int(np.searchsorted(stamps,stamp))
        left=right if stamps[right]==stamp else right-1
        span=int(stamps[right]-stamps[left])
        alpha=0. if span==0 else (stamp-int(stamps[left]))/span
        if (left,right)!=(row["left_pose_index"],row["right_pose_index"]) or abs(alpha-row["interpolation_alpha"])>1e-12:
            raise ValueError("The exported interpolation bracket differs from raw timestamps")
        a=poses[left,3:]/np.linalg.norm(poses[left,3:])
        b=poses[right,3:]/np.linalg.norm(poses[right,3:])
        if np.dot(a,b)<0: b=-b
        theta=np.arccos(np.clip(np.dot(a,b),-1.,1.))
        quaternion=(1-alpha)*a+alpha*b if theta<1e-12 else (np.sin((1-alpha)*theta)*a+np.sin(alpha*theta)*b)/np.sin(theta)
        body_rotation=Rotation.from_quat(quaternion)
        expected_rotation=body_rotation*Rotation.from_matrix(transform[:3,:3])
        expected_center=(1-alpha)*poses[left,:3]+alpha*poses[right,:3]+body_rotation.apply(transform[:3,3])
        position_errors.append(np.linalg.norm(np.asarray(row["C_W_m"])-expected_center))
        delta=expected_rotation.inv()*Rotation.from_matrix(row["R_WC"])
        angle_errors.append(float(delta.magnitude()))
    if max(position_errors)>1e-8 or max(angle_errors)>1e-8:
        raise ValueError("The exported camera association disagrees with closed-form raw SLERP")
    receipt={"status":"passed_engineering_bundle_verification", "all_images_decoded":len(rows),
        "all_exported_image_hashes_verified":len(rows),"raw_image_round_trip_samples":len(selected_samples),
        "all_raw_camera_info_messages_verified":info_count,"matching_candidate_pairs":len(pair_set),
        "candidate_graph_connected":True,"feature_match_graph_validated":False,"excluded_overlap":0,
        "raw_pose_cross_check_count":len(rows),"maximum_camera_center_delta_m":float(max(position_errors)),
        "maximum_camera_rotation_delta_rad":float(max(angle_errors)),
        "cross_check_reference_role":"closed_form_SLERP_from_raw_odometry_and_raw_tf_no_sfm",
        "legacy_approximation_failure_preserved":"INTERPOLATION_FAILURE_DIAGNOSIS.json",
        "GT_used_for_selection":False,"SfM_run":False,"boundary":"Image bundle and association correctness, not pose accuracy or triangulation success"}
    with (folder/"BUNDLE_VALIDATION.json").open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False); stream.write("\n")
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,required=True)
    args=parser.parse_args()
    main(args.folder.resolve(strict=True))
