"""Separate exact SLERP agreement from the legacy small-angle approximation."""

import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

folder=Path("C:/lidar-camera-fusion-mask-robust-downsampling/04_outputs/box2_initial_pose_quality_v2_20261002T193102/01_initial_frames")
rows=json.loads((folder/"INITIAL_FRAME_MANIFEST.json").read_text(encoding="utf-8"))["selected_frames"]
raw=np.load(folder/"raw_body_odometry.npz")
contract=json.loads((folder/"RAW_SENSOR_CONTRACT.json").read_text(encoding="utf-8"))
transform=np.asarray(contract["T_body_camera"])
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"box2_lio_guided_sfm_v1_20261002"))
from common import BASE,PoseCurve
legacy=PoseCurve(BASE/"03_metric_context/metric_odometry.tum",BASE/"03_metric_context/T_base_link_camera_optical.json")
old_position=[]; old_angle=[]; exact_position=[]; exact_angle=[]
for row in rows:
    old=legacy.at(row["camera_header_ns"])
    actual_r=Rotation.from_matrix(row["R_WC"])
    old_position.append(float(np.linalg.norm(np.asarray(row["C_W_m"])-old["C_W_m"])))
    old_angle.append(float((Rotation.from_matrix(old["R_WC"]).inv()*actual_r).magnitude()))
    left,right=row["left_pose_index"],row["right_pose_index"]
    alpha=row["interpolation_alpha"]
    a=raw["pose_xyz_xyzw"][left,3:]; b=raw["pose_xyz_xyzw"][right,3:]
    if np.dot(a,b)<0: b=-b
    theta=np.arccos(np.clip(np.dot(a,b),-1.,1.))
    q=(1-alpha)*a+alpha*b if theta<1e-12 else (np.sin((1-alpha)*theta)*a+np.sin(alpha*theta)*b)/np.sin(theta)
    body_r=Rotation.from_quat(q)
    expected_r=body_r*Rotation.from_matrix(transform[:3,:3])
    body_t=(1-alpha)*raw["pose_xyz_xyzw"][left,:3]+alpha*raw["pose_xyz_xyzw"][right,:3]
    expected_t=body_t+body_r.apply(transform[:3,3])
    exact_position.append(float(np.linalg.norm(expected_t-row["C_W_m"])))
    exact_angle.append(float((expected_r.inv()*actual_r).magnitude()))
result={"status":"legacy_reference_equivalence_failed_exact_raw_SLerp_diagnostic",
    "legacy_max_position_delta_m":max(old_position),"legacy_max_rotation_delta_rad":max(old_angle),
    "legacy_worst_frame":rows[int(np.argmax(old_angle))]["image_name"],
    "direct_raw_closed_form_SLERP_max_position_delta_m":max(exact_position),
    "direct_raw_closed_form_SLERP_max_rotation_delta_rad":max(exact_angle),
    "legacy_comparison_threshold_m_and_rad":1e-8,"selection_or_data_modified":False,
    "raw_tum_position_delta":float(np.max(np.abs(raw["pose_xyz_xyzw"][:,:3]-legacy.positions))),
    "raw_tum_timestamp_delta_ns":int(np.max(np.abs(raw["timestamp_ns"]-legacy.times)))}
with (folder/"INTERPOLATION_FAILURE_DIAGNOSIS.json").open("x",encoding="utf-8") as stream:
    json.dump(result,stream,indent=2);stream.write("\n")
print(json.dumps(result,indent=2))
