"""Verify Oxford base/camera/IMU conversions and reference time support."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from public_gt_audit import ROOT, DATA


def rigid(matrix):
    matrix=np.asarray(matrix,dtype=float)
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3],[0,0,0,1],atol=1e-10):
        raise ValueError("Invalid homogeneous transform")
    if not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-6) or abs(np.linalg.det(matrix[:3,:3])-1)>1e-6:
        raise ValueError("Extrinsic rotation is not SO(3)")
    return matrix


def main(folder):
    calibration=DATA/"calibration"
    text=(calibration/"README.md").read_text(encoding="utf-8")
    documented="T_base_lidar_t_xyz_q_xyzw = [0.0, 0.0, 0.124, 0.0, 0.0, 1.0, 0.0]"
    if documented not in text:
        raise ValueError("The official base-frame convention has changed")
    values=yaml.safe_load((calibration/"cam-lidar-imu.yaml").read_text(encoding="utf-8"))
    t_base_lidar=np.eye(4)
    t_base_lidar[:3,:3]=Rotation.from_quat([0,0,1,0]).as_matrix()
    t_base_lidar[:3,3]=[0,0,.124]
    transforms={}
    for name,item in values.items():
        t_camera_lidar=rigid(item["T_cam_lidar"])
        if not np.allclose(t_camera_lidar[:3,:3],Rotation.from_quat(item["C_q_CL"]).as_matrix(),atol=1e-6):
            raise ValueError("Matrix and quaternion camera-LiDAR extrinsics disagree")
        t_base_camera=rigid(t_base_lidar@np.linalg.inv(t_camera_lidar))
        transforms[name]={"T_base_camera":t_base_camera.tolist(),"T_camera_base":np.linalg.inv(t_base_camera).tolist(),"topic":item["rostopic"]}
    cam0=values["cam0"]
    t_cam0_imu=np.eye(4)
    t_cam0_imu[:3,:3]=Rotation.from_quat(cam0["C_q_CI"]).as_matrix()
    t_cam0_imu[:3,3]=cam0["C_r_CI"]
    t_imu_lidar=rigid(np.linalg.inv(t_cam0_imu)@np.asarray(cam0["T_cam_lidar"]))
    t_imu_base=rigid(t_imu_lidar@np.linalg.inv(t_base_lidar))
    lio_folder=ROOT/"04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/00_raw_lio"
    lio_receipt=json.loads((lio_folder/"raw_lio_receipt.json").read_text(encoding="utf-8"))
    if lio_receipt["estimated_pose_source_frame"]!="imu_sensor_frame" or lio_receipt["gt_used"]:
        raise ValueError("The reusable raw LIO is not an independent IMU-frame estimate")
    with (lio_folder/"estimated_motion_states.csv").open(encoding="utf-8") as stream:
        motion=list(csv.DictReader(stream))
    recorded=np.eye(4)
    recorded[:3,3]=[float(motion[0][name]) for name in ("tx","ty","tz")]
    recorded[:3,:3]=np.asarray([float(motion[0][f"r{i}{j}"]) for i in range(3) for j in range(3)]).reshape(3,3)
    extrinsic_delta=float(np.max(np.abs(t_imu_lidar-recorded)))
    if extrinsic_delta>1e-6:
        raise ValueError("The estimated LIO body extrinsic differs from official calibration")
    gt=np.loadtxt(folder/"gt-tum.txt")
    reference_stamps=np.rint(gt[:,0]*1e9).astype(np.int64)
    stamps=np.asarray([int(row["timestamp_ns"]) for row in motion],dtype=np.int64)
    valid=[]; gaps=[]
    for stamp in stamps:
        if stamp<reference_stamps[0] or stamp>reference_stamps[-1]:
            valid.append(False); continue
        right=int(np.searchsorted(reference_stamps,stamp))
        left=right if reference_stamps[right]==stamp else right-1
        span=int(reference_stamps[right]-reference_stamps[left])
        valid.append(span<=200_000_000); gaps.append(span)
    result={"status":"frame_and_time_contract_verified_ready_for_evaluation_not_scored",
        "gt_rows":len(gt),"raw_LIO_rows":len(stamps),"raw_LIO_timestamps_with_GT_support":sum(valid),
        "outside_GT_or_gap_policy":len(valid)-sum(valid),"maximum_associated_GT_bracket_ns":max(gaps),
        "T_base_lidar":t_base_lidar.tolist(),"base_transform_documented_source":str(calibration/"README.md"),
        "camera_extrinsics":transforms,"T_imu_base":t_imu_base.tolist(),
        "T_imu_lidar_logged_vs_official_max_delta":extrinsic_delta,
        "comparison_conventions":{"estimated_base":"T_world_lio_imu @ T_imu_base",
            "GT_camera":"T_world_GT_base @ T_base_camera","gauge_alignment":"SE3 in evaluation for metric trajectories; disclose full-fit vs initial-segment alignment",
            "time_rule":"sensor headers, offset zero, no extrapolation, maximum GT bracket 0.20 s"},
        "pose_accuracy_metrics_computed":False,"geometry_metrics_computed":False,
        "GT_allowed_for_frame_selection_or_map_optimization":False,
        "no_independent_Blind_Box2_test":True,
        "scientific_boundary":"Preflight proves compatible data/frame/time inputs, not algorithm accuracy or improvement"}
    with (folder/"GT_FRAME_TIME_PREFLIGHT.json").open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({key:value for key,value in result.items() if key not in {"camera_extrinsics","T_imu_base","T_base_lidar"}},ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,required=True)
    args=parser.parse_args()
    main(args.folder.resolve(strict=True))
