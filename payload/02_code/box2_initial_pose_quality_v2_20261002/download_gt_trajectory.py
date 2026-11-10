"""Fetch and validate the pinned Oxford GT trajectory for evaluation only."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from public_gt_audit import REVISION, SEQUENCE


def main(folder):
    listing=json.loads((folder/"OFFICIAL_PROCESSED_TREE.json").read_text(encoding="utf-8"))
    relative=f"sequences/{SEQUENCE}/processed/trajectory/gt-tum.txt"
    item=next(value for value in listing["items"] if value["path"]==relative)
    if item["size"]>5_000_000:
        raise ValueError("The requested trajectory is larger than the evaluation fetch limit")
    url=f"https://huggingface.co/datasets/ori-drs/oxford_spires_dataset/resolve/{REVISION}/{relative}?download=true"
    with urlopen(Request(url,headers={"User-Agent":"Box2-GT-audit/2"}),timeout=25) as response:
        body=response.read(5_000_001)
        resolved=response.geturl()
    if len(body)!=item["size"]:
        raise ValueError("Official trajectory byte count differs from pinned metadata")
    blob=hashlib.sha1(b"blob "+str(len(body)).encode("ascii")+b"\0"+body).hexdigest()
    if blob!=item["oid"]:
        raise ValueError("Official trajectory data identity differs from pinned metadata")
    data=np.loadtxt(io.BytesIO(body))
    if data.ndim!=2 or data.shape[1]!=8 or not np.isfinite(data).all():
        raise ValueError("GT trajectory has an invalid TUM numeric layout")
    delta=np.diff(data[:,0])
    norms=np.linalg.norm(data[:,4:],axis=1)
    if np.any(delta<=0) or np.max(np.abs(norms-1))>1e-4:
        raise ValueError("GT time ordering or quaternion normalization is invalid")
    path=folder/"gt-tum.txt"
    with path.open("xb") as stream:
        stream.write(body)
    receipt={"status":"downloaded_and_numeric_contract_verified_evaluation_only","path":str(path),
        "source":url,"resolved_source":resolved,"revision":REVISION,"sequence":SEQUENCE,
        "bytes":len(body),"official_git_blob_oid":item["oid"],"actual_git_blob_oid":blob,
        "data_sha256":hashlib.sha256(body).hexdigest(),"rows":len(data),"columns":"t_s tx ty tz qx qy qz qw",
        "timestamp_range_s":[float(data[0,0]),float(data[-1,0])],"max_gap_s":float(delta.max()),
        "position_bounds_m":[data[:,1:4].min(0).tolist(),data[:,1:4].max(0).tolist()],
        "maximum_quaternion_norm_error":float(np.max(np.abs(norms-1))),
        "official_pose_frame":"base_frame_in_world", "base_to_camera_mapping_verified":False,
        "GT_role":"evaluation_only_reference_trajectory","used_for_Box2_selection":False,
        "ATE_RPE_computed":False,"limits":"Matching timestamps are necessary but do not prove frame/extrinsic equivalence"}
    with (folder/"GT_TRAJECTORY_RECEIPT.json").open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write("\n")
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,required=True)
    args=parser.parse_args()
    main(args.folder.resolve(strict=True))
