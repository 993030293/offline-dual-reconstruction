"""Exercise pose direction, selection exclusions, and quality-gap behavior."""

import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from initial_selection import CoarseCameraCurve, POLICY, choose, static_transform


def row(i, x, sharpness=30., excluded=None):
    return {"camera_frame": i, "camera_header_ns": i * 100_000_000, "C_W_m": [x,0.,0.],
            "R_WC": np.eye(3).tolist(), "quaternion_WC_xyzw": [0.,0.,0.,1.], "coarse_pose_valid": True,
            "excluded_role": excluded, "quality": {"sharpness": sharpness,"clipped_fraction":.02,
            "mean_brightness":100.,"corner_count_half_resolution":200,"occupied_grid_cells_4x4":12,"quality_score":sharpness}}


class SelectionContracts(unittest.TestCase):
    def test_small_angle_is_constant_angular_velocity_not_linear_quaternion_approximation(self):
        poses=np.array([[0,0,0,0,0,0,1],[0,0,0,*Rotation.from_euler("z",3,degrees=True).as_quat()]])
        curve=CoarseCameraCurve([0,100_000_000],poses,np.eye(4))
        actual=Rotation.from_matrix(curve.at(17_000_000)["R_WC"])
        expected=Rotation.from_euler("z",.51,degrees=True)
        self.assertLess((expected.inv()*actual).magnitude(),1e-10)

    def test_body_rotation_rotates_camera_lever_arm(self):
        transform=np.eye(4); transform[0,3]=1
        poses=np.array([[0,0,0,0,0,0,1],[0,0,0,*Rotation.from_euler("z",90,degrees=True).as_quat()]])
        curve=CoarseCameraCurve([0,100_000_000],poses,transform)
        middle=curve.at(50_000_000)
        np.testing.assert_allclose(middle["C_W_m"],[np.sqrt(.5),np.sqrt(.5),0],atol=1e-10)
        with self.assertRaises(ValueError): curve.at(-1)
        with self.assertRaises(ValueError): curve.at(50_000_000,maximum_span_ns=10)

    def test_raw_static_chain_direction(self):
        transforms=[{"parent":"body","child":"mount","translation":[1,0,0],"rotation_xyzw":[0,0,0,1]},
                    {"parent":"mount","child":"optical","translation":[0,2,0],"rotation_xyzw":[0,0,0,1]}]
        matrix,path=static_transform(transforms,"optical","body")
        np.testing.assert_allclose(matrix[:3,3],[1,2,0]); self.assertEqual(path,["optical","mount","body"])

    def test_keep_new_motion_and_exclude_development_images(self):
        rows=[row(i,i*.06,excluded="H" if i==10 else None) for i in range(40)]
        selected,_=choose(rows,POLICY,20.)
        ids={r["camera_frame"] for r in selected}
        self.assertNotIn(10,ids); self.assertGreater(len(ids),15)
        self.assertGreater(max(r["C_W_m"][0] for r in selected),2.2)

    def test_quality_gap_never_relaxes_gate(self):
        rows=[row(i,i*.03,sharpness=1. if 10<=i<=30 else 30.) for i in range(41)]
        selected,_=choose(rows,POLICY,20.)
        self.assertTrue(all(not 10<=r["camera_frame"]<=30 for r in selected))
        self.assertGreater(max(b["camera_header_ns"]-a["camera_header_ns"] for a,b in zip(selected,selected[1:])),800_000_000)

    def test_stationary_interval_has_no_forced_frame_quota(self):
        selected,_=choose([row(i,0.) for i in range(30)],POLICY,20.)
        self.assertLessEqual(len(selected),5)

    def test_cap_is_reportable_and_not_overrun(self):
        policy=copy.deepcopy(POLICY); policy["maximum_exported_frames"]=6
        selected,_=choose([row(i,i*.2) for i in range(30)],policy,20.)
        self.assertEqual(len(selected),6)


if __name__ == "__main__":
    unittest.main()
