"""Bind physical visibility constraints to existing native BA parameters."""

import numpy as np
import pyceres


class PositiveDepthCost(pyceres.CostFunction):
    """Use the native xyzw-plus-translation pose block without copying it."""

    def __init__(self, sensor_from_rig, near_m):
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([7, 3])
        self.normal = sensor_from_rig.rotation.matrix()[2].copy()
        self.offset = float(sensor_from_rig.translation[2])
        self.near_m = float(near_m)
        if not np.isfinite(self.near_m) or self.near_m <= 0:
            raise ValueError('A physical depth residual requires a finite positive renderer near plane')

    def value_and_jacobians(self, pose, xyz):
        vector, scalar = pose[:3], pose[3]
        inner = np.cross(vector, xyz)
        rotated = xyz + 2*scalar*inner + 2*np.cross(vector, inner)
        z = self.normal @ (rotated+pose[4:]) + self.offset
        if z >= self.near_m:
            return 0., np.zeros(7), np.zeros(3)
        rotation_jacobian = np.column_stack([
            2*scalar*np.cross(axis, xyz) + 2*(np.cross(axis, inner)
                + np.cross(vector, np.cross(axis, xyz))) for axis in np.eye(3)])
        rotation_jacobian = np.column_stack((rotation_jacobian, 2*inner))
        matrix = np.column_stack([axis+2*scalar*np.cross(vector, axis)
            + 2*np.cross(vector, np.cross(vector, axis)) for axis in np.eye(3)])
        derivative = -1/self.near_m
        pose_jacobian = derivative*np.r_[self.normal @ rotation_jacobian, self.normal]
        point_jacobian = derivative*(self.normal @ matrix)
        return float((self.near_m-z)/self.near_m), pose_jacobian, point_jacobian

    def Evaluate(self, parameters, residuals, jacobians):
        value, pose_jacobian, point_jacobian = self.value_and_jacobians(parameters[0], parameters[1])
        residuals[0] = value
        if jacobians is not None:
            if jacobians[0] is not None:
                jacobians[0][:] = pose_jacobian
            if jacobians[1] is not None:
                jacobians[1][:] = point_jacobian
        return bool(np.isfinite(value) and np.isfinite(pose_jacobian).all()
            and np.isfinite(point_jacobian).all())


def add_positive_depth_residuals(adjuster, model, point_ids, near_m):
    import pycolmap

    costs, rows = [], []
    before = adjuster.problem.num_residuals()
    for key in sorted(set(point_ids)):
        point = model.points3D[key]
        for entry in point.track.elements:
            image = model.images[entry.image_id]
            frame = model.frames[image.frame_id]
            sensor_id = image.data_id.sensor_id
            sensor_from_rig = (pycolmap.Rigid3d() if frame.rig.is_ref_sensor(sensor_id)
                else frame.rig.sensor_from_rig(sensor_id))
            pose = frame.rig_from_world.params
            if (not adjuster.problem.has_parameter_block(pose)
                    or adjuster.problem.parameter_block_size(pose) != 7
                    or not adjuster.problem.has_parameter_block(point.xyz)
                    or adjuster.problem.parameter_block_size(point.xyz) != 3):
                raise ValueError('A depth residual must share the actual native pose and point blocks')
            cost = PositiveDepthCost(sensor_from_rig, near_m)
            adjuster.problem.add_residual_block(cost, None, [pose, point.xyz])
            costs.append(cost)
            rows.append({'point3D_id': int(key), 'image_name': image.name,
                'point2D_index': int(entry.point2D_idx)})
    if adjuster.problem.num_residuals()-before != len(rows):
        raise ValueError('The native problem did not activate every intended physical residual')
    return costs, rows
