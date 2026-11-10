"""Check principal-point, resizing, raster projection, and camera-z unit contracts."""

from types import SimpleNamespace
import numpy as np

from pgsr_calibration import scaled_intrinsics, projection_matrix, camera_K


def main():
    source = (1376.35822, 1378.02727, 812.004123, 635.131697)
    width, height = 812, 620
    params = scaled_intrinsics(source, (1624, 1240), (width, height))
    camera = SimpleNamespace(corner_K=params, image_width=width, image_height=height)
    matrix = camera_K(camera)
    pixels = np.array([[0., 0.], [811., 619.], [399., 267.], [6., 611.]])
    depths = np.array([.25, 3.5, 1.1, 17.])
    rays = np.column_stack(((pixels[:, 0] - matrix[0, 2]) / matrix[0, 0],
                            (pixels[:, 1] - matrix[1, 2]) / matrix[1, 1], np.ones(4)))
    xyz = rays * depths[:, None]
    recovered = xyz @ matrix.T
    recovered = recovered[:, :2] / recovered[:, 2, None]
    ray_error = float(np.max(np.abs(recovered - pixels)))
    projection = projection_matrix(params, width, height, .01, 100.)
    clip = np.column_stack((xyz, np.ones(4))) @ projection.T
    ndc = clip[:, :2] / clip[:, 3, None]
    raster_pixels = (ndc + 1) * np.array([width, height]) / 2 - .5
    raster_error = float(np.max(np.abs(raster_pixels - pixels)))
    expected = scaled_intrinsics(params, (width, height), (270, 206))
    scaled = camera_K(camera, 3.0)
    assert np.allclose(scaled[[0, 1], [0, 1]], expected[:2], atol=2e-5)
    assert np.allclose(scaled[[0, 1], [2, 2]], np.array(expected[2:]) - .5, atol=2e-5)
    assert np.allclose(xyz[:, 2], depths)
    assert ray_error < .0002 and raster_error < .0002
    assert abs(matrix[1, 2] - (height / 2 - .5)) > 5
    print({"status": "calibration_math_verified", "ray_error_px": ray_error,
           "raster_error_px": raster_error, "noncentral_principal_point_tested": True,
           "noninteger_resize_tested": True, "camera_z_depth_tested": True})


if __name__ == "__main__":
    main()
