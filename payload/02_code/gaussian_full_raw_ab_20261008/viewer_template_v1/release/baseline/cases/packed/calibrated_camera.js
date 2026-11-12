// Preserve calibrated capture rays while keeping the interactive camera independent.
export function createCameraProjection(camera, renderer, state) {
  let calibration = null;
  let lastWidth = -1, lastHeight = -1;
  function layout() {
    const canvas = renderer.domElement;
    let width = innerWidth, height = innerHeight;
    if (calibration) {
      const scale = Math.min(width / calibration.width, height / calibration.height);
      width = Math.floor(calibration.width * scale);
      height = Math.floor(calibration.height * scale);
    }
    if (width !== lastWidth || height !== lastHeight) {
      renderer.setSize(width, height);
      lastWidth = width;
      lastHeight = height;
    }
    canvas.style.position = "absolute";
    canvas.style.left = ((innerWidth - width) / 2) + "px";
    canvas.style.top = ((innerHeight - height) / 2) + "px";
    camera.aspect = width / height;
    if (calibration) {
      const { fx, fy, cx, cy, width: w, height: h } = calibration;
      const n = camera.near, f = camera.far;
      // Three.js camera axes are x-right, y-up, z-back; image coordinates are corner-based.
      camera.projectionMatrix.set(
        2 * fx / w, 0, 1 - 2 * cx / w, 0,
        0, 2 * fy / h, 2 * cy / h - 1, 0,
        0, 0, -(f + n) / (f - n), -2 * f * n / (f - n),
        0, 0, -1, 0);
      camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    } else {
      camera.updateProjectionMatrix();
    }
    state.projectionMode = calibration ? "calibrated_capture" : "free_camera";
    state.projectionMatrix = camera.projectionMatrix.toArray();
    state.calibration = calibration;
    state.viewport = { width, height, left: (innerWidth - width) / 2, top: (innerHeight - height) / 2 };
  }
  return {
    capture(view) {
      for (const key of ["fx", "fy", "cx", "cy", "width", "height"]) {
        if (!Number.isFinite(view[key])) throw new Error("Missing calibrated camera field: " + key);
      }
      if (Math.min(view.fx, view.fy, view.width, view.height) <= 0) throw new Error("Invalid camera calibration");
      calibration = Object.fromEntries(["fx", "fy", "cx", "cy", "width", "height"].map(key => [key, view[key]]));
      layout();
    },
    free() { calibration = null; layout(); },
    resize: layout,
  };
}
