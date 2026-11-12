// Restore the original calibrated pixel grid whenever interaction stops.
export function createMotionResolution(renderer, projection, state) {
  const fullRatio = Math.min(devicePixelRatio, 1.5);
  const movingRatio = fullRatio * 0.75;
  let ratio = fullRatio;
  state.motionResolution = { fullRatio, movingRatio, currentRatio: ratio, moving: false, transitions: 0 };
  return {
    beforeFrame(moving) {
      const desired = moving ? movingRatio : fullRatio;
      if (desired !== ratio) {
        renderer.setPixelRatio(desired);
        projection.resize();
        ratio = desired;
        state.motionResolution.transitions++;
      }
      state.motionResolution.currentRatio = ratio;
      state.motionResolution.moving = Boolean(moving);
      state.motionResolution.physicalWidth = renderer.domElement.width;
      state.motionResolution.physicalHeight = renderer.domElement.height;
    },
  };
}
