// Coalesce application and asynchronous splat updates into one browser frame.
export function createFrameScheduler() {
  let scheduled = false;
  let step = null;
  function request() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(time => {
      scheduled = false;
      if (step) step(time);
    });
  }
  return {
    request,
    start(callback) {
      if (step) throw new Error("Render scheduler is already started");
      step = callback;
      request();
    },
    isPending: () => scheduled,
  };
}
