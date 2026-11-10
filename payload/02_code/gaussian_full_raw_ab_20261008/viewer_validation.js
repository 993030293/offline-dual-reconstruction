// Publish read-only application telemetry to the DOM for local interaction checks.
const telemetry = document.createElement("pre");
telemetry.id = "viewer-validation";
telemetry.hidden = true;
document.body.appendChild(telemetry);
setInterval(() => {
  const state = window.box2Viewer;
  if (!state) return;
  telemetry.textContent = JSON.stringify({ ready: state.ready, frames: state.frames,
    position: state.getCameraPosition?.(), direction: state.getCameraDirection?.(),
    activeKeys: state.getActiveKeys?.(), sceneMode: state.sceneMode, navigationMode: state.navigationMode,
    playing: state.playing, currentView: state.currentView, loadedModels: state.loadedModels,
    modelLoadAttempts: state.modelLoadAttempts, fullGaussianCount: state.fullGaussianCount,
    boxGaussianCount: state.boxGaussianCount, errors: state.errors, loadFailures: state.loadFailures,
    motionResolution: state.motionResolution, renderPolicy: state.renderPolicy,
    pending: state.isRenderPending?.(), drawCalls: state.renderer?.info.render.calls,
    firstReadyRenderMs: state.firstReadyRenderMs,
    navigationToFirstReadyRenderMs: state.navigationToFirstReadyRenderMs,
    cacheReads: window.assetCacheTrace, integrityChecks: window.assetIntegrityTrace });
}, 200);
