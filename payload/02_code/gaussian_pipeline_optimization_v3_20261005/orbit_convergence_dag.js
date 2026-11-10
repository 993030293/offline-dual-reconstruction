// Finish exact idle controller updates before generating view-dependent Gaussian data.
export function createOrbitConvergenceDag({ controls, camera, state, maximumSteps = 4096 }) {
  if (!Number.isInteger(maximumSteps) || maximumSteps < 1 || maximumSteps > 4096) throw new Error("The registered CPU convergence bound changed");
  const nativeUpdate = controls.update;
  let pending = false, disposed = false;
  const totals = { marked: 0, converged: 0, structuralUnsupported: 0, cycleUnsupported: 0, boundUnsupported: 0,
    shadowUpdates: 0, replayUpdates: 0, preflightMs: 0, replayMs: 0 };
  const signature = (controller, object) => {
    const values = [...object.position.toArray(), ...object.quaternion.toArray(), ...controller.target.toArray()];
    if (!values.every(Number.isFinite)) throw new Error("The original controller contains non-finite pose values");
    return JSON.stringify(values.map(value => Object.is(value, -0) ? "-0" : value));
  };
  function eligible() {
    return state.navigationMode === "orbit" && !state.playing && controls.state === -1 && !controls.autoRotate &&
      controls._sphericalDelta.radius === 0 && controls._sphericalDelta.phi === 0 && controls._sphericalDelta.theta === 0 &&
      controls._panOffset.lengthSq() === 0 && controls._scale === 1 && !controls._performCursorZoom &&
      camera.isPerspectiveCamera && camera.parent === null && camera.matrixAutoUpdate && camera.matrixWorldAutoUpdate;
  }
  function shadowController() {
    const shadow = Object.create(Object.getPrototypeOf(controls));
    for (const [key, value] of Object.entries(controls)) {
      shadow[key] = value && typeof value.clone === "function" ? value.clone() : value;
    }
    shadow.object = camera.clone(); shadow._listeners = {};
    return shadow;
  }
  function update(...args) {
    if (disposed) throw new Error("The owned convergence helper was already disposed");
    if (!pending) return nativeUpdate.apply(controls, args);
    pending = false;
    if (!eligible()) { totals.structuralUnsupported++; return nativeUpdate.apply(controls, args); }
    const started = performance.now(), shadow = shadowController(), seen = new Set();
    let previous = signature(shadow, shadow.object), fixed = false, steps = 0, expected;
    seen.add(previous);
    for (; steps < maximumSteps;) {
      nativeUpdate.apply(shadow, args); shadow.object.updateMatrixWorld(); steps++; totals.shadowUpdates++;
      const next = signature(shadow, shadow.object);
      if (next === previous) { fixed = true; expected = next; break; }
      if (seen.has(next)) { totals.cycleUnsupported++; break; }
      seen.add(next); previous = next;
    }
    totals.preflightMs += performance.now() - started;
    if (!fixed) {
      if (steps === maximumSteps) totals.boundUnsupported++;
      return nativeUpdate.apply(controls, args);
    }
    const replayStart = performance.now();
    let changed = false;
    for (let i = 0; i < steps; ++i) {
      changed = nativeUpdate.apply(controls, args) || changed;
      camera.updateMatrixWorld(); totals.replayUpdates++;
    }
    totals.replayMs += performance.now() - replayStart;
    if (signature(controls, camera) !== expected) throw new Error("The exact original-controller replay differs from its unchanged preflight");
    totals.converged++;
    return changed;
  }
  return {
    markDiscrete(fraction = 0) {
      if (disposed) throw new Error("The owned convergence helper was already disposed");
      if (fraction === 0 && !state.playing && state.navigationMode === "orbit") { pending = true; totals.marked++; }
    },
    update,
    snapshot: () => ({ ...totals, pending, disposed }),
    dispose() {
      if (disposed) throw new Error("The owned convergence helper cannot be disposed twice");
      pending = false; disposed = true;
      return { ...totals, pending, disposed, nativeUpdateUnchanged: controls.update === nativeUpdate };
    },
  };
}
