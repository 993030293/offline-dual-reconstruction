// Observe owned native GPU state without changing the frozen R115 algorithm.
import { SparkRenderer as NativeSparkRenderer } from "@sparkjsdev/spark";
import { attachOriginHarness } from "/r118/origin_dependency_browser_005.mjs";

export function attachGpuDiagnostic(options) {
  const { state, renderer, spark, scene, camera } = options;
  const gl = renderer.getContext(), canvas = renderer.domElement;
  const originalGetError = gl.getError;
  const errorDescriptor = Object.getOwnPropertyDescriptor(gl, "getError");
  const objects = new WeakMap(), generations = new WeakMap(), observed = new Map();
  let nextObject = 0, nextProperties = 0, scope = "initial_attach";
  const trace = { events: [], snapshots: [], numericErrors: [], generates: [], nativeInterventions: 0 };
  const id = object => {
    if (!object) return null;
    if (!objects.has(object)) objects.set(object, ++nextObject);
    return objects.get(object);
  };
  const propertiesId = () => {
    if (!generations.has(renderer.properties)) generations.set(renderer.properties, ++nextProperties);
    return generations.get(renderer.properties);
  };
  function textureState(texture) {
    if (!texture) return null;
    const propertiesPresent = renderer.properties.has(texture);
    const properties = propertiesPresent ? renderer.properties.get(texture) : null;
    const handle = properties?.__webglTexture;
    return { object: id(texture), version: texture.version, propertiesPresent,
      handle: id(handle), liveNativeTexture: Boolean(handle && gl.isTexture(handle)),
      propertiesVersion: properties?.__version ?? null, renderTargetTexture: texture.isRenderTargetTexture ?? false };
  }
  function accumulatorState(accumulator) {
    if (!accumulator) return null;
    const target = accumulator.target;
    const present = Boolean(target && renderer.properties.has(target));
    const properties = present ? renderer.properties.get(target) : null;
    const frames = [properties?.__webglFramebuffer].flat().filter(Boolean);
    return { object: id(accumulator), version: accumulator.version, mappingVersion: accumulator.mappingVersion,
      numSplats: accumulator.numSplats, maxSplats: accumulator.maxSplats, target: id(target),
      targetPropertiesPresent: present, framebuffers: frames.map(handle => ({ object: id(handle), live: gl.isFramebuffer(handle) })),
      textures: target?.textures?.map(textureState) ?? [] };
  }
  function snapshot(reason) {
    const row = { reason, scope, timeMs: performance.now(), propertiesGeneration: propertiesId(),
      contextLost: gl.isContextLost(), current: accumulatorState(spark.current), display: accumulatorState(spark.display),
      pool: (spark.accumulators ?? []).map(accumulatorState), orderingTexture: textureState(spark.orderingTexture),
      sorting: spark.sorting, sortDirty: spark.sortDirty,
      sortTimeoutPending: spark.sortTimeoutId !== undefined, updateTimeoutPending: spark.updateTimeoutId !== undefined,
      sortedCenter: spark.sortedCenter?.toArray(), sortedDir: spark.sortedDir?.toArray(),
      activeSplats: spark.activeSplats, dependency: options.dependency ? { ...options.dependency.trace } : null };
    trace.snapshots.push(row);
    return row;
  }
  const boundGetError = function (...args) {
    const code = originalGetError.apply(this, args);
    if (code !== gl.NO_ERROR) trace.numericErrors.push({ code, hex: "0x" + code.toString(16), scope,
      timeMs: performance.now(), stack: new Error("Original numeric GL error observation").stack,
      nativeState: snapshot("original_getError_nonzero") });
    return code;
  };
  gl.getError = boundGetError;
  function observeAccumulators() {
    for (const accumulator of new Set([spark.current, spark.display, ...(spark.accumulators ?? [])])) {
      if (!accumulator || observed.has(accumulator)) continue;
      const descriptor = Object.getOwnPropertyDescriptor(accumulator, "generate"), original = accumulator.generate;
      const bound = function (...args) {
        const row = { object: id(this), scope, before: snapshot("before_original_generate"), completed: false };
        trace.generates.push(row);
        try { const result = original.apply(this, args); row.completed = true; return result; }
        finally { row.after = snapshot("after_original_generate"); }
      };
      accumulator.generate = bound;
      observed.set(accumulator, { descriptor, original, bound });
    }
  }
  const lostListener = () => { trace.events.push({ event: "webglcontextlost", timeMs: performance.now() }); snapshot("actual_context_lost_event"); };
  const restoredListener = () => { trace.events.push({ event: "webglcontextrestored", timeMs: performance.now() }); snapshot("actual_context_restored_event"); };
  canvas.addEventListener("webglcontextlost", lostListener);
  canvas.addEventListener("webglcontextrestored", restoredListener);
  observeAccumulators();
  snapshot("initial_attach");
  attachOriginHarness(options);
  const harness = state.originHarness;
  const action = harness.action, capture = harness.capture, restore = harness.restore;
  harness.action = async (...args) => {
    scope = "action:" + args[0]; observeAccumulators(); snapshot("before_action");
    try { return await action(...args); }
    finally { snapshot("after_action"); }
  };
  harness.capture = async (...args) => {
    scope = "capture:" + args[0]; observeAccumulators(); snapshot("before_capture");
    try { return await capture(...args); }
    finally { snapshot("after_capture"); }
  };
  harness.restore = async () => {
    scope = "owned_cleanup";
    let result;
    try { result = await restore(); }
    finally {
      for (const [accumulator, row] of observed) {
        if (accumulator.generate !== row.bound) throw new Error("Owned native generate observer changed externally");
        if (row.descriptor) Object.defineProperty(accumulator, "generate", row.descriptor);
        else delete accumulator.generate;
      }
      if (gl.getError !== boundGetError) throw new Error("Owned numeric GL observer changed externally");
      if (errorDescriptor) Object.defineProperty(gl, "getError", errorDescriptor);
      else delete gl.getError;
      canvas.removeEventListener("webglcontextlost", lostListener);
      canvas.removeEventListener("webglcontextrestored", restoredListener);
    }
    return { ...result, numericGLObserverRestored: gl.getError === originalGetError,
      generateObserversRestored: [...observed].every(([accumulator, row]) => accumulator.generate === row.original),
      diagnosticContextListenersRemoved: true };
  };
  state.gpuDiagnostic = {
    snapshot, trace,
    export: () => ({ ...trace, latest: snapshot("explicit_export") }),
    nativeIntervention: async () => {
      if (trace.nativeInterventions !== 0) throw new Error("Only one preregistered original-native diagnostic intervention is allowed");
      trace.nativeInterventions++;
      scope = "diagnostic_original_native_intervention";
      snapshot("before_one_native_full_update");
      // This diagnostic probe cannot promote or rescue the frozen rejected R115 candidate.
      await NativeSparkRenderer.prototype.updateInternal.call(spark, { scene, camera, autoUpdate: false });
      snapshot("after_one_native_full_update");
      return await harness.settle();
    }
  };
}
