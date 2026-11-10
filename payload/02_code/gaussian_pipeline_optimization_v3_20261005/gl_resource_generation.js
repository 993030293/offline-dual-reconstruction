// Track owned WebGL handle generations and optionally omit cleanup of handles invalidated by loss.
export function installGlResourceGeneration({ renderer, enabled = false }) {
  const gl = renderer.getContext(), births = new WeakMap(), originals = [], events = [];
  let generation = 0, nextId = 0, disposed = false;
  const totals = { creations: 0, deleteCalls: 0, knownStaleDeleteCalls: 0, skippedKnownStaleDeletes: 0,
    unknownDeleteCalls: 0, actualContextLossEvents: 0, actualContextRestoreEvents: 0 };
  const kinds = ["Buffer", "Framebuffer", "Renderbuffer", "Texture", "Program", "Shader", "VertexArray", "Sampler", "TransformFeedback", "Query"];
  const loss = () => { generation++; totals.actualContextLossEvents++; };
  const restoration = () => { totals.actualContextRestoreEvents++; };
  renderer.domElement.addEventListener("webglcontextlost", loss);
  renderer.domElement.addEventListener("webglcontextrestored", restoration);
  for (const kind of [...kinds, "Sync"]) {
    const createName = kind === "Sync" ? "fenceSync" : "create" + kind, deleteName = "delete" + kind;
    if (typeof gl[createName] !== "function" || typeof gl[deleteName] !== "function") throw new Error("The original WebGL2 resource API changed: " + kind);
    const createOriginal = gl[createName], deleteOriginal = gl[deleteName];
    const create = function (...args) {
      if (disposed) throw new Error("The owned GL generation tracker was already disposed");
      const handle = createOriginal.apply(this, args);
      if (handle !== null) { births.set(handle, { kind, generation, id: ++nextId }); totals.creations++; }
      return handle;
    };
    const remove = function (...args) {
      if (disposed) throw new Error("The owned GL generation tracker was already disposed");
      totals.deleteCalls++;
      const handle = args[0], birth = handle && births.get(handle), stale = birth && birth.kind === kind && birth.generation < generation;
      if (handle && !birth) totals.unknownDeleteCalls++;
      if (stale) {
        if (events.length >= 4096) throw new Error("The registered GL generation event bound was exceeded");
        totals.knownStaleDeleteCalls++;
        events.push({ method: deleteName, birthGeneration: birth.generation, currentGeneration: generation, id: birth.id, skipped: enabled });
        if (enabled) { totals.skippedKnownStaleDeletes++; return undefined; }
      }
      return deleteOriginal.apply(this, args);
    };
    gl[createName] = create; gl[deleteName] = remove;
    originals.push(() => { if (gl[createName] !== create || gl[deleteName] !== remove) throw new Error("An owned GL resource method changed"); gl[createName] = createOriginal; gl[deleteName] = deleteOriginal; });
  }
  return {
    snapshot: () => ({ generation, totals: { ...totals }, events: events.map(row => ({ ...row })), enabled, disposed }),
    dispose() {
      if (disposed) throw new Error("The GL generation tracker cannot be disposed twice");
      for (const restore of originals.reverse()) restore();
      renderer.domElement.removeEventListener("webglcontextlost", loss);
      renderer.domElement.removeEventListener("webglcontextrestored", restoration);
      disposed = true;
      return { generation, totals: { ...totals }, events: events.map(row => ({ ...row })), enabled, disposed, originalAliasesRestored: true };
    },
  };
}
