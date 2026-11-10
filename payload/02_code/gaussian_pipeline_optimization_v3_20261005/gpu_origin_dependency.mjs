// Bind GPU residency to the frozen private origin dependency implementation.
import { createOriginDependency as createFrozenOriginDependency } from "/r118/frozen_origin_dependency.mjs";

export function createOriginDependency(options) {
  let source = createFrozenOriginDependency.toString();
  const changes = [];
  function replace(before, after) {
    if (source.split(before).length !== 2) throw new Error("Frozen R115 factory anchor changed");
    source = source.replace(before, after);
    changes.push({ before, after });
  }
  replace("const owners = new WeakSet();", "const owners = new WeakSet();\n  const gpuStates = new Map();");
  replace("const trace = { admitted: 0, original: 0, suppressed: 0, completeInitialWords: 0 };",
    "const trace = { admitted: 0, original: 0, suppressed: 0, completeInitialWords: 0, gpuNativeRebuilds: 0, contextLosses: 0, contextRestores: 0 };");
  replace("  function arrays(packed) {", `  function liveAccumulator(owner, accumulator) {
    const renderer = owner.renderer, gl = renderer.getContext();
    const target = accumulator?.target;
    if (!target || !renderer.properties.has(target)) return false;
    const framebuffer = renderer.properties.get(target).__webglFramebuffer;
    const frames = [framebuffer].flat().filter(Boolean);
    if (!frames.length || !frames.every(handle => gl.isFramebuffer(handle))) return false;
    return target.textures?.length > 0 && target.textures.every(texture => {
      if (!renderer.properties.has(texture)) return false;
      const handle = renderer.properties.get(texture).__webglTexture;
      return Boolean(handle && gl.isTexture(handle));
    });
  }

  function gpuResident(owner) {
    const state = gpuStates.get(owner);
    return Boolean(state && !state.invalidated &&
      state.properties === owner.renderer.properties && !owner.renderer.getContext().isContextLost() &&
      liveAccumulator(owner, owner.current) && liveAccumulator(owner, owner.display));
  }

  function rememberGpu(owner) {
    const state = gpuStates.get(owner);
    if (state && !owner.renderer.getContext().isContextLost() &&
        liveAccumulator(owner, owner.current) && liveAccumulator(owner, owner.display)) {
      state.properties = owner.renderer.properties;
      state.invalidated = false;
    }
  }

  function arrays(packed) {`);
  replace("      owners.add(this);", `      owners.add(this);
      const canvas = this.renderer.domElement;
      const state = { canvas, invalidated: true, properties: null };
      state.lost = () => { state.invalidated = true; trace.contextLosses++; };
      state.restored = () => { state.invalidated = true; trace.contextRestores++; };
      canvas.addEventListener("webglcontextlost", state.lost);
      canvas.addEventListener("webglcontextrestored", state.restored);
      gpuStates.set(this, state);`);
  replace("      safe &&= eligible.size > 0;", `      safe &&= eligible.size > 0;
      const rebuildGpu = safe && !this.renderer.getContext().isContextLost() && !gpuResident(this);
      safe &&= gpuResident(this);`);
  replace("          pending = originalUpdate.call(this, args);", `          // Invalid native render targets require the original full generate/sort path before reuse.
          if (rebuildGpu) trace.gpuNativeRebuilds++;
          pending = originalUpdate.call(this, rebuildGpu ? { ...args, autoUpdate: false } : args);`);
  replace("        return await pending;\n      }", `        const result = await pending;
        for (const mesh of eligible) recordNativeState(mesh, records.get(mesh));
        rememberGpu(this);
        return result;
      }`);
  replace("    bindingCount: () => records.size, compatible, originalUpdateSource: source, trace,",
    "    bindingCount: () => records.size + gpuStates.size, resourceBindingCount: () => gpuStates.size, compatible, originalUpdateSource: source, trace,");
  replace("    restoreAll: () => { for (const mesh of [...records.keys()]) restore(mesh); }", `    restoreAll: () => {
      try { for (const mesh of [...records.keys()]) restore(mesh); }
      finally {
        for (const state of gpuStates.values()) {
          state.canvas.removeEventListener("webglcontextlost", state.lost);
          state.canvas.removeEventListener("webglcontextrestored", state.restored);
        }
        gpuStates.clear();
      }
    }`);
  const factory = new Function("return (" + source + ");")();
  const result = factory(options);
  result.sourceChanges = changes;
  return result;
}
