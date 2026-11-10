import { installGlResourceGenerationV2 } from "./gl_resource_generation_v2.js";

// Cross a browser frame after actual loss before requesting restoration, with bounded event waits.
export async function runGlResourceGenerationGenericV3() {
  const canvas = document.createElement("canvas"), foreignCanvas = document.createElement("canvas");
  document.body.append(canvas, foreignCanvas);
  const gl = canvas.getContext("webgl2"), other = foreignCanvas.getContext("webgl2");
  if (!gl || !other) throw new Error("Two real WebGL2 contexts are required");
  const rows = [], aliases = [], forwarded = {}, checks = {};
  const kinds = ["Buffer", "Framebuffer", "Renderbuffer", "Texture", "Program", "Shader", "VertexArray", "Sampler", "TransformFeedback", "Query", "Sync"];
  const assert = (condition, message) => { if (!condition) throw new Error(message); };
  const create = (context, kind) => kind === "Shader" ? context.createShader(context.VERTEX_SHADER) : kind === "Sync" ? context.fenceSync(context.SYNC_GPU_COMMANDS_COMPLETE, 0) : context["create" + kind]();
  const wait = name => new Promise((resolve, reject) => {
    const listener = event => { clearTimeout(timer); resolve(event); };
    const timer = setTimeout(() => { canvas.removeEventListener(name, listener); reject(new Error("The actual bounded context event did not arrive: " + name)); }, 10000);
    canvas.addEventListener(name, listener, { once: true });
  });
  const unknown = gl.createTexture(), foreign = other.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, unknown); other.bindTexture(other.TEXTURE_2D, foreign);
  canvas.addEventListener("webglcontextlost", event => event.preventDefault());
  for (const kind of kinds) {
    const name = "delete" + kind, original = gl[name];
    const wrapper = function (...args) { forwarded[name] = (forwarded[name] ?? 0) + 1; return original.apply(this, args); };
    gl[name] = wrapper; aliases.push({ name, original, wrapper });
  }
  const registry = installGlResourceGenerationV2({ renderer: { getContext: () => gl, domElement: canvas } });
  try {
    gl.deleteTexture(unknown);
    assert(forwarded.deleteTexture === 1 && !gl.isTexture(unknown) && gl.getError() === gl.NO_ERROR, "Unknown current-context cleanup was not forwarded");
    gl.deleteTexture(foreign);
    const foreignError = gl.getError();
    assert(forwarded.deleteTexture === 2 && foreignError === gl.INVALID_OPERATION && other.isTexture(foreign), "The unknown foreign-resource error was hidden");
    rows.push({ name: "unknown_and_foreign_handle_forwarding", foreignError });
    for (let cycle = 0; cycle < 2; ++cycle) {
      const old = kinds.map(kind => ({ kind, handle: create(gl, kind) }));
      assert(old.every(row => row.handle !== null) && gl.getError() === gl.NO_ERROR, "A current resource could not be created");
      const extension = gl.getExtension("WEBGL_lose_context");
      assert(extension, "Actual context loss is unavailable");
      const lost = wait("webglcontextlost"); extension.loseContext(); await lost;
      assert(gl.isContextLost(), "The original context did not actually become lost");
      const expectedLossError = gl.getError();
      assert(expectedLossError === gl.CONTEXT_LOST_WEBGL, "The expected actual loss status was not recorded");
      await new Promise(requestAnimationFrame);
      const restored = wait("webglcontextrestored"); extension.restoreContext(); await restored;
      assert(!gl.isContextLost(), "The original context did not actually restore");
      const before = { ...forwarded };
      for (const row of old) gl["delete" + row.kind](row.handle);
      assert(JSON.stringify(before) === JSON.stringify(forwarded) && gl.getError() === gl.NO_ERROR, "Recorded stale cleanup reached the restored context or generated an error");
      const staleTexture = old.find(row => row.kind === "Texture").handle;
      gl.deleteTexture.call(other, staleTexture);
      const receiverError = other.getError();
      assert(forwarded.deleteTexture === (before.deleteTexture ?? 0) + 1 && receiverError === other.INVALID_OPERATION, "A foreign receiver error was hidden by stale classification");
      const current = kinds.map(kind => ({ kind, handle: create(gl, kind) }));
      assert(current.every(row => row.handle !== null) && gl.getError() === gl.NO_ERROR, "A restored current-generation allocation failed");
      const nativeBefore = { ...forwarded };
      for (const row of current) gl["delete" + row.kind](row.handle);
      assert(kinds.every(kind => forwarded["delete" + kind] === (nativeBefore["delete" + kind] ?? 0) + 1) && gl.getError() === gl.NO_ERROR, "Current-generation deletion was not forwarded");
      const buffer = gl.createBuffer(); let wrongType = null;
      try { gl.deleteTexture(buffer); } catch (error) { wrongType = String(error); }
      assert(wrongType?.includes("TypeError") && gl.getError() === gl.NO_ERROR, "The original invalid-argument exception did not propagate");
      gl.deleteBuffer(buffer);
      rows.push({ name: "actual_loss_restore_generation_" + (cycle + 1), staleResources: old.length, expectedLossError, receiverError, wrongType });
    }
    const snapshot = registry.snapshot(), disposed = registry.dispose();
    const unchanged = aliases.every(row => gl[row.name] === row.wrapper);
    for (const row of aliases) { assert(gl[row.name] === row.wrapper, "An owned native-forwarding alias changed"); gl[row.name] = row.original; }
    checks.unknown_current_delete_and_foreign_error_forwarded = rows[0].foreignError === gl.INVALID_OPERATION;
    checks.two_actual_context_loss_and_restore_events = snapshot.generation === 2 && snapshot.totals.actualContextLossEvents === 2 && snapshot.totals.actualContextRestoreEvents === 2;
    checks.all_22_known_old_generation_resources_only_skipped = snapshot.totals.skippedKnownStaleDeletes === 22 && snapshot.events.length === 22;
    checks.current_generation_and_wrong_receiver_errors_forwarded = rows.slice(1).every(row => row.receiverError === other.INVALID_OPERATION && row.wrongType.includes("TypeError"));
    checks.unknown_handles_not_registered_as_stale = snapshot.totals.unknownDeleteCalls === 2;
    checks.all_owned_GL_aliases_restored = unchanged && disposed.originalAliasesRestored && aliases.every(row => gl[row.name] === row.original);
    checks.normal_GL_error_state_empty = gl.getError() === gl.NO_ERROR && other.getError() === other.NO_ERROR;
    return { checks, rows, snapshot, disposed, forwarded, extraGPUWriteBytes: 0, extraGPUReadBytes: 0,
      allocation_only_not_pixel_payload_work_zero_declared_no_performance_claim: true, shared_prerequisite_count: 0, new_optimization_count: 0 };
  } finally {
    other.deleteTexture(foreign);
    gl.getExtension("WEBGL_lose_context")?.loseContext(); other.getExtension("WEBGL_lose_context")?.loseContext();
    canvas.remove(); foreignCanvas.remove();
  }
}
