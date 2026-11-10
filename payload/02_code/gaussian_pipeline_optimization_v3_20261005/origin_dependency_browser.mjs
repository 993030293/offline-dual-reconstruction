// Observe private original renderers without changing vendor code or sort results.
import * as THREE from "three";
import * as Spark from "@sparkjsdev/spark";

const nextFrame = () => new Promise(resolve => requestAnimationFrame(resolve));
const identity = array => Array.from(array);

export function attachOriginHarness({ state, renderer, scene, camera, spark, controls,
  modelRegistry, requestRender, dependency }) {
  let worker = spark.sortWorker;
  let ordering = null;
  const workers = new Map();
  const initialQuaternion = camera.quaternion.clone();
  let rotationBase = null;
  let baseTransforms = null;
  let extraGenerator = null;
  let edit = null;
  const publishedSources = new Set();
  const trace = { sorts: 0, publishedBytes: 0, captures: 0, GPUfinishes: 0 };
  function observeWorker(value) {
    if (!value || workers.has(value)) return;
    const original = value.call;
    const bound = async function (name, args, ...rest) {
      const result = await original.call(this, name, args, ...rest);
      if (name === "sortSplats32") {
        ordering = result.ordering.slice();
        trace.sorts++;
      }
      return result;
    };
    value.call = bound;
    workers.set(value, { original, bound });
  }
  Object.defineProperty(spark, "sortWorker", { configurable: true,
    get: () => worker, set(value) { worker = value; observeWorker(value); } });
  observeWorker(worker);
  function mesh() {
    return scene.children.find(node => node instanceof Spark.SplatMesh && node.visible);
  }
  async function settle() {
    const started = performance.now();
    let closed = 0;
    while (performance.now() - started < 15000) {
      await nextFrame();
      if (state.errors.length) throw new Error("Original viewer error: " + state.errors.join("\n"));
      const idle = state.ready && !state.modelLoading && !state.playing &&
        !spark.sorting && !spark.sortDirty && spark.sortTimeoutId === undefined &&
        spark.updateTimeoutId === undefined && !state.isRenderPending() &&
        spark.display.mappingVersion === spark.current.mappingVersion;
      closed = idle ? closed + 1 : 0;
      if (closed === 2) {
        const gl = renderer.getContext();
        gl.finish();
        trace.GPUfinishes++;
        if (gl.getError() !== gl.NO_ERROR) throw new Error("Actual GPU completion reported an error");
        return { elapsedMs: performance.now() - started, sourceSortingClosed: true };
      }
    }
    throw new Error("Original source sorting and demand rendering did not close in 15 seconds");
  }
  async function publish(label, part, bytes) {
    const response = await fetch("/r115/publish/" + label + "/" + part, { method: "POST", body: bytes });
    if (!response.ok) throw new Error("Owned binary publication failed: " + response.status);
    trace.publishedBytes += bytes.byteLength;
  }
  function readOrderingGPU() {
    const texture = spark.orderingTexture;
    if (!texture || !ordering) throw new Error("The real sort result is unavailable");
    const gl = renderer.getContext();
    const native = renderer.properties.get(texture).__webglTexture;
    if (!native) throw new Error("Original ordering texture has not reached the GPU");
    const beforeRead = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING);
    const beforeDraw = gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING);
    const buffer = new Uint32Array(texture.image.width * texture.image.height * 4);
    const framebuffer = gl.createFramebuffer();
    try {
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, native, 0);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      if (gl.checkFramebufferStatus(gl.READ_FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error("Original integer ordering texture cannot be read completely");
      }
      gl.readPixels(0, 0, texture.image.width, texture.image.height, gl.RGBA_INTEGER,
        gl.UNSIGNED_INT, buffer);
      if (gl.getError() !== gl.NO_ERROR) throw new Error("Actual ordering GPU read failed");
    } finally {
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, beforeRead);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, beforeDraw);
      gl.deleteFramebuffer(framebuffer);
    }
    if (buffer.length !== ordering.length) throw new Error("Full CPU/GPU ordering capacity differs");
    for (let i = 0; i < buffer.length; i++) {
      if (buffer[i] !== ordering[i]) throw new Error("Actual GPU ordering differs from the full worker result at " + i);
    }
    return buffer;
  }
  async function readPackedGPU() {
    const current = spark.current, target = current.target;
    if (!target || current.extSplats) throw new Error("An original native packed accumulator is required");
    const count = Math.ceil(current.numSplats / 2048) * 2048;
    const output = new Uint32Array(count * 4);
    const saved = spark.saveRenderState(renderer);
    const width = target.width, height = target.height;
    try {
      for (let base = 0; base < count; base += width * height) {
        const layer = Math.floor(base / (width * height));
        const rows = Math.min(height, Math.ceil((count - base) / width));
        const part = output.subarray(base * 4, (base + width * rows) * 4);
        renderer.setRenderTarget(target, layer);
        await renderer.readRenderTargetPixelsAsync(target, 0, 0, width, rows, part, undefined, 0);
      }
    } finally {
      spark.resetRenderState(renderer, saved);
    }
    return output;
  }
  async function capture(label) {
    const closure = await settle();
    const gl = renderer.getContext();
    const pixels = new Uint8Array(gl.drawingBufferWidth * gl.drawingBufferHeight * 4);
    gl.readPixels(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    if (gl.getError() !== gl.NO_ERROR) throw new Error("Original complete framebuffer read failed");
    const gpuOrdering = readOrderingGPU();
    const gpuPacked = await readPackedGPU();
    const cameraBytes = new Float64Array([...camera.position.toArray(), ...camera.quaternion.toArray(),
      ...camera.matrixWorld.elements, ...camera.projectionMatrix.elements]);
    for (const [name, bytes] of [["pixels", pixels], ["ordering_gpu", gpuOrdering],
      ["ordering_worker", ordering], ["packed_gpu", gpuPacked], ["camera", cameraBytes]]) {
      await publish(label, name + ".bin", new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
    }
    const active = mesh();
    if (!publishedSources.has(state.sceneMode)) {
      for (const [name, array] of [["source_packed", active.splats.packedArray], ["source_sh1", active.splats.extra.sh1],
        ["source_sh2", active.splats.extra.sh2], ["source_sh3", active.splats.extra.sh3]]) {
        await publish(label, name + ".bin", new Uint8Array(array.buffer, array.byteOffset, array.byteLength));
      }
      publishedSources.add(state.sceneMode);
    }
    const result = { label, width: gl.drawingBufferWidth, height: gl.drawingBufferHeight,
      mode: state.sceneMode, activeSplats: spark.activeSplats, pointCount: spark.current.numSplats,
      maxSh: active.maxSh, numSh: active.splats.getNumSh(),
      meshCount: active.numSplats, projection: state.projectionMode,
      originalInstance: active instanceof Spark.SplatMesh && Object.getPrototypeOf(active) === Spark.SplatMesh.prototype,
      sourceSortingClosed: closure.sourceSortingClosed, errors: [...state.errors],
      nativeIntegrityChecks: (globalThis.assetIntegrityTrace ?? []).every(row => row.all_native_checks_retained),
      trace: { ...trace }, dependency: dependency ? { ...dependency.trace } : null };
    trace.captures++;
    return result;
  }
  async function action(name, index = 0) {
    await settle();
    if (name === "capture-view") state.setView(index);
    else if (name === "select") await modelRegistry.select(index);
    else if (name === "rotation-base") {
      document.getElementById("navigation-mode").value = "fly";
      document.getElementById("navigation-mode").dispatchEvent(new Event("change", { bubbles: true }));
      rotationBase = { position: camera.position.clone(), quaternion: camera.quaternion.clone() };
    } else if (name === "rotate") {
      if (!rotationBase) throw new Error("The fixed origin rotation base is missing");
      camera.position.copy(rotationBase.position);
      camera.quaternion.copy(rotationBase.quaternion).multiply(new THREE.Quaternion()
        .setFromAxisAngle(new THREE.Vector3(0, 1, 0), index * (2 * Math.PI / 24)));
    } else if (name === "translate") camera.position.x += .07;
    else if (name === "model-transform") {
      const m = mesh();
      baseTransforms ??= { position: m.position.clone(), quaternion: m.quaternion.clone(), scale: m.scale.clone() };
      m.position.x += .03;
      m.rotation.z += .17;
      m.scale.multiplyScalar(1.01);
    } else if (name === "world-transform") {
      scene.position.y += .05;
      scene.rotation.z += .11;
    } else if (name === "recolor") { mesh().recolor.setRGB(.87, .96, .94); mesh().opacity = .91; }
    else if (name === "force") await spark.update({ scene, camera });
    else if (name === "radial-off") spark.sortRadial = false;
    else if (name === "radial-on") spark.sortRadial = true;
    else if (name === "onFrame") mesh().onFrame = () => {};
    else if (name === "onFrame-clear") mesh().onFrame = undefined;
    else if (name === "view-uniform") mesh().enableViewToObject = true;
    else if (name === "view-uniform-clear") mesh().enableViewToObject = false;
    else if (name === "global-edit") { edit = new Spark.SplatEdit({ opacity: .97 }); scene.add(edit); }
    else if (name === "global-edit-clear") { scene.remove(edit); edit = null; }
    else if (name === "other-generator") {
      extraGenerator = new Spark.SplatGenerator({ numSplats: 0, update: () => {} }); scene.add(extraGenerator);
    } else if (name === "other-generator-clear") { scene.remove(extraGenerator); extraGenerator = null; }
    else if (name === "data-mutation") { mesh().splats.packedArray[7] ^= 1; mesh().needsUpdate = true; mesh().splats.needsUpdate = true; }
    else if (name === "restore-bindings") {
      if (dependency) dependency.restoreAll();
    } else if (name !== "none") throw new Error("Unregistered actual browser action " + name);
    requestRender();
    return await settle();
  }
  async function restore() {
    await settle();
    dependency?.restoreAll();
    for (const [observed, row] of workers) {
      if (observed.call !== row.bound) throw new Error("Another writer changed the private worker observer");
      observed.call = row.original;
    }
    delete spark.sortWorker;
    spark.sortWorker = worker;
    return { contextBindings: dependency?.bindingCount() ?? 0, workerObserversRestored: true,
      nativeMeshPrototypePreserved: scene.children.filter(n => n instanceof Spark.SplatMesh)
        .every(n => Object.getPrototypeOf(n) === Spark.SplatMesh.prototype), trace };
  }
  state.originHarness = { capture, action, settle, restore, camera, spark, scene, controls,
    initialQuaternion: identity(initialQuaternion.toArray()), dependency };
}
