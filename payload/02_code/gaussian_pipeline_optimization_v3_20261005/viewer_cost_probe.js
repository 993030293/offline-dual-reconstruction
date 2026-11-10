// Observe original asynchronous work without changing model, camera, or sorting arithmetic.
export function installViewerCostProbe({ state, scene, camera, renderer, spark, SplatMesh, preset, projection, setMode, syncFlightAngles, requestRender }) {
  const originalRender = renderer.render;
  const originalReadback = spark.readbackDepth;
  const originalDriveSort = spark.driveSort;
  const events = [], restorers = [];
  let active = 0, disposed = false, hook = false, attachedWorker = null;
  const totals = { renderCalls: 0, renderSubmitMs: 0, readbackCalls: 0, readbackWallMs: 0,
    observedWorkerCalls: 0, workerWallMs: 0, workerOrderingBytes: 0, workerReadbackBytes: 0,
    sortStarts: 0, sortCallsIncludingNoop: 0, sortPromiseWallMsIncludingNested: 0 };

  function attachWorker() {
    const worker = spark.sortWorker;
    if (!hook || !worker || attachedWorker === worker) return;
    if (typeof worker.call !== "function") throw new Error("The actual sort-worker API changed");
    const original = worker.call;
    const wrapped = async function (...args) {
      const started = performance.now(), name = args[0], input = args[1];
      const observed = name === "sortSplats32";
      if (observed) {
        totals.observedWorkerCalls++;
        totals.workerOrderingBytes += input.ordering.byteLength;
        totals.workerReadbackBytes += input.readback.byteLength;
      }
      active++;
      try { return await original.apply(this, args); }
      finally {
        const elapsed = performance.now() - started;
        if (observed) { totals.workerWallMs += elapsed; events.push({ kind: "workerSort32", ms: elapsed }); }
        active--;
      }
    };
    worker.call = wrapped;
    restorers.push(() => { if (worker.call !== wrapped) throw new Error("The owned sort-worker hook changed"); worker.call = original; });
    attachedWorker = worker;
  }

  function enable() {
    if (hook || disposed) throw new Error("The private cost probe cannot be enabled twice");
    hook = true;
    const render = function (...args) {
      const started = performance.now();
      try { return originalRender.apply(this, args); }
      finally { totals.renderCalls++; totals.renderSubmitMs += performance.now() - started; }
    };
    const readback = async function (...args) {
      const started = performance.now();
      totals.readbackCalls++; active++;
      try { return await originalReadback.apply(this, args); }
      finally { totals.readbackWallMs += performance.now() - started; active--; }
    };
    const sort = async function (...args) {
      attachWorker();
      const started = performance.now(), previous = this.lastSortTime;
      totals.sortCallsIncludingNoop++; active++;
      try { return await originalDriveSort.apply(this, args); }
      finally {
        if (this.lastSortTime !== previous) totals.sortStarts++;
        totals.sortPromiseWallMsIncludingNested += performance.now() - started;
        active--; attachWorker();
      }
    };
    renderer.render = render; spark.readbackDepth = readback; spark.driveSort = sort;
    restorers.push(() => {
      if (renderer.render !== render || spark.readbackDepth !== readback || spark.driveSort !== sort) throw new Error("An owned browser timing hook changed");
      renderer.render = originalRender; spark.readbackDepth = originalReadback; spark.driveSort = originalDriveSort;
    });
  }

  function pose() {
    return { position: camera.position.toArray(), direction: camera.getWorldDirection(camera.position.clone()).toArray(),
      projection: camera.projectionMatrix.toArray(), dimensions: [renderer.domElement.width, renderer.domElement.height],
      count: [state.fullGaussianCount, state.boxGaussianCount], mode: state.sceneMode,
      activeSplats: spark.activeSplats, currentSplats: spark.current?.numSplats,
      sorting: spark.sorting, sortDirty: spark.sortDirty, sortTimeout: spark.sortTimeoutId !== undefined,
      frame: state.frames, pendingRAF: state.isRenderPending() };
  }

  return {
    enable, attachWorker, preset,
    snapshot: () => ({ totals: { ...totals }, eventCount: events.length, active, disposed, hook, ...pose() }),
    forceFrame: requestRender,
    freeCamera(position, direction) {
      setMode("fly"); camera.position.fromArray(position);
      camera.lookAt(camera.position.clone().add({ x: direction[0], y: direction[1], z: direction[2] }));
      syncFlightAngles(); projection.free(); requestRender();
    },
    bounds() {
      let selected;
      scene.traverse(node => { if (node instanceof SplatMesh && node.visible) selected = node; });
      if (!selected) throw new Error("The actual visible native Gaussian model was not found");
      const box = selected.getBoundingBox(true), center = box.getCenter(box.min.clone());
      return { center: center.toArray(), radius: box.min.distanceTo(box.max) / 2 };
    },
    async capture() {
      const gl = renderer.getContext(), canvas = renderer.domElement;
      const bytes = new Uint8Array(canvas.width * canvas.height * 4), started = performance.now();
      gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
      const completeReadPixelsMs = performance.now() - started, error = gl.getError();
      const digest = await crypto.subtle.digest("SHA-256", bytes);
      let nonuniform = 0;
      for (let i = 4; i < bytes.length; i += 4) if (bytes[i] !== bytes[0] || bytes[i + 1] !== bytes[1] || bytes[i + 2] !== bytes[2]) nonuniform++;
      return { ...pose(), rgba_sha256: [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, "0")).join(""),
        nonuniformPixels: nonuniform, completeReadPixelsMs, glError: error, totals: { ...totals } };
    },
    dispose() {
      if (active) throw new Error("The timing hooks cannot restore while native asynchronous calls are active");
      for (const restore of restorers.reverse()) restore();
      disposed = true;
      return { active, disposed, originalRender: renderer.render === originalRender, originalReadback: spark.readbackDepth === originalReadback,
        originalDriveSort: spark.driveSort === originalDriveSort, observedWorker: Boolean(attachedWorker) };
    },
  };
}
