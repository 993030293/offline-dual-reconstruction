// Observe native dependency transitions without changing native call arguments or results.
export function installRadialDependencyObserver012({ state, scene, camera, spark, SplatMesh }) {
  const events = [], restorers = [], wrappedAccumulators = new WeakSet();
  let active = 0, disposed = false, label = "initial", nextUpdate = 0, synchronousUpdate = null;
  const totals = { updates: 0, automaticUpdates: 0, forcedUpdates: 0, prepares: 0, generates: 0 };
  function record(event) {
    if (events.length >= 20000) throw new Error("The registered radial dependency event limit was exceeded");
    events.push(event);
  }
  function modelState() {
    const models = [];
    scene.traverse(node => {
      if (!(node instanceof SplatMesh) || !node.visible) return;
      const context = node.context, transform = context?.viewToObject;
      models.push({ version: node.version, mappingVersion: node.mappingVersion, numSplats: node.numSplats,
        initialized: node.isInitialized, hasRgbDir: context?.splats?.hasRgbDir?.(), covSplats: node.covSplats,
        enableViewToObject: node.enableViewToObject, enableViewToWorld: node.enableViewToWorld,
        enableWorldToView: node.enableWorldToView, lodEnabled: context?.enableLod?.value,
        viewTranslate: transform?.translate?.value?.toArray(), viewRotate: transform?.rotate?.value?.toArray(),
        viewScale: transform?.scale?.value, generatorDirty: node.generatorDirty,
        objectModifiers: node.objectModifiers?.length ?? 0, worldModifiers: node.worldModifiers?.length ?? 0,
        hasOnFrame: Boolean(node.onFrame), hasSkinning: Boolean(node.skinning) });
    });
    return models;
  }
  function snapshot() {
    return { frame: state.frames, position: camera.getWorldPosition(camera.position.clone()).toArray(),
      direction: camera.getWorldDirection(camera.position.clone()).toArray(), sortRadial: spark.sortRadial,
      sortedCenter: spark.sortedCenter.toArray(), sortedDirection: spark.sortedDir.toArray(),
      currentVersion: spark.current?.version, mappingVersion: spark.current?.mappingVersion,
      currentOrigin: spark.current?.viewOrigin.toArray(), currentDirection: spark.current?.viewDirection.toArray(),
      displayVersion: spark.display?.version, displayMappingVersion: spark.display?.mappingVersion,
      numSplats: spark.current?.numSplats, activeSplats: spark.activeSplats,
      sorting: spark.sorting, sortDirty: spark.sortDirty, models: modelState() };
  }
  function wrapAccumulator(accumulator) {
    if (!accumulator || wrappedAccumulators.has(accumulator)) return;
    const originalPrepare = accumulator.prepareGenerate, originalGenerate = accumulator.generate;
    if (typeof originalPrepare !== "function" || typeof originalGenerate !== "function") throw new Error("The native accumulator API changed");
    const prepare = function (...args) {
      const before = snapshot(), started = performance.now(), result = originalPrepare.apply(this, args);
      totals.prepares++;
      record({ kind: "prepare", label, updateId: synchronousUpdate, started, submitMs: performance.now() - started,
        before, after: snapshot(), nextVersion: result.version, nextMappingVersion: result.mappingVersion,
        nextOrigin: this.viewOrigin.toArray(), nextDirection: this.viewDirection.toArray(),
        sourceMapping: this.mapping.map(row => ({ version: row.version, mappingVersion: row.mappingVersion,
          base: row.base, count: row.count, nodeVersion: row.node.version })) });
      return result;
    };
    const generate = function (...args) {
      const started = performance.now();
      totals.generates++;
      try { return originalGenerate.apply(this, args); }
      finally { record({ kind: "generate", label, updateId: synchronousUpdate, started,
        submitMs: performance.now() - started, count: args[0]?.count, version: this.version,
        mappingVersion: this.mappingVersion, origin: this.viewOrigin.toArray(), direction: this.viewDirection.toArray() }); }
    };
    accumulator.prepareGenerate = prepare; accumulator.generate = generate;
    wrappedAccumulators.add(accumulator);
    restorers.push(() => {
      if (accumulator.prepareGenerate !== prepare || accumulator.generate !== generate) throw new Error("An owned accumulator observer changed");
      accumulator.prepareGenerate = originalPrepare; accumulator.generate = originalGenerate;
    });
  }
  const originalUpdate = spark.updateInternal;
  if (typeof originalUpdate !== "function") throw new Error("The native renderer update API changed");
  const update = async function (...args) {
    if (disposed) throw new Error("The owned radial observer was already disposed");
    for (const accumulator of [this.current, this.display, ...this.accumulators]) wrapAccumulator(accumulator);
    const id = ++nextUpdate, before = snapshot(), started = performance.now();
    totals.updates++;
    if (args[0]?.autoUpdate) totals.automaticUpdates++; else totals.forcedUpdates++;
    active++; synchronousUpdate = id;
    let promise;
    try { promise = originalUpdate.apply(this, args); }
    finally { synchronousUpdate = null; }
    try { return await promise; }
    finally {
      record({ kind: "update", label, updateId: id, started, wallMsIncludingNested: performance.now() - started,
        autoUpdate: args[0]?.autoUpdate, before, after: snapshot() });
      active--;
    }
  };
  spark.updateInternal = update;
  restorers.push(() => { if (spark.updateInternal !== update) throw new Error("The owned radial update observer changed"); spark.updateInternal = originalUpdate; });
  return {
    setLabel(value) { label = value; },
    snapshot: () => ({ active, disposed, eventCount: events.length, totals: { ...totals }, native: snapshot() }),
    dispose() {
      if (active || disposed) throw new Error("The native radial observer must drain before restoration");
      for (const restore of restorers.reverse()) restore();
      disposed = true;
      return { active, disposed, originalAliasesRestored: true, totals, events, eventCount: events.length,
        intervals_include_nested_not_additive: true };
    },
  };
}
