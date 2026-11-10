// Observe original controller and native version changes without updating matrices.
export function installViewerControllerTrace({ state, scene, camera, controls, renderer, spark, SplatMesh }) {
  const originalUpdate = controls.update, originalRender = renderer.render;
  const events = [];
  let disposed = false;
  const pose = () => ({ position: camera.position.toArray(), quaternion: camera.quaternion.toArray(),
    target: controls.target.toArray(), matrixWorld: camera.matrixWorld.toArray(),
    sphericalDelta: [controls._sphericalDelta.radius, controls._sphericalDelta.phi, controls._sphericalDelta.theta],
    panOffset: controls._panOffset.toArray(), scale: controls._scale, orbitState: controls.state,
    playing: state.playing, navigationMode: state.navigationMode, frame: state.frames });
  function native() {
    const models = [];
    scene.traverse(node => {
      if (node instanceof SplatMesh && node.visible) {
        const transform = node.context?.viewToObject;
        models.push({ numSplats: node.numSplats, version: node.version,
          translate: transform?.translate?.value?.toArray(), rotate: transform?.rotate?.value?.toArray(),
          scale: transform?.scale?.value });
      }
    });
    return { currentVersion: spark.current?.version, currentMappingVersion: spark.current?.mappingVersion,
      currentOrigin: spark.current?.viewOrigin.toArray(), currentDirection: spark.current?.viewDirection.toArray(),
      sortedCenter: spark.sortedCenter.toArray(), sortedDir: spark.sortedDir.toArray(),
      sorting: spark.sorting, sortDirty: spark.sortDirty, lastSortTime: spark.lastSortTime, models };
  }
  function record(event) {
    if (events.length >= 10000) throw new Error("The registered controller trace event bound was exceeded");
    events.push(event);
  }
  const update = function (...args) {
    const before = pose(), started = performance.now();
    const changed = originalUpdate.apply(this, args);
    record({ kind: "controls", started, elapsedMs: performance.now() - started, before, after: pose(), changed });
    return changed;
  };
  const render = function (...args) {
    if (args[0] !== scene) return originalRender.apply(this, args);
    const before = pose(), beforeNative = native(), started = performance.now();
    const result = originalRender.apply(this, args);
    record({ kind: "mainRender", started, elapsedMs: performance.now() - started, before, after: pose(),
      beforeNative, afterNative: native(), dimensions: [renderer.domElement.width, renderer.domElement.height] });
    return result;
  };
  controls.update = update; renderer.render = render;
  return {
    dispose() {
      if (disposed || controls.update !== update || renderer.render !== render) throw new Error("An owned controller trace hook changed");
      controls.update = originalUpdate; renderer.render = originalRender; disposed = true;
      return { disposed, originalUpdate: controls.update === originalUpdate, originalRender: renderer.render === originalRender,
        events, eventCount: events.length };
    },
  };
}
