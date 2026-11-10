import * as THREE from "three";
import { SplatMesh, SplatGenerator, ExtSplats } from "@sparkjsdev/spark";

// Exercise dependency changes on the actual loaded scene and original native constructors.
export async function runRadialDependencyGeneric({ state, scene, camera, spark, cache }) {
  const fixtures = [], keys = ["rgba_sha256", "position", "direction", "quaternion", "world", "projection", "dimensions", "count", "mode", "activeSplats", "currentSplats"];
  let model;
  scene.traverseVisible(node => { if (node.constructor === SplatMesh) model = node; });
  if (!model || !model.isInitialized) throw new Error("The real primary Gaussian model was unavailable");
  const { position, direction } = state.radialGenericOrigin;
  const original = { sortRadial: spark.sortRadial, frameUpdate: model.frameUpdate, onFrame: model.onFrame,
    objectModifiers: model.objectModifiers, position: model.position.clone(), recolor: model.recolor.clone(),
    flags: [model.enableViewToObject, model.enableViewToWorld, model.enableWorldToView] };
  async function settle(previous) {
    const until = performance.now() + 15000;
    while (performance.now() < until) {
      const p = state.costProbe.snapshot();
      if (state.frames > previous && !state.modelLoading && !state.isRenderPending() && !p.sorting && !p.sortDirty && !p.sortTimeout && p.active === 0 && cache.snapshot().active === 0) return;
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
    throw new Error("The registered native dependency fixture did not drain");
  }
  async function pose(angle) {
    const c = Math.cos(angle), s = Math.sin(angle), d = direction;
    const changedDirection = [c * d[0] + s * d[2], d[1], -s * d[0] + c * d[2]];
    const previous = state.frames;
    state.costProbe.freeCamera(position, changedDirection);
    await settle(previous);
    return state.costProbe.capture();
  }
  async function force() {
    const before = cache.snapshot().totals.nativeReadbacks, previous = state.frames;
    await spark.update({ scene, camera });
    await settle(previous);
    return cache.snapshot().totals.nativeReadbacks - before;
  }
  const cases = [
    { name: "static_packed_pure_orientation", apply: () => () => {}, unsupported: false },
    { name: "explicit_native_force", apply: () => () => {}, unsupported: false, forced: true },
    { name: "nonradial_direction_sort", apply: () => { spark.sortRadial = false; return () => { spark.sortRadial = original.sortRadial; }; }, unsupported: true },
    ...["enableViewToObject", "enableViewToWorld", "enableWorldToView"].map((flag, i) => ({ name: flag, unsupported: true,
      apply: () => { model[flag] = true; return () => { model[flag] = original.flags[i]; }; } })),
    { name: "native_onFrame_callback", unsupported: true, apply: () => { model.onFrame = () => {}; return () => { model.onFrame = original.onFrame; }; } },
    { name: "unknown_frameUpdate_wrapper", unsupported: true, apply: () => { model.frameUpdate = context => original.frameUpdate(context); return () => { model.frameUpdate = original.frameUpdate; }; } },
    { name: "unknown_visible_native_generator", unsupported: true, apply: () => { const extra = new SplatGenerator({ numSplats: 0, update: () => {} }); scene.add(extra); return () => { scene.remove(extra); }; } },
    { name: "native_passthrough_object_modifier", unsupported: true, apply: () => {
      model.objectModifiers = [{ apply: ({ gsplat }) => ({ gsplat }) }]; model.updateGenerator();
      return () => { model.objectModifiers = original.objectModifiers; model.updateGenerator(); };
    } },
    { name: "native_empty_ExtSplats_producer_guard", unsupported: true, apply: () => {
      const extra = new SplatMesh({ extSplats: new ExtSplats(), editable: false }); scene.add(extra);
      return () => { scene.remove(extra); };
    } },
    { name: "native_empty_covariance_producer_guard", unsupported: true, apply: () => {
      const extra = new SplatMesh({ extSplats: new ExtSplats(), covSplats: true, editable: false }); scene.add(extra);
      return () => { scene.remove(extra); };
    } },
    { name: "actual_source_version_signal", unsupported: false, apply: () => { model.updateVersion(); return () => {}; } },
    { name: "actual_recolor_update", unsupported: false, apply: () => { model.recolor.multiplyScalar(.8); return () => { model.recolor.copy(original.recolor); }; } },
    { name: "actual_model_translation_update", unsupported: false, apply: () => { model.position.add(new THREE.Vector3(.002, .003, -.001)); return () => { model.position.copy(original.position); }; } },
    { name: "two_static_native_meshes_same_immutable_source", unsupported: false, apply: () => {
      const extra = new SplatMesh({ splats: model.splats ?? model.packedSplats, editable: false });
      extra.position.copy(model.position); extra.quaternion.copy(model.quaternion); extra.scale.copy(model.scale); scene.add(extra);
      return () => { scene.remove(extra); };
    } },
  ];
  try {
    for (const item of cases) {
      const restore = item.apply();
      try {
        const reset = await pose(0), forcedResetSorts = await force();
        const before = cache.snapshot();
        const first = await pose(Math.PI / 12), midpoint = cache.snapshot();
        const second = await pose(Math.PI / 6), after = cache.snapshot();
        const explicitForceSorts = item.forced ? await force() : null;
        const nativeErrors = [];
        scene.traverseVisible(node => { if (node instanceof SplatGenerator && node.generatorError) nativeErrors.push(String(node.generatorError)); });
        const passed = forcedResetSorts > 0 && reset.glError === 0 && first.glError === 0 && second.glError === 0 && !nativeErrors.length &&
          (!item.forced || explicitForceSorts > 0) && (!item.unsupported || after.totals.suppressedUnusedProducerChanges === before.totals.suppressedUnusedProducerChanges && after.totals.nativeReadbacks > midpoint.totals.nativeReadbacks);
        fixtures.push({ name: item.name, unsupported: item.unsupported, forcedResetSorts, explicitForceSorts,
          reset, first, second, before, midpoint, after, nativeErrors, passed });
        if (!passed) throw new Error("The actual native dependency fixture failed: " + item.name);
      } finally { restore(); }
      await pose(0); await force();
    }
  } finally {
    spark.sortRadial = original.sortRadial; model.frameUpdate = original.frameUpdate; model.onFrame = original.onFrame;
    model.objectModifiers = original.objectModifiers; model.position.copy(original.position); model.recolor.copy(original.recolor);
    [model.enableViewToObject, model.enableViewToWorld, model.enableWorldToView] = original.flags;
  }
  return { status: "actual_native_dependency_components_not_formal_speed", fixtures, keys,
    checks: { all16_actual_native_components_passed: fixtures.length === 16 && fixtures.every(row => row.passed),
      original_camera_origin_all_components_preserved: fixtures.every(row => [row.reset, row.first, row.second].every(c => JSON.stringify(c.position) === JSON.stringify(position))),
      no_valid_nonempty_covariance_LOD_or_unknown_object_GT_claim: true },
    empty_Ext_cov_guard_cases_are_native_zero_population_not_nonempty_format_quality: true,
    actual_nonempty_LOD_source_not_available_not_rendered: true, new_optimization_count: 0 };
}
