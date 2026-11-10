import * as THREE from "three";
import { SplatGenerator, SplatEdit, PackedSplats } from "@sparkjsdev/spark";

// Bound updateInternal retains Spark's MIT attribution and every native statement except direction invalidation.
// Source: Spark, Copyright (c) 2025 WORLD LABS TECHNOLOGIES, INC.
// Every executed snapshot retains the original full SPARK-LICENSE.txt.
export function installRadialTranslationDependency({ spark, scene, renderer, SplatMesh, glGeneration, enabled }) {
  const originalUpdate = spark.updateInternal, originalReadback = spark.readbackDepth;
  const nativeSource = originalUpdate.toString();
  const originalDependency = "const viewChanged = center.distanceTo(this.sortedCenter) > .001 || dir.dot(this.sortedDir) < .999;";
  const replacementDependency = "const viewChanged = center.distanceTo(this.sortedCenter) > .001 || (!canIgnoreDirection(scene, camera, center) && dir.dot(this.sortedDir) < .999);";
  if (!nativeSource.startsWith("async updateInternal(") || nativeSource.split(originalDependency).length !== 2) throw new Error("The exact installed native update dependency changed");
  const body = nativeSource.replace("async updateInternal(", "async function updateInternal(").replace(originalDependency, replacementDependency);
  const inverse = body.replace("async function updateInternal(", "async updateInternal(").replace(replacementDependency, originalDependency);
  if (inverse !== nativeSource) throw new Error("The bound native update has unrelated changes");
  const bindings = new Map(), events = [];
  const totals = { nativeUpdates: 0, ignoredDirectionChecks: 0, eligibleDirectionChanges: 0,
    producerMatrixCalls: 0, suppressedUnusedProducerChanges: 0, nativeReadbacks: 0 };
  let active = 0, disposed = false, synchronousNode = null, completedReadbackGeneration = null, producerScopeAllowed = false;
  const originalMeshUpdate = SplatMesh.prototype.update;
  function nativeFrameUpdate(node) { return node.frameUpdate?.toString() === "(context) => this.update(context)"; }
  function eligible(node) {
    const binding = bindings.get(node), nativePacked = node.splats ?? node.packedSplats;
    return node.constructor === SplatMesh && node.isInitialized && binding &&
      node.update === binding.update && binding.originalNodeUpdate === originalMeshUpdate && nativeFrameUpdate(node) &&
      node.constructGenerator === SplatMesh.prototype.constructGenerator && nativePacked?.constructor === PackedSplats &&
      node.context.splats === nativePacked && nativePacked.fetchSplat === PackedSplats.prototype.fetchSplat && !node.covSplats && !node.extSplats && !node.paged &&
      !node.enableViewToObject && !node.enableViewToWorld && !node.enableWorldToView && !node.skinning && !node.onFrame &&
      !node.splatRgba && !node.rgbaDisplaceEdits && !(node.objectModifiers?.length) && !(node.worldModifiers?.length) &&
      !(node.covObjectModifiers?.length) && !(node.covWorldModifiers?.length) && !(node.edits?.length) &&
      !node.context.enableLod.value && !nativePacked?.lodSplats && !node.generatorDirty;
  }
  function bindModels() {
    scene.traverse(node => {
      if (!(node instanceof SplatMesh) || bindings.has(node)) return;
      const transform = node.context.viewToObject, originalMatrix = transform.updateFromMatrix, originalNodeUpdate = node.update;
      const update = function (...args) {
        const previousNode = synchronousNode; synchronousNode = node;
        try { return originalNodeUpdate.apply(this, args); }
        finally { synchronousNode = previousNode; }
      };
      const matrix = function (...args) {
        const previousTranslate = this.translate.value.clone(), changed = originalMatrix.apply(this, args);
        totals.producerMatrixCalls++;
        if (enabled && producerScopeAllowed && synchronousNode === node && eligible(node) && changed && previousTranslate.equals(this.translate.value)) {
          totals.suppressedUnusedProducerChanges++;
          return false;
        }
        return changed;
      };
      node.update = update; transform.updateFromMatrix = matrix;
      bindings.set(node, { update, matrix, originalNodeUpdate, originalMatrix, transform });
    });
  }
  function allVisibleConsumersEligible(camera) {
    let allEligible = true, count = 0;
    scene.traverseVisible(node => {
      if (node instanceof SplatEdit) allEligible = false;
      if (!(node instanceof SplatGenerator) || !camera.layers.test(node.layers)) return;
      count++;
      if (!eligible(node)) allEligible = false;
    });
    return allEligible && count > 0;
  }
  function canIgnoreDirection(actualScene, camera, center) {
    if (!enabled || actualScene !== scene || renderer.xr.isPresenting || !(spark.sortRadial ?? true) ||
        completedReadbackGeneration !== glGeneration.snapshot().generation || spark.sorting || spark.sortDirty ||
        !spark.orderingTexture || !center.equals(spark.sortedCenter) || !center.equals(spark.current.viewOrigin) ||
        spark.current.mappingVersion !== spark.display.mappingVersion) return false;
    if (!allVisibleConsumersEligible(camera)) return false;
    totals.ignoredDirectionChecks++;
    const dir = camera.getWorldDirection(center.clone());
    if (dir.dot(spark.sortedDir) < .999) {
      totals.eligibleDirectionChanges++;
      if (events.length >= 4096) throw new Error("The registered radial dependency event limit was exceeded");
      events.push({ kind: "eligible_direction", center: center.toArray(), direction: dir.toArray(),
        version: spark.current.version, mappingVersion: spark.current.mappingVersion, glGeneration: completedReadbackGeneration });
    }
    return true;
  }
  const compiled = new Function("THREE", "SplatMesh", "canIgnoreDirection", "return (" + body + ");")(THREE, SplatMesh, canIgnoreDirection);
  const update = async function (...args) {
    if (disposed) throw new Error("The owned radial dependency helper was disposed");
    bindModels(); totals.nativeUpdates++; active++;
    const previousScope = producerScopeAllowed;
    producerScopeAllowed = enabled && args[0]?.autoUpdate === true && (spark.sortRadial ?? true) &&
      args[0]?.scene === scene && !renderer.xr.isPresenting && allVisibleConsumersEligible(args[0].camera);
    let promise;
    try { promise = (enabled ? compiled : originalUpdate).apply(this, args); }
    finally { producerScopeAllowed = previousScope; }
    try { return await promise; }
    finally { active--; }
  };
  const readback = async function (...args) {
    const generation = glGeneration.snapshot().generation;
    totals.nativeReadbacks++; active++;
    try {
      const result = await originalReadback.apply(this, args);
      if (generation === glGeneration.snapshot().generation) completedReadbackGeneration = generation;
      return result;
    } finally { active--; }
  };
  spark.updateInternal = update; spark.readbackDepth = readback;
  function snapshot() {
    return { enabled, active, disposed, boundModels: bindings.size, totals: { ...totals }, events: [...events],
      fullOriginalMethodInverseExact: inverse === nativeSource, completedReadbackGeneration,
      original_data_sorting_projection_and_population_preserved: true };
  }
  return {
    snapshot,
    dispose() {
      if (disposed || active || spark.updateInternal !== update || spark.readbackDepth !== readback) throw new Error("The owned radial dependency work or aliases were not drained");
      spark.updateInternal = originalUpdate; spark.readbackDepth = originalReadback;
      for (const [node, binding] of bindings) {
        if (node.update !== binding.update || binding.transform.updateFromMatrix !== binding.matrix) throw new Error("An owned native producer alias changed");
        node.update = binding.originalNodeUpdate; binding.transform.updateFromMatrix = binding.originalMatrix;
      }
      disposed = true;
      return { ...snapshot(), originalAliasesRestored: true };
    },
  };
}
