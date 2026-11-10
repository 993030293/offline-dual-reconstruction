// Bind exact static SH origin dependencies in an owned native renderer only.
export function createOriginDependency({ THREE, NativeSparkRenderer, NativeSplatMesh,
  NativePackedSplats, NativeSplatGenerator, NativeSplatTransformer }) {
  const originalUpdate = NativeSparkRenderer.prototype.updateInternal;
  const originalMeshUpdate = NativeSplatMesh.prototype.update;
  const anchor = "const viewChanged = center.distanceTo(this.sortedCenter) > .001 || dir.dot(this.sortedDir) < .999;";
  const source = originalUpdate.toString();
  if (source.split(anchor).length !== 2 || !originalMeshUpdate.toString().includes(
    "this.context.viewToObject.updateFromMatrix(viewToObjectMatrix)")) {
    throw new Error("The original native update dependency source differs");
  }
  const replacement = "const viewChanged = center.distanceTo(this.sortedCenter) > .001 || " +
    "(!this.originDependencyEligible && dir.dot(this.sortedDir) < .999);";
  const boundUpdate = new Function("THREE", "SplatMesh", "return ({" +
    source.replace(anchor, replacement) + "}).updateInternal;")(THREE, NativeSplatMesh);
  const records = new Map();
  const owners = new WeakSet();
  const meshMethods = Object.fromEntries(["update", "constructGenerator", "updateGenerator",
    "updateVersion", "updateMappingVersion"].map(name => [name, NativeSplatMesh.prototype[name]]));
  const packedMethods = Object.fromEntries(["fetchSplat", "prepareFetchSplat", "getNumSplats",
    "getNumSh", "hasRgbDir", "setMaxSh", "getTexture", "maybeUpdateSource", "ensureShTextures"]
    .map(name => [name, NativePackedSplats.prototype[name]]));
  const nativeTransform = NativeSplatTransformer.prototype.updateFromMatrix;
  const nativeTransformUpdate = NativeSplatTransformer.prototype.update;
  let active = null;
  let activeOwner = null;
  const trace = { admitted: 0, original: 0, suppressed: 0, completeInitialWords: 0 };

  function arrays(packed) {
    return [packed.packedArray, packed.extra.sh1, packed.extra.sh2, packed.extra.sh3];
  }

  function sameData(row, packed) {
    const current = arrays(packed);
    if (packed.numSplats !== row.numSplats || packed.maxSplats !== row.maxSplats ||
        packed.maxSh !== 3 || packed.getNumSh() !== 3 || packed.target ||
        JSON.stringify(packed.splatEncoding) !== row.encoding) return false;
    if (current.some((actual, part) => actual !== row.arrays[part] || actual.length !== row.lengths[part])) return false;
    // The caller owns static buffers; foreign unannounced in-place writes are outside this contract.
    if (row.warm && (packed.needsUpdate || packed.source !== row.source ||
        packed.source?.version !== row.sourceVersion ||
        ["sh1Texture", "sh2Texture", "sh3Texture"].some((key, i) =>
          packed.extra[key]?.value !== row.shTextures[i] ||
          packed.extra[key]?.value.version !== row.shVersions[i]))) return false;
    return true;
  }

  function recordNativeState(mesh, row) {
    row.generator = mesh.generator;
    row.version = mesh.version;
    row.source = row.packed.source;
    row.sourceVersion = row.source?.version;
    row.shTextures = ["sh1Texture", "sh2Texture", "sh3Texture"].map(key => row.packed.extra[key]?.value);
    row.shVersions = row.shTextures.map(texture => texture?.version);
    row.warm = true;
  }

  function compatible(mesh) {
    const row = records.get(mesh);
    const packed = mesh.splats;
    return Boolean(row && Object.getPrototypeOf(mesh) === NativeSplatMesh.prototype && mesh.isInitialized &&
      mesh.context === row.meshContext && mesh.context.viewToObject === row.context &&
      row.context.updateFromMatrix === row.binding && mesh.frameUpdate === row.frameUpdate &&
      mesh.generator === row.generator && mesh.covGenerator === undefined &&
      (!row.warm || mesh.version === row.version && !mesh.generatorDirty) &&
      Object.entries(meshMethods).every(([name, method]) => mesh[name] === method) &&
      [mesh.context.transform, mesh.context.viewToWorld, mesh.context.worldToView].every(context =>
        Object.getPrototypeOf(context) === NativeSplatTransformer.prototype &&
        context.updateFromMatrix === nativeTransform && context.update === nativeTransformUpdate) &&
      !mesh.enableViewToObject && !mesh.enableViewToWorld &&
      !mesh.enableWorldToView && !mesh.covSplats && !mesh.extSplats && !mesh.paged &&
      !mesh.skinning && !mesh.edits && !mesh.rgbaDisplaceEdits && !mesh.splatRgba && !mesh.onFrame &&
      !mesh.objectModifiers?.length && !mesh.worldModifiers?.length &&
      packed instanceof NativePackedSplats && Object.getPrototypeOf(packed) === NativePackedSplats.prototype &&
      packed === row.packed && packed.isInitialized && mesh.maxSh === 3 &&
      Object.entries(packedMethods).every(([name, method]) => packed[name] === method) &&
      !packed.lod && !packed.lodSplats && !mesh.context.enableLod.value && sameData(row, packed));
  }

  function install(mesh, owner) {
    if (records.has(mesh)) throw new Error("The owned mesh already has a dependency binding");
    if (!owners.has(owner) || Object.getPrototypeOf(mesh) !== NativeSplatMesh.prototype ||
        !mesh.isInitialized || !mesh.context?.viewToObject ||
        Object.getPrototypeOf(mesh.splats) !== NativePackedSplats.prototype ||
        !arrays(mesh.splats).every(array => array instanceof Uint32Array)) {
      throw new Error("An original owned SplatMesh is required before binding");
    }
    const context = mesh.context.viewToObject;
    const original = context.updateFromMatrix;
    if (Object.getPrototypeOf(context) !== NativeSplatTransformer.prototype || original !== nativeTransform) {
      throw new Error("An unmodified native view context is required before binding");
    }
    const own = Object.hasOwn(context, "updateFromMatrix");
    const binding = function (matrix) {
      const previous = context.translate.value;
      const x = previous.x, y = previous.y, z = previous.z;
      const changed = original.call(this, matrix);
      const current = context.translate.value;
      if (changed && activeOwner === owner && active?.has(mesh) && Object.is(x, current.x) &&
          Object.is(y, current.y) && Object.is(z, current.z)) {
        trace.suppressed++;
        return false;
      }
      return changed;
    };
    context.updateFromMatrix = binding;
    const packed = mesh.splats, data = arrays(packed);
    trace.completeInitialWords += data.reduce((sum, array) => sum + array.length, 0);
    records.set(mesh, { context, original, binding, own, owner, packed,
      meshContext: mesh.context, frameUpdate: mesh.frameUpdate, generator: mesh.generator,
      arrays: data, lengths: data.map(array => array.length), warm: false, numSplats: packed.numSplats,
      maxSplats: packed.maxSplats, encoding: JSON.stringify(packed.splatEncoding) });
    return mesh;
  }

  function restore(mesh) {
    const row = records.get(mesh);
    if (!row) throw new Error("The owned mesh has no dependency binding");
    if (row.context.updateFromMatrix !== row.binding) {
      throw new Error("Another writer changed the owned context binding");
    }
    if (row.own) row.context.updateFromMatrix = row.original;
    else delete row.context.updateFromMatrix;
    records.delete(mesh);
  }

  class OriginSparkRenderer extends NativeSparkRenderer {
    constructor(options) {
      super(options);
      owners.add(this);
    }
    async updateInternal(args) {
      const eligible = new Set();
      let safe = active === null && args.autoUpdate === true && this.sortRadial === true &&
        !this.accumExtSplats && !this.covSplats && !this.pagedExtSplats && !NativeSparkRenderer.sparkOverride;
      args.scene.traverse(node => {
        if (node.isSplatEdit || node.isSplatEditSdf) safe = false;
        if (!(node instanceof NativeSplatGenerator)) return;
        if (records.get(node)?.owner !== this || !compatible(node)) safe = false;
        else eligible.add(node);
      });
      safe &&= eligible.size > 0;
      if (!safe) {
        trace.original++;
        const savedActive = active, savedOwner = activeOwner;
        active = null;
        activeOwner = null;
        let pending;
        try {
          pending = originalUpdate.call(this, args);
        } finally {
          active = savedActive;
          activeOwner = savedOwner;
        }
        return await pending;
      }
      trace.admitted++;
      this.originDependencyEligible = true;
      active = eligible;
      activeOwner = this;
      try {
        const pending = boundUpdate.call(this, args);
        // All mesh frameUpdate calls finish before the native sort's first await.
        active = null;
        activeOwner = null;
        this.originDependencyEligible = false;
        for (const mesh of eligible) recordNativeState(mesh, records.get(mesh));
        return await pending;
      } finally {
        active = null;
        activeOwner = null;
        this.originDependencyEligible = false;
      }
    }
  }
  return { SparkRenderer: OriginSparkRenderer, install, restore,
    bindingCount: () => records.size, compatible, originalUpdateSource: source, trace,
    restoreAll: () => { for (const mesh of [...records.keys()]) restore(mesh); } };
}
