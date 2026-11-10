// Observe the original GPU ordering submission without changing its arguments.
export function installOrderingUploadCost({ renderer, spark }) {
  const gl = renderer.getContext(), original = gl.texSubImage2D;
  if (typeof original !== "function") throw new Error("The original WebGL upload API changed");
  const events = [];
  const totals = { allTexSubImage2DCalls: 0, orderingCalls: 0, orderingSubmittedBytes: 0,
    originalImageAliasCalls: 0, existingTextureUpdateCalls: 0, validPrefixBoundBytes: 0,
    unmodifiedTailBytesPotential: 0, originalCallMs: 0 };
  let disposed = false;
  const wrapped = function (...args) {
    if (disposed) throw new Error("The owned upload counter was already disposed");
    totals.allTexSubImage2DCalls++;
    const texture = spark.orderingTexture, property = texture ? renderer.properties.get(texture) : null;
    const matched = args.length === 9 && args[0] === gl.TEXTURE_2D && args[1] === 0 && args[2] === 0 && args[3] === 0 &&
      args[4] === 4096 && Number.isInteger(args[5]) && args[5] >= 1 && args[6] === gl.RGBA_INTEGER &&
      args[7] === gl.UNSIGNED_INT && args[8] instanceof Uint32Array && property?.__webglTexture &&
      gl.getParameter(gl.TEXTURE_BINDING_2D) === property.__webglTexture;
    const started = performance.now();
    const result = original.apply(this, args);
    if (matched) {
      if (events.length >= 4096) throw new Error("The registered upload observation bound was exceeded");
      const rows = args[5], bytes = 4096 * rows * 16, imageAlias = args[8] === texture.image.data;
      const active = spark.activeSplats, prefixRows = Math.max(1, Math.ceil(active / 16384));
      if (!Number.isInteger(active) || active < 0 || active > args[8].length || bytes > args[8].byteLength || prefixRows > rows) throw new Error("The original ordering prefix or texture dimensions violated the observed source contract");
      totals.orderingCalls++; totals.orderingSubmittedBytes += bytes;
      totals.originalImageAliasCalls += Number(imageAlias);
      totals.existingTextureUpdateCalls += Number(!imageAlias);
      totals.validPrefixBoundBytes += imageAlias ? bytes : prefixRows * 4096 * 16;
      totals.unmodifiedTailBytesPotential += imageAlias ? 0 : (rows - prefixRows) * 4096 * 16;
      totals.originalCallMs += performance.now() - started;
      events.push({ rows, submittedBytes: bytes, prefixRows, activeSplats: active,
        numSplats: spark.current?.numSplats, currentMaxSplats: spark.current?.maxSplats,
        rendererMaxSplats: spark.maxSplats, orderingArrayBytes: args[8].byteLength,
        textureHeight: texture.image.height, initialImageAlias: imageAlias,
        mappingVersion: spark.current?.mappingVersion });
    }
    return result;
  };
  gl.texSubImage2D = wrapped;
  return {
    snapshot: () => ({ totals: { ...totals }, events: events.map(row => ({ ...row })), disposed, unmodifiedOriginalArguments: true }),
    dispose() {
      if (disposed || gl.texSubImage2D !== wrapped) throw new Error("The owned original upload alias changed before restoration");
      gl.texSubImage2D = original; disposed = true;
      return { totals: { ...totals }, eventCount: events.length, disposed, originalAliasRestored: gl.texSubImage2D === original };
    },
  };
}
