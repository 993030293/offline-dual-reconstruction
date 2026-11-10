// Preserve ordering storage and initialization while limiting existing GPU updates to readable rows.
export function installOrderingActivePrefix({ renderer, spark, enabled = false }) {
  const gl = renderer.getContext(), original = gl.texSubImage2D;
  if (typeof original !== "function") throw new Error("The original WebGL upload API changed");
  const events = [];
  const totals = { allTexSubImage2DCalls: 0, orderingCalls: 0, orderingSubmittedBytes: 0,
    originalRequestedBytes: 0, originalImageAliasCalls: 0, existingTextureUpdateCalls: 0,
    trimmedCalls: 0, trimmedBytes: 0, originalCallMs: 0 };
  let disposed = false;
  const wrapped = function (...args) {
    if (disposed) throw new Error("The owned upload helper was already disposed");
    totals.allTexSubImage2DCalls++;
    const texture = spark.orderingTexture, property = texture ? renderer.properties.get(texture) : null;
    const matched = args.length === 9 && args[0] === gl.TEXTURE_2D && args[1] === 0 && args[2] === 0 && args[3] === 0 &&
      args[4] === 4096 && Number.isInteger(args[5]) && args[5] >= 1 && args[6] === gl.RGBA_INTEGER &&
      args[7] === gl.UNSIGNED_INT && args[8] instanceof Uint32Array && property?.__webglTexture &&
      gl.getParameter(gl.TEXTURE_BINDING_2D) === property.__webglTexture;
    let record;
    if (matched) {
      if (events.length >= 4096) throw new Error("The registered upload observation bound was exceeded");
      const requestedRows = args[5], imageAlias = args[8] === texture.image.data, active = spark.activeSplats;
      const prefixRows = Math.max(1, Math.ceil(active / 16384));
      if (!Number.isInteger(active) || active < 0 || active > args[8].length || requestedRows * 16384 > args[8].length || prefixRows > requestedRows || requestedRows > texture.image.height) throw new Error("The original ordering prefix or allocated texture bounds changed");
      if (enabled && !imageAlias) args[5] = prefixRows;
      record = { requestedRows, uploadedRows: args[5], prefixRows, activeSplats: active,
        submittedBytes: args[5] * 65536, originalRequestedBytes: requestedRows * 65536,
        numSplats: spark.current?.numSplats, currentMaxSplats: spark.current?.maxSplats,
        rendererMaxSplats: spark.maxSplats, orderingArrayBytes: args[8].byteLength,
        textureHeight: texture.image.height, initialImageAlias: imageAlias,
        mappingVersion: spark.current?.mappingVersion };
    }
    const started = performance.now(), result = original.apply(this, args);
    if (record) {
      totals.orderingCalls++; totals.orderingSubmittedBytes += record.submittedBytes;
      totals.originalRequestedBytes += record.originalRequestedBytes;
      totals.originalImageAliasCalls += Number(record.initialImageAlias);
      totals.existingTextureUpdateCalls += Number(!record.initialImageAlias);
      totals.trimmedCalls += Number(record.uploadedRows < record.requestedRows);
      totals.trimmedBytes += record.originalRequestedBytes - record.submittedBytes;
      totals.originalCallMs += performance.now() - started;
      events.push(record);
    }
    return result;
  };
  gl.texSubImage2D = wrapped;
  return {
    snapshot: () => ({ totals: { ...totals }, events: events.map(row => ({ ...row })), disposed, enabled }),
    dispose() {
      if (disposed || gl.texSubImage2D !== wrapped) throw new Error("The owned original upload alias changed before restoration");
      gl.texSubImage2D = original; disposed = true;
      return { totals: { ...totals }, eventCount: events.length, disposed, enabled, originalAliasRestored: gl.texSubImage2D === original };
    },
  };
}
