// Skip only a fixed native renderer whose static centers are provably behind the camera.
const HALF_MAX = 65504;
const U32 = 2 ** -24;
const U16 = 2 ** -11;
const GAMMA32 = 128 * U32 / (1 - 128 * U32);
const GAMMA16 = 8 * U16 / (1 - 8 * U16);
let halfTable;

function halfValues() {
  if (halfTable) return halfTable;
  const values = new Float64Array(65536);
  for (let i = 0; i < values.length; i++) {
    const sign = i & 32768 ? -1 : 1;
    const exponent = (i >>> 10) & 31, mantissa = i & 1023;
    values[i] = exponent === 31 ? (mantissa ? NaN : sign * Infinity)
      : sign * (exponent ? (1 + mantissa / 1024) * 2 ** (exponent - 15) : mantissa * 2 ** -24);
  }
  halfTable = values;
  return values;
}

export function packedCenterBounds(packed) {
  const array = packed?.packedArray, count = packed?.numSplats;
  if (!(array instanceof Uint32Array) || !Number.isSafeInteger(count) || count <= 0
      || array.length < count * 4 || packed.lod || packed.splatEncoding?.lodOpacity !== false) {
    return { supported: false, reason: "unsupported_fixed_packed_layout" };
  }
  const table = halfValues(), minimum = [Infinity, Infinity, Infinity], maximum = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < count; i++) {
    const word1 = array[i * 4 + 1], word2 = array[i * 4 + 2];
    const x = table[word1 & 65535], y = table[word1 >>> 16], z = table[word2 & 65535];
    if (![x, y, z].every(Number.isFinite)) return { supported: false, reason: "nonfinite_center" };
    minimum[0] = Math.min(minimum[0], x); maximum[0] = Math.max(maximum[0], x);
    minimum[1] = Math.min(minimum[1], y); maximum[1] = Math.max(maximum[1], y);
    minimum[2] = Math.min(minimum[2], z); maximum[2] = Math.max(maximum[2], z);
  }
  return { supported: true, minimum, maximum, count };
}

export function behindHalfSpace(bounds, position, forward) {
  if (!bounds.supported || ![...position, ...forward].every(Number.isFinite)) {
    return { rejected: false, reason: "nonfinite_or_unsupported_geometry" };
  }
  const norm = Math.hypot(...forward);
  if (Math.abs(norm - 1) > 64 * Number.EPSILON) return { rejected: false, reason: "nonunit_camera" };
  let maximumDepth = 0, magnitude = 1;
  for (let axis = 0; axis < 3; axis++) {
    const lo = bounds.minimum[axis] - position[axis], hi = bounds.maximum[axis] - position[axis];
    if (Math.max(Math.abs(lo), Math.abs(hi)) >= HALF_MAX) return { rejected: false, reason: "half_range" };
    maximumDepth += forward[axis] * (forward[axis] >= 0 ? hi : lo);
    magnitude += Math.abs(position[axis]) + Math.max(Math.abs(bounds.minimum[axis]), Math.abs(bounds.maximum[axis]));
  }
  // Cover the fixed transform arithmetic and the native accumulator's half re-encoding.
  const margin = 3 * (GAMMA32 + GAMMA16) * magnitude + 8 * 2 ** -24;
  return { rejected: maximumDepth < -margin, maximumDepth, margin, reason: "bounded_half_space" };
}

function identityTransform(matrix) {
  return matrix.elements.every((value, index) => value === (index % 5 === 0 ? 1 : 0));
}

export function createBehindCameraCuller({ scene, spark, SplatMesh, enabled = true }) {
  const bounds = new WeakMap();
  const trace = { frames: 0, rejectedFrames: 0, nativeFrames: 0, builds: 0, buildMs: 0, last: null, disposed: false };
  function beforeFrame(camera) {
    trace.frames++;
    spark.visible = true;
    if (!enabled || trace.disposed) { trace.nativeFrames++; return; }
    scene.updateMatrixWorld(true);
    camera.updateMatrixWorld(true);
    const models = [];
    scene.traverseVisible(node => { if (node instanceof SplatMesh) models.push(node); });
    const position = camera.position.toArray(), forward = camera.getWorldDirection(camera.position.clone()).toArray();
    const supportedCamera = camera.isPerspectiveCamera && camera.parent === null && camera.near > 0
      && camera.far > camera.near && !spark.renderer.xr.isPresenting;
    let rejected = supportedCamera && models.length > 0;
    const rows = [];
    for (const model of models) {
      const packed = model.splats;
      if (!identityTransform(model.matrixWorld) || model.objectModifiers?.length || model.worldModifiers?.length
          || model.edits?.length || model.skinning || model.paged || model.onFrame || !model.isInitialized
          || packed?.needsUpdate) {
        bounds.delete(model);
        rejected = false;
        rows.push({ rejected: false, reason: "dynamic_or_unsupported_model" });
        continue;
      }
      let record = bounds.get(model);
      if (!record || record.packed !== packed || record.array !== packed.packedArray || record.count !== model.numSplats) {
        const started = performance.now();
        record = { packed, array: packed.packedArray, count: model.numSplats, bounds: packedCenterBounds(packed) };
        trace.buildMs += performance.now() - started; trace.builds++;
        bounds.set(model, record);
      }
      const decision = behindHalfSpace(record.bounds, position, forward);
      rows.push(decision); rejected &&= decision.rejected;
    }
    spark.visible = !rejected;
    if (rejected) trace.rejectedFrames++; else trace.nativeFrames++;
    trace.last = { rejected, supportedCamera, models: rows };
  }
  function dispose() { spark.visible = true; trace.disposed = true; }
  return { beforeFrame, dispose, trace };
}

export function installDrawCounter(gl) {
  const trace = { calls: 0, triangles: 0, vertices: 0, byMethod: {} };
  const originals = new Map();
  for (const name of ["drawArrays", "drawElements", "drawArraysInstanced", "drawElementsInstanced"]) {
    const original = gl[name]; originals.set(name, original);
    gl[name] = function (...args) {
      const result = original.apply(this, args);
      const mode = args[0], count = name.includes("Elements") ? args[1] : args[2];
      const instances = name.endsWith("Instanced") ? args[4 - Number(name.includes("Arrays"))] : 1;
      const triangles = mode === gl.TRIANGLES ? count / 3
        : mode === gl.TRIANGLE_STRIP || mode === gl.TRIANGLE_FAN ? Math.max(0, count - 2) : 0;
      trace.calls++; trace.vertices += count * instances; trace.triangles += triangles * instances;
      trace.byMethod[name] = (trace.byMethod[name] ?? 0) + 1;
      return result;
    };
  }
  return { trace, dispose() { for (const [name, original] of originals) gl[name] = original; } };
}
