// Derive a versioned visibility mask while preserving the native full ordering.
const fs = require("fs"), crypto = require("crypto");
const digest = bytes => crypto.createHash("sha256").update(bytes).digest("hex");

function once(source, before, after) {
  if (source.split(before).length !== 2) throw new Error("The original Spark frustum source anchor changed: " + before.slice(0, 80));
  return source.replace(before, after);
}

function derive(source) {
  let code = source;
  const types = Array.from({ length: 6 }, (_, i) => `\t\t\t\tr145Plane${i}: "vec4",`).join("\n");
  const inputs = Array.from({ length: 6 }, (_, i) => `\t\t\t\tr145Plane${i}: SplatAccumulator.r145Planes[${i}],`).join("\n");
  const planeNames = Array.from({ length: 6 }, (_, i) => "r145Plane" + i);
  let begin = code.indexOf("var OutputSplatDepth = class extends Dyno {");
  const end = code.indexOf("var OutputCovSplatDepth = class extends Dyno {", begin);
  if (begin < 0 || end < begin) throw new Error("The original depth class boundaries changed");
  let block = code.slice(begin, end);
  block = once(block, '\t\t\t\tsortRadial: "bool"\n', '\t\t\t\tsortRadial: "bool",\n\t\t\t\tr145Enabled: "bool",\n' + types + "\n");
  block = once(block, "\t\t\t\tsortRadial\n", "\t\t\t\tsortRadial,\n\t\t\t\tr145Enabled: SplatAccumulator.r145Enabled,\n" + inputs + "\n");
  block = once(block, "const { gsplat, viewCenter, viewDir, sortRadial } = inputs;", "const { gsplat, viewCenter, viewDir, sortRadial, r145Enabled, " + planeNames.join(", ") + " } = inputs;");
  const original = "                metric = dot(center, ${viewDir}) + bias;\n              }\n            }\n            target3 = floatToVec4(metric);";
  const tests = planeNames.map(name => `                (dot(${"${" + name + "}"}, r145Point) < -r145Margin * (dot(abs(${"${" + name + "}"}.xyz), vec3(1.0)) + abs(${"${" + name + "}"}.w)))`).join(" ||\n");
  const masked = `                metric = dot(center, \${viewDir}) + bias;
              }
              if (\${r145Enabled} && metric >= 0.0 && !isinf(metric) && !isnan(metric) && all(lessThan(abs(center), vec3(65504.0)))) {
                vec4 r145Point = vec4(\${gsplat}.center, 1.0);
                float r145Magnitude = 1.0 + dot(abs(\${gsplat}.center), vec3(1.0)) + dot(abs(\${viewCenter}), vec3(1.0));
                // Cover half re-encoding and all fixed FP32 transform operations conservatively.
                float r145Margin = 0.0117890 * r145Magnitude + 0.000000477;
                if (
${tests}
                ) { r145MaskBit = 0x80000000u; }
              }
            }
            target3 = uintToVec4(floatBitsToUint(metric) | r145MaskBit);`;
  block = once(block, original, masked);
  block = once(block, "float metric = 1.0 / 0.0;", "float metric = 1.0 / 0.0;\n            uint r145MaskBit = 0u;");
  code = code.slice(0, begin) + block + code.slice(end);
  code = once(code, "\t\tthis.sortRadialUniform = new DynoBool({ value: true });", "\t\tthis.sortRadialUniform = new DynoBool({ value: true });\n\t\tthis.r145Enabled = new DynoBool({ value: false });\n\t\tthis.r145Planes = Array.from({ length: 6 }, () => new DynoVec4({ value: new THREE.Vector4() }));");
  code = once(code, "prepareGenerate({ renderer, scene, timer, camera, sortRadial, renderSize, previous, lodInstances }) {", "prepareGenerate({ renderer, scene, timer, camera, sortRadial, renderSize, previous, lodInstances, r145Owner }) {\n\t\tr145Prepare(r145Owner, camera, renderer);");
  code = once(code, "\t\t\tlodInstances: this.enableLod ? this.lodInstances : void 0\n", "\t\t\tlodInstances: this.enableLod ? this.lodInstances : void 0,\n\t\t\tr145Owner: this\n");
  code = once(code, "\t\tconst viewChanged = center.distanceTo(this.sortedCenter) > .001 || dir.dot(this.sortedDir) < .999;", `\t\tconst r145Supported = r145CameraSupported(this, camera, renderer);
\t\tconst r145Key = r145Supported ? [...camera.matrixWorld.elements, ...camera.projectionMatrix.elements, this.clipXY].join(",") : null;
\t\tconst viewChanged = center.distanceTo(this.sortedCenter) > .001 || dir.dot(this.sortedDir) < .999 || r145Key !== (this.r145CameraKey ?? null);
`);
  code = once(code, "\t\t\tgenerate();\n\t\t\tif (this.flushAfterGenerate)", `\t\t\tgenerate();
\t\t\tthis.r145CameraKey = r145Key;
\t\t\tnext.r145MaskKey = r145Key;
\t\t\tnext.r145Epoch = this.r145Epoch = (this.r145Epoch ?? 0) + 1;
\t\t\tif (this.flushAfterGenerate)`);
  code = once(code, "\t\tconst { numSplats, maxSplats } = current;", `\t\tconst { numSplats, maxSplats } = current;
\t\tconst r145Stamp = { key: current.r145MaskKey, epoch: current.r145Epoch,
\t\t\tversion: current.version, mappingVersion: current.mappingVersion };`);
  code = once(code, "\t\tif (this.sortPause > 0) await new Promise((resolve) => setTimeout(resolve, this.sortPause));", `\t\tconst r145Mask = r145Stamp.key === null || r145Stamp.key === undefined ? null : new Uint8Array(numSplats);
\t\tif (r145Mask) {
\t\t\tfor (let i = 0; i < numSplats; i++) {
\t\t\t\tconst bits = readback[i];
\t\t\t\tif ((bits & 0x80000000) !== 0 && (bits & 0x7f800000) !== 0x7f800000) {
\t\t\t\t\tr145Mask[i] = 1;
\t\t\t\t\treadback[i] = bits & 0x7fffffff;
\t\t\t\t}
\t\t\t}
\t\t}
\t\tif (this.sortPause > 0) await new Promise((resolve) => setTimeout(resolve, this.sortPause));`);
  code = once(code, "\t\tif (this.current.mappingVersion === current.mappingVersion) {", "\t\tr145PublishMask(this, result, r145Mask, r145Stamp, rows);\n\t\tif (this.current.mappingVersion === current.mappingVersion) {");
  code = once(code, "\t\tspark.dirty = false;", "\t\tr145SelectMask(spark, this, camera, renderer);\n\t\tspark.dirty = false;");
  code = once(code, "\t\tconst accumulators = /* @__PURE__ */ new Set();", "\t\tthis.r145MaskedTexture?.dispose();\n\t\tthis.r145MaskedTexture = null;\n\t\tthis.r145MaskedOrdering = null;\n\t\tthis.r145MaskStamp = null;\n\t\tconst accumulators = /* @__PURE__ */ new Set();");
  const helpers = `
function r145CameraSupported(owner, camera, renderer) {
  return Boolean(owner && owner.sortRadial === true && camera.isPerspectiveCamera && !renderer.xr.isPresenting
    && typeof owner.display?.extSplats === "boolean" && owner.display?.covSplats === false
    && Number.isFinite(owner.clipXY) && owner.clipXY > 0 && camera.near > 0 && camera.far > camera.near
    && [...camera.matrixWorld.elements, ...camera.projectionMatrix.elements].every(Number.isFinite));
}
function r145Prepare(owner, camera, renderer) {
  const enabled = r145CameraSupported(owner, camera, renderer);
  SplatAccumulator.r145Enabled.value = enabled;
  if (!enabled) return;
  const matrix = camera.projectionMatrix.clone().multiply(camera.matrixWorldInverse);
  const e = matrix.elements, axis = owner.clipXY;
  const specifications = [[0, 1, axis], [0, -1, axis], [1, 1, axis], [1, -1, axis], [2, 1, 1], [2, -1, 1]];
  for (let i = 0; i < specifications.length; i++) {
    const [row, sign, wscale] = specifications[i];
    const value = SplatAccumulator.r145Planes[i].value;
    value.set(sign * e[row] + wscale * e[3], sign * e[row + 4] + wscale * e[7],
      sign * e[row + 8] + wscale * e[11], sign * e[row + 12] + wscale * e[15]);
  }
}
function r145PublishMask(owner, result, mask, stamp, rows) {
  if (!mask) { owner.r145MaskStamp = null; return; }
  const ordering = new Uint32Array(rows * 16384);
  let count = 0;
  for (let i = 0; i < result.activeSplats; i++) {
    const index = result.ordering[i];
    if (index >= mask.length) throw new Error("The native ordering escaped its visibility mask");
    if (mask[index] === 0) ordering[count++] = index;
  }
  if (owner.r145MaskedTexture && rows !== owner.r145MaskedTexture.image.height) {
    owner.r145MaskedTexture.dispose(); owner.r145MaskedTexture = null;
  }
  if (!owner.r145MaskedTexture) {
    const texture = new THREE.DataTexture(ordering, 4096, rows, THREE.RGBAIntegerFormat, THREE.UnsignedIntType);
    texture.internalFormat = "RGBA32UI"; texture.needsUpdate = true;
    owner.r145MaskedTexture = texture;
  } else {
    owner.r145MaskedTexture.image.data = ordering;
    if (!owner.renderer.properties.has(owner.r145MaskedTexture)) owner.r145MaskedTexture.needsUpdate = true;
    else uploadU32DataTextureRows(owner.renderer, owner.r145MaskedTexture, 4096, rows, ordering);
  }
  owner.r145MaskedCount = count;
  owner.r145MaskedOrdering = ordering.subarray(0, count);
  owner.r145MaskStamp = stamp;
}
function r145SelectMask(owner, mesh, camera, renderer) {
  const stamp = owner.r145MaskStamp, display = owner.display;
  const key = r145CameraSupported(owner, camera, renderer)
    ? [...camera.matrixWorld.elements, ...camera.projectionMatrix.elements, owner.clipXY].join(",") : null;
  const matches = stamp && stamp.key === key && key !== null && stamp.epoch === display.r145Epoch
    && stamp.version === display.version && stamp.mappingVersion === display.mappingVersion;
  if (matches) {
    mesh.geometry.instanceCount = owner.r145MaskedCount;
    mesh.uniforms.ordering.value = owner.r145MaskedTexture;
  }
  owner.r145MaskSelected = Boolean(matches);
}
`;
  return code + helpers;
}

function instrument(source) {
  let code = once(source, "\t\tthis.activeSplats = result.activeSplats;", "\t\tthis.activeSplats = result.activeSplats;\n\t\tthis.r145LastOrdering = new Uint32Array(result.ordering.subarray(0, result.activeSplats));\n\t\tthis.r145SortCount = (this.r145SortCount ?? 0) + 1;");
  code = once(code, "\t\tspark.dirty = false;", "\t\tspark.r145DrawActive = geometry.instanceCount;\n\t\tspark.r145DrawOrdering = this.uniforms.ordering.value === spark.r145MaskedTexture ? spark.r145MaskedOrdering : spark.r145LastOrdering;\n\t\tspark.dirty = false;");
  return code;
}

module.exports = { derive, instrument, digest, once };
