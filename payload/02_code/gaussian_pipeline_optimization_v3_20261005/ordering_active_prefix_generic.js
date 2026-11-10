// Compare real integer texture prefixes and preserve initialization and unrelated bindings.
import * as THREE from "./three.module.js";
import { installOrderingActivePrefix } from "./ordering_active_prefix.js";

function equal(left, right, count = left.length) {
  if (left.length < count || right.length < count) return false;
  for (let i = 0; i < count; i++) if (left[i] !== right[i]) return false;
  return true;
}

export async function runOrderingPrefixGeneric() {
  const cases = [], failures = [];
  let extraGPUWriteBytes = 0, extraGPUReadBytes = 0;
  for (const enabled of [false, true]) {
    const canvas = document.createElement("canvas"), renderer = new THREE.WebGLRenderer({ canvas });
    const gl = renderer.getContext(), originalAlias = gl.texSubImage2D;
    const spark = { activeSplats: 0, current: { numSplats: 0, maxSplats: 0, mappingVersion: 0 }, maxSplats: 0, orderingTexture: null };
    const owned = [], helper = installOrderingActivePrefix({ renderer, spark, enabled });
    const read = texture => {
      const pixels = new Uint32Array(texture.image.width * texture.image.height * 4), framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, renderer.properties.get(texture).__webglTexture, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error("The original integer readback framebuffer is incomplete");
      gl.readPixels(0, 0, texture.image.width, texture.image.height, gl.RGBA_INTEGER, gl.UNSIGNED_INT, pixels);
      extraGPUReadBytes += pixels.byteLength;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.deleteFramebuffer(framebuffer);
      return pixels;
    };
    try {
      for (const rows of [1, 2, 4, 33]) {
        const initial = new Uint32Array(rows * 16384);
        for (let i = 0; i < initial.length; i++) initial[i] = (i * 2654435761) >>> 0;
        const originalBytes = initial.slice();
        const texture = new THREE.DataTexture(initial, 4096, rows, THREE.RGBAIntegerFormat, THREE.UnsignedIntType);
        texture.internalFormat = "RGBA32UI"; texture.needsUpdate = true;
        spark.orderingTexture = texture; spark.activeSplats = initial.length; spark.current.numSplats = initial.length;
        spark.current.maxSplats = initial.length; spark.maxSplats = initial.length;
        renderer.initTexture(texture); owned.push(texture); extraGPUWriteBytes += initial.byteLength;
        const imageData = texture.image.data;
        const counts = [...new Set([initial.length, 0, 1, 17, 16383, 16384, 16385, Math.min(60451, initial.length), initial.length - 1, initial.length].filter(n => n <= initial.length))];
        for (const active of counts) {
          spark.activeSplats = active; spark.current.numSplats = active; spark.current.mappingVersion++;
          const ordering = new Uint32Array(initial.length);
          for (let i = 0; i < ordering.length; i++) ordering[i] = i % 3 === 0 ? 0xffffffff : ((i * 2246822519) ^ active) >>> 0;
          const before = ordering.slice(), beforeImage = texture.image.data;
          gl.bindTexture(gl.TEXTURE_2D, renderer.properties.get(texture).__webglTexture);
          gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 4096, rows, gl.RGBA_INTEGER, gl.UNSIGNED_INT, ordering);
          const gpu = read(texture);
          const exact = equal(gpu, ordering, active) && equal(ordering, before) && equal(initial, originalBytes) && beforeImage === imageData && texture.image.data === imageData && texture.image.height === rows && gl.getError() === gl.NO_ERROR;
          const last = helper.snapshot().events.at(-1);
          cases.push({ enabled, rows, active, exact, uploadedRows: last.uploadedRows, requestedRows: last.requestedRows, wholeTextureTailCompared: active === initial.length });
          if (!exact) failures.push({ enabled, rows, active });
        }
        spark.activeSplats = 0;
        gl.bindTexture(gl.TEXTURE_2D, renderer.properties.get(texture).__webglTexture);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 4096, rows, gl.RGBA_INTEGER, gl.UNSIGNED_INT, imageData);
        const aliasEvent = helper.snapshot().events.at(-1);
        if (!equal(read(texture), imageData) || aliasEvent.uploadedRows !== rows || !aliasEvent.initialImageAlias) failures.push({ enabled, rows, originalImageAlias: true });
      }
      const orderingTexture = spark.orderingTexture;
      const otherData = new Uint32Array(32768); otherData.fill(0xdeadbeef);
      const other = new THREE.DataTexture(otherData, 4096, 2, THREE.RGBAIntegerFormat, THREE.UnsignedIntType);
      other.internalFormat = "RGBA32UI"; other.needsUpdate = true;
      renderer.initTexture(other); owned.push(other); extraGPUWriteBytes += otherData.byteLength;
      gl.bindTexture(gl.TEXTURE_2D, renderer.properties.get(other).__webglTexture);
      const beforeCalls = helper.snapshot().totals.orderingCalls;
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 4096, 2, gl.RGBA_INTEGER, gl.UNSIGNED_INT, otherData);
      extraGPUWriteBytes += otherData.byteLength;
      if (!equal(read(other), otherData) || helper.snapshot().totals.orderingCalls !== beforeCalls || spark.orderingTexture !== orderingTexture || gl.getError() !== gl.NO_ERROR) failures.push({ enabled, differentBinding: true });
      const statistics = helper.dispose();
      extraGPUWriteBytes += statistics.totals.orderingSubmittedBytes;
      if (!statistics.originalAliasRestored || gl.texSubImage2D !== originalAlias) failures.push({ enabled, originalAliasRestored: false });
    } finally {
      if (gl.texSubImage2D !== originalAlias) helper.dispose();
      for (const texture of owned) texture.dispose();
      renderer.dispose(); renderer.forceContextLoss();
    }
  }
  return { checks: { all_real_integer_texture_active_prefixes_exact: cases.every(row => row.exact),
    boundaries_zero_partial_full_high_bit_and_growth_shrink_covered: cases.length >= 56,
    original_arrays_image_alias_capacity_initialization_other_bindings_and_GL_errors_preserved: failures.length === 0,
    candidate_real_trim_calls_present: cases.some(row => row.enabled && row.uploadedRows < row.requestedRows),
    native_calls_original_full_rows: cases.filter(row => !row.enabled).every(row => row.uploadedRows === row.requestedRows) },
    cases, failures, extraGPUWriteBytes, extraGPUReadBytes, generic_only_not_original_Gaussian_render_or_context_restore: true };
}
