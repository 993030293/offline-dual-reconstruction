// Isolate native GPU payload and ordering from visibility-mask selection.
const fs = require("fs"), path = require("path"), Module = require("module");
const driver = path.join(__dirname, "run_frustum_sort.cjs");
const { once } = require("./frustum_sort_build.cjs");
const main = `
async function inspectGpu(page, name) {
  const record = await page.evaluate(() => {
    const state = window.box2Viewer, spark = state.r138.spark, renderer = state.renderer, gl = renderer.getContext();
    const read = (texture, arrayTexture) => {
      const handle = renderer.properties.get(texture)?.__webglTexture;
      if (!handle) throw new Error("The original GPU texture has no owned handle");
      const { width, height, depth = 1 } = texture.image;
      const bytes = new Uint32Array(width * height * depth * 4);
      const previousRead = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING), framebuffer = gl.createFramebuffer();
      try {
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, framebuffer);
        for (let layer = 0; layer < depth; layer++) {
          if (arrayTexture) gl.framebufferTextureLayer(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, handle, 0, layer);
          else gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, handle, 0);
          gl.readBuffer(gl.COLOR_ATTACHMENT0);
          if (gl.checkFramebufferStatus(gl.READ_FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error("The diagnostic framebuffer is incomplete");
          const part = bytes.subarray(layer * width * height * 4, (layer + 1) * width * height * 4);
          gl.readPixels(0, 0, width, height, gl.RGBA_INTEGER, gl.UNSIGNED_INT, part);
        }
        const error = gl.getError();
        if (error !== gl.NO_ERROR) throw new Error("The actual GPU texture read failed: " + error);
      } finally { gl.bindFramebuffer(gl.READ_FRAMEBUFFER, previousRead); gl.deleteFramebuffer(framebuffer); }
      return { width, height, depth, base64: new Uint8Array(bytes.buffer).toBase64() };
    };
    return { displayPacked: read(spark.display.target.textures[0], true), fullGpuOrder: read(spark.orderingTexture, false),
      maskedGpuOrder: spark.r145MaskedTexture ? read(spark.r145MaskedTexture, false) : null,
      fullCpuOrder: new Uint8Array(spark.r145LastOrdering.buffer).toBase64(),
      maskedCpuOrder: (() => { const a = spark.r145MaskedOrdering; return a ? new Uint8Array(a.buffer, a.byteOffset, a.byteLength).toBase64() : null; })(),
      population: spark.display.numSplats, originalActive: spark.activeSplats, active: spark.r145DrawActive,
      epoch: spark.display.r145Epoch ?? null, version: spark.display.version, mapping: spark.display.mappingVersion,
      selected: spark.r145MaskSelected ?? false, viewOrigin: spark.display.viewOrigin.toArray() };
  });
  for (const field of ["displayPacked", "fullGpuOrder", "maskedGpuOrder", "fullCpuOrder", "maskedCpuOrder"]) {
    const value = record[field]; if (value === null) continue;
    const bytes = Buffer.from(typeof value === "string" ? value : value.base64, "base64");
    const filename = path.join(output, name + "_" + field + ".u32"); fs.writeFileSync(filename, bytes, { flag: "wx" });
    record[field] = { ...(typeof value === "string" ? {} : { width: value.width, height: value.height, depth: value.depth }),
      path: filename, bytes: bytes.length, sha256: hash(bytes) };
  }
  return record;
}
async function payloadEpisode(browser, enabled) {
  const entry = await contextPage(browser, enabled), { page, context, errors } = entry;
  const records = [];
  try {
    await page.goto(entry.url, { waitUntil: "domcontentloaded" }); await settle(page);
    await page.getByLabel("移动方式").selectOption("fly"); await page.getByLabel("移动时优先流畅").uncheck(); await settle(page);
    for (const [range, index] of [["scene", 0], ["box", 132]]) {
      await page.getByLabel("展示范围").selectOption(range);
      await page.waitForFunction(r => window.box2Viewer.sceneMode === r && !window.box2Viewer.modelLoading, range); await settle(page);
      const captureRow = await step(page, range + "_" + index, () => page.evaluate(i => window.box2Viewer.setView(i), index));
      const gpu = await inspectGpu(page, (enabled ? "candidate" : "native") + "_" + range);
      let fullOnly = null;
      if (enabled) {
        fullOnly = await step(page, range + "_full", () => page.evaluate(() => {
          const s = window.box2Viewer; s.r138.spark.r145MaskStamp = null; s.r138.forceFrame();
        }));
      }
      records.push({ range, index, capture: captureRow, gpu, fullOnly });
    }
    check(errors.length === 0, "actual_gpu_payload_browser_errors"); return { enabled, records, errors };
  } finally {
    try { await page.evaluate(() => { const s = window.box2Viewer; s.r138.dispose(); s.r138.spark.dispose(); }); }
    finally { await context.close(); }
  }
}
function compareBytes(left, right, length) {
  const a = fs.readFileSync(left.path).subarray(0, length), b = fs.readFileSync(right.path).subarray(0, length);
  if (a.length !== length || b.length !== length) throw new Error("The GPU readback denominator is incomplete");
  let changed = 0; const first = [];
  for (let i = 0; i < length; i++) if (a[i] !== b[i]) { changed++; if (first.length < 12) first.push({ byte: i, native: a[i], candidate: b[i] }); }
  return { exact: changed === 0, bytes: length, changed, first };
}
(async () => {
  const started = performance.now(), gpuBefore = gpuState(), browser = await chromium.launch({ headless: true, args: ["--use-angle=d3d11"] });
  let result;
  try {
    const native = await payloadEpisode(browser, false), candidate = await payloadEpisode(browser, true);
    const rows = native.records.map((a, i) => {
      const b = candidate.records[i];
      return { range: a.range, index: a.index,
        payload: compareBytes(a.gpu.displayPacked, b.gpu.displayPacked, a.gpu.population * 16),
        nativeUpload: compareBytes(a.gpu.fullCpuOrder, a.gpu.fullGpuOrder, a.gpu.originalActive * 4),
        candidateUpload: compareBytes(b.gpu.fullCpuOrder, b.gpu.fullGpuOrder, b.gpu.originalActive * 4),
        maskedUpload: compareBytes(b.gpu.maskedCpuOrder, b.gpu.maskedGpuOrder, b.gpu.active * 4),
        selectedPixelsExact: a.capture.rgba_sha256 === b.capture.rgba_sha256,
        candidateFullPixelsExact: a.capture.rgba_sha256 === b.fullOnly.rgba_sha256,
        candidateFullVersusMaskedExact: b.capture.rgba_sha256 === b.fullOnly.rgba_sha256 };
    });
    check([...sources, ...inputIdentities].every(row => identity(row.path).sha256 === row.sha256), "actual_gpu_payload_original_sources_unchanged");
    result = { status: "actual_gpu_payload_diagnostic_only", native, candidate, rows, sources, inputIdentities, derivation, checks,
      gpuBefore, gpuAfter: gpuState(), newOptimizationCount: 0, noUserTabsOrListeningService: true };
  } catch (error) {
    result = { status: "failed_verification", error: String(error), stack: error.stack, sources, inputIdentities, derivation, checks, newOptimizationCount: 0 }; throw error;
  } finally { await browser.close(); result.elapsedSeconds = (performance.now() - started) / 1000; publish(result);
    console.log(JSON.stringify({ status: result.status, rows: result.rows, elapsedSeconds: result.elapsedSeconds })); }
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
`;
const injection = `
source = once(source, 'const sources = sourceFiles.map(identity), inputIdentities = inputs.map(identity);',
  'sourceFiles.push(' + ${JSON.stringify(JSON.stringify(driver))} + ');\\nconst sources = sourceFiles.map(identity), inputIdentities = inputs.map(identity);');
const payloadMainStart = source.indexOf("(async () => {\\n  const started = performance.now(), gpuBefore = gpuState();");
if (payloadMainStart < 0) throw new Error("The actual private browser entry changed");
source = source.slice(0, payloadMainStart) + ${JSON.stringify(main)};
derived._compile(source, __filename);
`;
const program = once(fs.readFileSync(driver, "utf8"), "derived._compile(source, __filename);", injection);
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(program, __filename);
