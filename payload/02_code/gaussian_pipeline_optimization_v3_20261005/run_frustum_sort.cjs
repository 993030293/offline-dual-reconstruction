// Run the fixed native preview with a privately derived depth-mask runtime.
const fs = require("fs"), path = require("path"), Module = require("module");
const { derive, instrument, digest, once } = require("./frustum_sort_build.cjs");
const originalPath = path.join(__dirname, "run_behind_camera_draw.cjs");
const rootPath = path.resolve(__dirname, "../..");
const releasePath = path.join(rootPath, "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release");
const vendorPath = path.join(releasePath, "shared-build/vendor/spark.module.js");
const originalVendor = fs.readFileSync(vendorPath, "utf8");
const nativeVendor = instrument(originalVendor), candidateVendor = instrument(derive(originalVendor));
const derivation = { original_sha256: digest(originalVendor), native_runtime_sha256: digest(nativeVendor),
  candidate_runtime_sha256: digest(candidateVendor), unchanged_original_SPKPK001_encoding: true };
let source = fs.readFileSync(originalPath, "utf8");
source = once(source, 'behind_camera_draw_001") + path.sep)', 'r145_001") + path.sep)');
source = once(source, 'const sourceFiles = [__filename, helper, path.join(caseRoot, "viewer.js")];',
  'const sourceFiles = [__filename, helper, path.join(caseRoot, "viewer.js"), ' + JSON.stringify(originalPath) + ', ' + JSON.stringify(path.join(__dirname, "frustum_sort_build.cjs")) + '];');
source = once(source, 'const helperSource = fs.readFileSync(helper, "utf8"), viewerSource = fs.readFileSync(path.join(caseRoot, "viewer.js"), "utf8");',
  'const helperSource = fs.readFileSync(helper, "utf8"), viewerSource = fs.readFileSync(path.join(caseRoot, "viewer.js"), "utf8");\n' +
  'const nativeVendor = ' + JSON.stringify(nativeVendor) + ';\nconst candidateVendor = ' + JSON.stringify(candidateVendor) + ';\nconst derivation = ' + JSON.stringify(derivation) + ';\n' +
  'fs.writeFileSync(path.join(output, "DERIVATION.json"), JSON.stringify(derivation, null, 2));\n' +
  'fs.writeFileSync(path.join(output, "native_runtime.js"), nativeVendor);\nfs.writeFileSync(path.join(output, "candidate_runtime.js"), candidateVendor);\n' +
  'fs.copyFileSync(path.join(release, "shared-build/vendor/SPARK-LICENSE.txt"), path.join(output, "SPARK-LICENSE.txt"));');
source = once(source, 'enabled: " + enabled + " });"', 'enabled: " + false + " });"');
source = once(source, '  return code;\n}\nasync function contextPage',
  '  code = replaceOnce(code, \'assets.scene.spark_sha256);\', JSON.stringify(enabled ? derivation.candidate_runtime_sha256 : derivation.native_runtime_sha256) + \');\');\n  return code;\n}\nasync function contextPage');
source = once(source, 'else if (relative.endsWith("/viewer.js")) bytes = viewer(enabled);',
  'else if (relative.endsWith("/viewer.js")) bytes = viewer(enabled);\n    else if (relative.endsWith("/spark.module.js")) bytes = enabled ? candidateVendor : nativeVendor;');
source = once(source, 'async function capture(page, name) {\n  return page.evaluate(async name => {',
  'let orderingCapture = 0;\nasync function capture(page, name) {\n  const captured = await page.evaluate(async name => {');
source = once(source, 'draw: structuredClone(s.r138.drawCounter.trace), culling: structuredClone(s.r138.culler.trace), frames: s.frames };',
  'draw: structuredClone(s.r138.drawCounter.trace), culling: structuredClone(s.r138.culler.trace), frames: s.frames,\n      active: s.r138.spark.r145DrawActive ?? s.r138.spark.activeSplats, originalActive: s.r138.spark.activeSplats, sorts: s.r138.spark.r145SortCount ?? 0, cameraKey: s.r138.spark.r145CameraKey ?? null,\n      maskSelected: s.r138.spark.r145MaskSelected ?? false, ext: s.r138.spark.display.extSplats, cov: s.r138.spark.display.covSplats,\n      fullOrderingBase64: new Uint8Array(s.r138.spark.r145LastOrdering?.buffer ?? new ArrayBuffer(0)).toBase64(),\n      orderingBase64: (() => { const array = s.r138.spark.r145DrawOrdering ?? new Uint32Array(0); return new Uint8Array(array.buffer, array.byteOffset, array.byteLength).toBase64(); })() };');
source = once(source, '  }, name);\n}\nasync function step',
  '  }, name);\n  const folder = path.join(output, "ordering"); fs.mkdirSync(folder, { recursive: true });\n  const index = orderingCapture++;\n  for (const field of ["ordering", "fullOrdering"]) {\n    const filename = path.join(folder, String(index) + "_" + field + ".u32");\n    const bytes = Buffer.from(captured[field + "Base64"], "base64");\n    fs.writeFileSync(filename, bytes, { flag: "wx" });\n    delete captured[field + "Base64"];\n    captured[field] = { path: filename, bytes: bytes.length, sha256: hash(bytes) };\n  }\n  return captured;\n}\nasync function step');
source = once(source, 'ext: s.r138.spark.display.extSplats, cov: s.r138.spark.display.covSplats,',
  'worldMatrix: s.r138.camera.matrixWorld.elements.slice(), quaternion: s.r138.camera.quaternion.toArray(),\n      ext: s.r138.spark.display.extSplats, cov: s.r138.spark.display.covSplats,');
const mainStart = source.indexOf("(async () => {\n  const started = performance.now(), gpuBefore = gpuState();");
if (mainStart < 0) throw new Error("The original browser campaign entry changed");
source = source.slice(0, mainStart) + `
function orderSubsequence(left, right) {
  const a = fs.readFileSync(left.ordering.path), b = fs.readFileSync(right.ordering.path);
  if (a.length !== left.active * 4 || b.length !== right.active * 4 || b.length > a.length) return false;
  let at = 0;
  for (let i = 0; i < b.length; i += 4) {
    const value = b.readUInt32LE(i);
    while (at < a.length && a.readUInt32LE(at) !== value) at += 4;
    if (at >= a.length) return false;
    at += 4;
  }
  return true;
}
function compareFrustum(a, b) {
  check(a.captures.length === b.captures.length, "all_current_position_denominators");
  const rows = a.captures.map((left, i) => {
    const right = b.captures[i];
    const exact = ["name", "width", "height", "rgba_sha256", "projection", "position", "direction", "count", "mode"].every(key => JSON.stringify(left[key]) === JSON.stringify(right[key]));
    const ordering = orderSubsequence(left, right);
    check(exact && ordering && left.originalActive === right.originalActive && left.fullOrdering.sha256 === right.fullOrdering.sha256, "original_pixels_camera_population_full_and_retained_order_" + i);
    return { name: left.name, exact, retained_order_subsequence: ordering, native_active: left.active, candidate_active: right.active };
  });
  const na = a.final.draw.triangles, nb = b.final.draw.triangles;
  return { rows, nativeTriangles: na, candidateTriangles: nb, reduction: 1 - nb / na,
    nativeSeconds: a.elapsedSeconds, candidateSeconds: b.elapsedSeconds,
    rawWholeNotIncreased: b.elapsedSeconds <= a.elapsedSeconds };
}
async function shortEpisode(browser, enabled) {
  const entry = await contextPage(browser, enabled), { page, context, errors } = entry;
  const captures = [];
  try {
    await page.goto(entry.url, { waitUntil: "domcontentloaded" }); await settle(page);
    await page.getByLabel("移动方式").selectOption("fly");
    await page.getByLabel("移动时优先流畅").uncheck(); await settle(page);
    captures.push(await capture(page, "actual_scene"));
    await page.getByLabel("展示范围").selectOption("box");
    await page.waitForFunction(() => window.box2Viewer.sceneMode === "box" && !window.box2Viewer.modelLoading); await settle(page);
    captures.push(await capture(page, "actual_box"));
    check(!enabled || captures.every(row => row.cameraKey !== null && row.ext === false && row.cov === false), "actual_current_renderer_packed_noncov_candidate_enabled");
    for (const scale of [.01, 1, 100]) for (const count of [17, 31, 71]) {
      const previous = await page.evaluate(() => window.box2Viewer.frames);
      await page.evaluate(async ({ scale, count }) => {
        const { PackedSplats, SplatMesh } = await import("@sparkjsdev/spark");
        const s = window.box2Viewer;
        for (const model of s.r138.models()) model.visible = false;
        if (s.r145Fixture) { s.r138.scene.remove(s.r145Fixture); s.r145Fixture.dispose(); }
        const capacity = Math.ceil(count / 2048) * 2048, array = new Uint32Array(capacity * 4);
        const { DataUtils } = await import("three");
        for (let i = 0; i < count; i++) {
          const x = scale * Math.sin(i * .7), y = scale * (i / count - .65), z = scale * Math.cos(i * 1.3);
          array[4 * i] = 0xff00ff00;
          array[4 * i + 1] = DataUtils.toHalfFloat(x) | (DataUtils.toHalfFloat(y) << 16);
          array[4 * i + 2] = DataUtils.toHalfFloat(z); array[4 * i + 3] = 0x7f7f7f;
        }
        const packed = new PackedSplats({ packedArray: array, numSplats: count, maxSh: 3,
          extra: { sh1: new Uint32Array(capacity * 2), sh2: new Uint32Array(capacity * 4), sh3: new Uint32Array(capacity * 4) }, lod: false,
          splatEncoding: { rgbMin: 0, rgbMax: 1, lnScaleMin: -12, lnScaleMax: 9, sh1Max: 1, sh2Max: 1, sh3Max: 1, lodOpacity: false } });
        await packed.initialized;
        const mesh = new SplatMesh({ splats: packed, lod: false }); await mesh.initialized;
        if (packed.numSplats !== count || mesh.numSplats !== count) throw new Error("The original native fixture discarded points");
        s.r145Fixture = mesh; s.r138.scene.add(mesh); s.r138.freeCamera([0, 0, 0], [0, -1, 0]);
      }, { scale, count });
      await settle(page, previous);
      captures.push(await capture(page, "generic_" + scale + "_" + count));
    }
    for (const name of ["near_far", "roll", "zoom", "unsupported_camera", "return_supported"]) {
      captures.push(await step(page, name, () => page.evaluate(name => {
        const s = window.box2Viewer, camera = s.r138.camera;
        if (name === "near_far") { camera.near = .5; camera.far = 20; }
        if (name === "roll") camera.rotateZ(.71);
        if (name === "zoom") { camera.zoom = 1.8; camera.updateProjectionMatrix(); }
        if (name === "unsupported_camera") camera.isPerspectiveCamera = false;
        if (name === "return_supported") camera.isPerspectiveCamera = true;
        camera.updateProjectionMatrix(); s.r138.forceFrame();
      }, name)));
    }
    check(errors.length === 0, "generic_original_runtime_errors");
    return { captures, errors, final: await capture(page, "generic_final") };
  } catch (error) {
    fs.writeFileSync(path.join(output, "GENERIC_PARTIAL_" + enabled + ".json"), JSON.stringify({ error: String(error), captures, errors }, null, 2), { flag: "wx" });
    throw error;
  } finally {
    try { await page.evaluate(() => { const s = window.box2Viewer; s.r145Fixture?.dispose(); s.r138.dispose(); }); }
    finally { await context.close(); }
  }
}
(async () => {
  const started = performance.now(), gpuBefore = gpuState();
  const browser = await chromium.launch({ headless: true, args: ["--use-angle=d3d11"] });
  let result;
  try {
    if (mode === "generic") {
      const a = await shortEpisode(browser, false), b = await shortEpisode(browser, true);
      const rows = a.captures.map((left, i) => {
        const right = b.captures[i], exact = ["name", "width", "height", "rgba_sha256", "projection", "position", "direction", "count", "mode"].every(key => JSON.stringify(left[key]) === JSON.stringify(right[key]));
        const ordering = orderSubsequence(left, right);
        check(exact && ordering && left.originalActive === right.originalActive && left.fullOrdering.sha256 === right.fullOrdering.sha256, "generic_original_pixels_full_and_retained_order_" + i);
        if (left.name === "unsupported_camera") check(left.ordering.sha256 === right.ordering.sha256, "unsupported_camera_original_complete_ordering_restored");
        return { name: left.name, exact, retained_order_subsequence: ordering, native_active: left.active, candidate_active: right.active };
      });
      result = { status: "generic_passed_screen_pending", rows, native: a, candidate: b };
    } else if (mode === "recover") {
      let caught = false;
      try { await episode(browser, true, "injected", true); }
      catch (error) { caught = String(error).includes("registered_after_actual_back_view_injection"); failures.push({ scope: "actual_registered_injection", error: String(error) }); }
      check(caught, "actual_consumer_failure_propagated_owned_context_closed");
      const a = await episode(browser, false, "recovery"), b = await episode(browser, true, "recovery");
      result = { status: "actual_new_context_complete_recovery_passed", pairs: [compareFrustum(a, b)] }; episodes.push(a, b);
    } else {
      const pairs = [];
      for (let trial = 0; trial < (mode === "formal" ? 3 : 1); trial++) {
        const order = trial % 2 ? [true, false] : [false, true], rows = new Map();
        for (const enabled of order) { gpuState(); const row = await episode(browser, enabled, trial); rows.set(enabled, row); episodes.push(row); }
        const pair = compareFrustum(rows.get(false), rows.get(true)); pairs.push(pair);
        if (pair.reduction < .2 || !pair.rawWholeNotIncreased || pair.candidateSeconds > 1.1 * pair.nativeSeconds) break;
      }
      result = { status: pairs.length === (mode === "formal" ? 3 : 1) && pairs.every(row => row.reduction >= .2 && row.rawWholeNotIncreased)
        ? "measured_protection_passed_acceptance_pending" : "screen_negative_no_promotion", pairs };
    }
    check([...sources, ...inputIdentities].every(row => identity(row.path).sha256 === row.sha256), "all_original_source_and_asset_identities_unchanged");
    Object.assign(result, { mode, sources, inputIdentities, derivation, checks, episodes, failures, gpuBefore, gpuAfter: gpuState(),
      browserVersion: browser.version(), newOptimizationCount: 0, noUserTabsOrListeningService: true });
  } catch (error) {
    result = { status: "failed_verification", mode, error: String(error), stack: error.stack, sources, inputIdentities,
      derivation, checks, episodes, failures, newOptimizationCount: 0 };
    throw error;
  } finally {
    await browser.close(); result.elapsedSeconds = (performance.now() - started) / 1000;
    publish(result);
    console.log(JSON.stringify({ status: result.status, mode, checks: result.checks.length, pairs: result.pairs?.map(({ rows, ...rest }) => rest), elapsedSeconds: result.elapsedSeconds }));
  }
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
`;
const derived = new Module(__filename, module);
source = source.replaceAll('"direction", "count", "mode"]', '"direction", "count", "mode", "worldMatrix", "quaternion"]');
derived.filename = __filename;
derived.paths = Module._nodeModulePaths(__dirname);
derived._compile(source, __filename);
