// Run the immutable native viewer through private browser routes without a listening service.
const fs = require("fs"), path = require("path"), crypto = require("crypto");
const { chromium } = require("C:/Users/99303/git/playwright/node_modules/playwright");
const root = path.resolve(__dirname, "../..");
const release = path.join(root, "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release");
const output = path.resolve(process.argv[3] ?? "");
const mode = process.argv[2];
const helper = path.join(__dirname, "behind_camera_draw.js");
const caseRoot = path.join(release, "baseline/cases/packed");
const prefix = "http://127.0.0.1:28218";
if (!["generic", "screen", "formal", "recover"].includes(mode)
    || !output.startsWith(path.join(root, "04_outputs/gaussian_pipeline_optimization_v3_20261005/behind_camera_draw_001") + path.sep)
    || fs.existsSync(output)) throw new Error("A fresh private operation path is required");
fs.mkdirSync(output, { recursive: true });
fs.mkdirSync(path.join(output, "sources"));
const hash = bytes => crypto.createHash("sha256").update(bytes).digest("hex");
const identity = filename => { const bytes = fs.readFileSync(filename); return { path: filename, bytes: bytes.length, sha256: hash(bytes) }; };
const publish = value => fs.writeFileSync(path.join(output, "RESULT.json"), JSON.stringify(value, null, 2) + "\n", { flag: "wx" });
const replaceOnce = (s, before, after) => {
  if (s.split(before).length !== 2) throw new Error("The pinned viewer source anchor changed");
  return s.replace(before, after);
};
const sourceFiles = [__filename, helper, path.join(caseRoot, "viewer.js")];
const inputs = [path.join(release, "baseline/ASSET_MANIFEST.json"), ...fs.readdirSync(caseRoot)
  .filter(name => /\.(js|json|spk|html)$/.test(name) && !name.startsWith("export"))
  .map(name => path.join(caseRoot, name)), ...fs.readdirSync(path.join(release, "shared-build/vendor"))
  .filter(name => /\.(js|txt)$/.test(name)).map(name => path.join(release, "shared-build/vendor", name))];
const sources = sourceFiles.map(identity), inputIdentities = inputs.map(identity);
for (const filename of sourceFiles) fs.copyFileSync(filename, path.join(output, "sources", path.basename(filename)));
const helperSource = fs.readFileSync(helper, "utf8"), viewerSource = fs.readFileSync(path.join(caseRoot, "viewer.js"), "utf8");
const checks = [], episodes = [], failures = [];
function check(value, name) { checks.push({ name, passed: Boolean(value) }); if (!value) throw new Error(name); }
function gpuState() {
  const { execFileSync } = require("child_process");
  const line = execFileSync("nvidia-smi", ["--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"], { encoding: "utf8" }).trim();
  const [total, used, free] = line.split(",").map(Number);
  if (![total, used, free].every(Number.isFinite) || used > 12288 || free < 2048) throw new Error("The original GPU resource budget is unavailable");
  return { totalMiB: total, usedMiB: used, freeMiB: free, scope: "whole_device_not_owned_browser_peak" };
}
function viewer(enabled) {
  let code = 'import { createBehindCameraCuller, installDrawCounter } from "./r138.js";\n' + viewerSource;
  code = replaceOnce(code, "  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));", "  const drawCounter = installDrawCounter(renderer.getContext());\n  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));");
  code = replaceOnce(code, "  scene.add(new SparkRenderer({ renderer, onDirty: requestRender, ...filterOptions }));",
    "  const spark = new SparkRenderer({ renderer, onDirty: requestRender, ...filterOptions });\n  scene.add(spark);\n  const culler = createBehindCameraCuller({ scene, spark, SplatMesh, enabled: " + enabled + " });");
  code = replaceOnce(code, "  const modelLoadStarted = performance.now();", `  state.r138 = { scene, camera, spark, culler, drawCounter, preset,
    models: () => { const found = []; scene.traverse(node => { if (node instanceof SplatMesh) found.push(node); }); return found; },
    freeCamera(position, direction) {
      setMode("fly"); camera.position.fromArray(position); camera.lookAt(camera.position.clone().add(new THREE.Vector3(...direction)));
      syncFlightAngles(); projection.free(); requestRender();
    }, forceFrame: requestRender,
    dispose() { culler.dispose(); drawCounter.dispose(); }
  };
  const modelLoadStarted = performance.now();`);
  code = replaceOnce(code, "    renderer.render(scene, camera); state.frames++;", "    culler.beforeFrame(camera);\n    renderer.render(scene, camera); state.frames++;");
  return code;
}
async function contextPage(browser, enabled) {
  const context = await browser.newContext({ viewport: { width: 960, height: 600 }, deviceScaleFactor: 1 });
  const page = await context.newPage(), errors = [];
  page.on("pageerror", error => errors.push(String(error)));
  await context.route(prefix + "/**", async route => {
    const requested = new URL(route.request().url()).pathname;
    const relative = decodeURIComponent(requested.replace(/^\/(native|candidate)\//, ""));
    const filename = path.resolve(release, relative);
    if (!filename.startsWith(release + path.sep)) throw new Error("A diagnostic request escaped the pinned release");
    let bytes;
    if (relative.endsWith("/r138.js")) bytes = helperSource;
    else if (relative.endsWith("/viewer.js")) bytes = viewer(enabled);
    else bytes = fs.readFileSync(filename);
    const extension = path.extname(filename);
    await route.fulfill({ status: 200, body: bytes,
      contentType: extension === ".js" ? "text/javascript" : extension === ".json" ? "application/json" : extension === ".html" ? "text/html" : "application/octet-stream" });
  });
  return { context, page, errors, url: prefix + "/" + (enabled ? "candidate" : "native") + "/baseline/cases/packed/index.html" };
}
async function settle(page, previous = -1) {
  await page.waitForFunction(prior => {
    const s = window.box2Viewer;
    return s?.ready && !s.modelLoading && s.frames > prior && !s.isRenderPending() && !s.r138.spark.sorting
      && s.r138.spark.sortTimeoutId === undefined && !s.r138.spark.sortDirty;
  }, previous, { timeout: 45000, polling: "raf" });
}
async function capture(page, name) {
  return page.evaluate(async name => {
    const s = window.box2Viewer, renderer = s.renderer, gl = renderer.getContext(), canvas = renderer.domElement;
    const bytes = new Uint8Array(canvas.width * canvas.height * 4);
    gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    return { name, width: canvas.width, height: canvas.height,
      rgba_sha256: [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, "0")).join(""),
      projection: s.projectionMatrix, position: s.getCameraPosition(), direction: s.getCameraDirection(),
      count: [s.fullGaussianCount, s.boxGaussianCount], mode: s.sceneMode,
      draw: structuredClone(s.r138.drawCounter.trace), culling: structuredClone(s.r138.culler.trace), frames: s.frames };
  }, name);
}
async function step(page, name, action) {
  const previous = await page.evaluate(() => window.box2Viewer.frames);
  await action(); await settle(page, previous);
  return capture(page, name);
}
async function episode(browser, enabled, trial, injected = false) {
  const started = performance.now(), entry = await contextPage(browser, enabled), { context, page, errors } = entry;
  const captures = [];
  let completed;
  try {
    await page.goto(entry.url, { waitUntil: "domcontentloaded" }); await settle(page);
    check(await page.evaluate(() => /RTX 5060 Ti/.test(window.box2Viewer.renderer.getContext().getParameter(window.box2Viewer.renderer.getContext().getExtension("WEBGL_debug_renderer_info").UNMASKED_RENDERER_WEBGL))), "actual_hardware_renderer_" + trial + "_" + enabled);
    await page.getByLabel("移动方式").selectOption("fly");
    await page.getByLabel("移动时优先流畅").uncheck();
    await settle(page);
    for (const range of ["scene", "box"]) {
      if (range !== "scene") await page.getByLabel("展示范围").selectOption(range);
      await page.waitForFunction(range => window.box2Viewer.sceneMode === range && !window.box2Viewer.modelLoading, range);
      await settle(page);
      const pathCount = await page.evaluate(() => Number(document.getElementById("path-slider").max) + 1);
      for (let i = 0; i < 24; i++) {
        const index = Math.floor(i * (pathCount - 1) / 23);
        captures.push(await step(page, range + "_capture_" + index, () => page.evaluate(index => window.box2Viewer.setView(index), index)));
      }
      for (const preset of ["top", "drone"]) captures.push(await step(page, range + "_" + preset, () => page.locator('button[data-preset="' + preset + '"]').click()));
      const bounds = await page.evaluate(() => {
        const s = window.box2Viewer, mesh = s.r138.models().find(model => model.visible);
        const box = mesh.getBoundingBox(true), center = box.getCenter(box.min.clone()), radius = box.min.distanceTo(box.max) / 2;
        return { center: center.toArray(), radius, near: s.r138.camera.near };
      });
      for (let i = 0; i < 12; i++) {
        const z = 1 - 2 * (i + 0.5) / 12, angle = i * Math.PI * (3 - Math.sqrt(5));
        const xy = Math.sqrt(1 - z * z), direction = [xy * Math.cos(angle), xy * Math.sin(angle), z];
        const distance = 4 * bounds.radius + 2 * bounds.near;
        const position = bounds.center.map((value, axis) => value + direction[axis] * distance);
        for (const [label, sign] of [["front", -1], ["away", 1], ["return", -1]]) {
          captures.push(await step(page, range + "_free_" + i + "_" + label,
            () => page.evaluate(({ position, direction }) => window.box2Viewer.r138.freeCamera(position, direction), { position, direction: direction.map(value => value * sign) })));
          if (injected && i === 0 && label === "away") throw new Error("registered_after_actual_back_view_injection");
        }
      }
    }
    const result = { enabled, trial, captures, errors, elapsedSeconds: (performance.now() - started) / 1000,
      final: await capture(page, "final") };
    check(errors.length === 0, "native_runtime_errors_" + trial + "_" + enabled);
    completed = result;
    return result;
  } catch (error) {
    const state = await page.evaluate(() => {
      const s = window.box2Viewer;
      return s ? { frames: s.frames, ready: s.ready, modelLoading: s.modelLoading, mode: s.sceneMode,
        scheduled: s.isRenderPending(), sorting: s.r138?.spark.sorting, sortDirty: s.r138?.spark.sortDirty,
        timeoutSet: s.r138?.spark.sortTimeoutId !== undefined, errors: s.errors, culling: s.r138?.culler.trace } : null;
    });
    fs.writeFileSync(path.join(output, "EPISODE_FAILURE_" + trial + "_" + enabled + ".json"), JSON.stringify({ error: String(error), captures, errors, state }, null, 2) + "\n", { flag: "wx" });
    throw error;
  } finally {
    let cleanupError;
    try { await page.evaluate(() => window.box2Viewer?.r138?.dispose()); }
    catch (error) { cleanupError = error; failures.push({ scope: "cleanup_evaluation", error: String(error) }); }
    finally { await context.close(); }
    if (completed) completed.elapsedSeconds = (performance.now() - started) / 1000;
    if (cleanupError) throw cleanupError;
  }
}
async function generic(browser) {
  const entry = await contextPage(browser, false), { context, page, errors } = entry;
  try {
    await page.goto(entry.url); await settle(page);
    await page.getByLabel("展示范围").selectOption("box"); await settle(page);
    const result = await page.evaluate(async () => {
      const { packedCenterBounds, behindHalfSpace, createBehindCameraCuller } = await import("./r138.js");
      const THREE = await import("three"), { PackedSplats, SplatMesh } = await import("@sparkjsdev/spark");
      const s = window.box2Viewer, checks = [], actual = [];
      window.r138GenericProgress = { checks, actual };
      const add = (name, passed) => { checks.push({ name, passed: Boolean(passed) }); if (!passed) throw new Error(name + " " + JSON.stringify(window.r138GenericProgress.last ?? null)); };
      for (const model of s.r138.models()) {
        const candidate = packedCenterBounds(model.splats), native = model.getBoundingBox(true);
        add("actual_full_bounds_" + model.numSplats, candidate.supported && candidate.minimum.every((v, i) => Object.is(v, native.min.toArray()[i])) && candidate.maximum.every((v, i) => Object.is(v, native.max.toArray()[i])));
        actual.push({ count: model.numSplats, candidate, native: { minimum: native.min.toArray(), maximum: native.max.toArray() } });
      }
      for (const scale of [0.01, 1, 100]) for (const count of [17, 31, 71]) {
        // The pinned native format rounds capacity down to complete 2048-splat rows.
        const capacity = Math.ceil(count / 2048) * 2048, array = new Uint32Array(capacity * 4);
        for (let i = 0; i < count; i++) {
          const x = scale * Math.sin(i * 0.7), y = scale * (i / count - 0.5), z = scale * Math.cos(i * 1.3);
          array[4 * i] = 0xff0000ff;
          array[4 * i + 1] = THREE.DataUtils.toHalfFloat(x) | (THREE.DataUtils.toHalfFloat(y) << 16);
          array[4 * i + 2] = THREE.DataUtils.toHalfFloat(z); array[4 * i + 3] = 0x7f7f7f;
        }
        const packed = new PackedSplats({ packedArray: array, numSplats: count, maxSh: 3,
          extra: { sh1: new Uint32Array(capacity * 2), sh2: new Uint32Array(capacity * 4), sh3: new Uint32Array(capacity * 4) }, splatEncoding: { lodOpacity: false }, lod: false });
        const model = new SplatMesh({ splats: packed, lod: false }); await model.initialized;
        const candidate = packedCenterBounds(packed), native = model.getBoundingBox(true);
        window.r138GenericProgress.last = { scale, count, candidate, native: { minimum: native.min.toArray(), maximum: native.max.toArray() }, packedCount: packed.numSplats, packedLod: packed.lod, encoding: packed.splatEncoding };
        add("native_keeps_all_fixture_centers_" + scale + "_" + count, packed.numSplats === count && model.numSplats === count);
        add("heterogeneous_native_bounds_" + scale + "_" + count, candidate.supported && candidate.minimum.every((v, i) => Object.is(v, native.min.toArray()[i])) && candidate.maximum.every((v, i) => Object.is(v, native.max.toArray()[i])));
        add("behind_" + scale + "_" + count, behindHalfSpace(candidate, [scale * 10, 0, 0], [1, 0, 0]).rejected);
        add("front_" + scale + "_" + count, !behindHalfSpace(candidate, [scale * 10, 0, 0], [-1, 0, 0]).rejected);
        model.dispose();
      }
      const point = { supported: true, minimum: [0, 0, 0], maximum: [0, 0, 0] };
      const shortPacked = new PackedSplats({ packedArray: new Uint32Array(17 * 4), numSplats: 17, splatEncoding: { lodOpacity: false }, lod: false });
      add("original_short_fixture_native_capacity_zero_reproduced", shortPacked.numSplats === 0 && !packedCenterBounds(shortPacked).supported);
      shortPacked.dispose();
      const halfArray = new Uint32Array(65536 * 4);
      for (let i = 0; i < 65536; i++) { halfArray[i * 4 + 1] = i | (i << 16); halfArray[i * 4 + 2] = i; }
      const allHalf = new PackedSplats({ packedArray: halfArray, numSplats: 65536, splatEncoding: { lodOpacity: false }, lod: false });
      let allHalfExact = true, halfCount = 0, nonfiniteCount = 0;
      allHalf.forEachSplat((index, center) => {
        const value = packedCenterBounds({ packedArray: halfArray.subarray(index * 4, index * 4 + 4), numSplats: 1, lod: false, splatEncoding: { lodOpacity: false } });
        halfCount++;
        if (![center.x, center.y, center.z].every(Number.isFinite)) { nonfiniteCount++; allHalfExact &&= !value.supported; }
        else allHalfExact &&= value.supported && value.minimum.every((v, i) => Object.is(v, center.toArray()[i])) && value.maximum.every((v, i) => Object.is(v, center.toArray()[i]));
      });
      add("all65536_native_half_bits_exact_or_nonfinite_native_route", allHalfExact && halfCount === 65536 && nonfiniteCount === 2048);
      window.r138GenericProgress.half = { halfCount, nonfiniteCount, allHalfExact }; allHalf.dispose();
      add("near_boundary_abstain", !behindHalfSpace(point, [1e-8, 0, 0], [1, 0, 0]).rejected);
      add("unsupported_nonunit_camera", !behindHalfSpace(point, [100, 0, 0], [2, 0, 0]).rejected);
      add("unsupported_half_range", !behindHalfSpace(point, [1e6, 0, 0], [1, 0, 0]).rejected);
      add("unsupported_nonfinite", !behindHalfSpace(point, [Infinity, 0, 0], [1, 0, 0]).rejected);
      const model = s.r138.models().find(value => value.visible), bounds = packedCenterBounds(model.splats);
      const oldCamera = s.r138.camera.clone(), culler = createBehindCameraCuller({ scene: s.r138.scene, spark: s.r138.spark, SplatMesh });
      const box = model.getBoundingBox(true), radius = box.min.distanceTo(box.max) / 2, center = box.getCenter(box.min.clone());
      s.r138.camera.position.copy(center).add(new THREE.Vector3(4 * radius + 1, 0, 0));
      s.r138.camera.lookAt(s.r138.camera.position.clone().add(new THREE.Vector3(1, 0, 0)));
      culler.beforeFrame(s.r138.camera); add("actual_behind_hidden", !s.r138.spark.visible);
      model.position.x = 1; culler.beforeFrame(s.r138.camera); add("transform_routes_native", s.r138.spark.visible); model.position.x = 0;
      model.objectModifiers = [{}]; culler.beforeFrame(s.r138.camera); add("modifier_routes_native", s.r138.spark.visible); model.objectModifiers = undefined;
      model.splats.needsUpdate = true; culler.beforeFrame(s.r138.camera); add("update_routes_native", s.r138.spark.visible); model.splats.needsUpdate = false;
      culler.beforeFrame(s.r138.camera); add("rebuild_after_update", !s.r138.spark.visible && culler.trace.builds >= 2);
      const oldNear = s.r138.camera.near; s.r138.camera.near = -1; culler.beforeFrame(s.r138.camera); add("unknown_projection_routes_native", s.r138.spark.visible); s.r138.camera.near = oldNear;
      culler.beforeFrame(s.r138.camera); culler.dispose(); add("dispose_restores_native", s.r138.spark.visible);
      s.r138.camera.position.copy(oldCamera.position); s.r138.camera.quaternion.copy(oldCamera.quaternion);
      return { checks, actual, trace: culler.trace, scope: "native_bounds_and_route_component_not_cross_object_rendering" };
    });
    const direct = await page.evaluate(() => {
      const s = window.box2Viewer, before = { frames: s.frames, scheduled: s.isRenderPending() };
      s.r138.preset("top"); return { before, after: { frames: s.frames, scheduled: s.isRenderPending() } };
    });
    check(!direct.before.scheduled && !direct.after.scheduled, "original_private_preset_does_not_request_frame_reproduced");
    const beforeClick = await page.evaluate(() => window.box2Viewer.frames);
    await page.locator('button[data-preset="top"]').click(); await settle(page, beforeClick);
    result.nativePresetRegression = { direct, afterActualButtonFrames: await page.evaluate(() => window.box2Viewer.frames) };
    check(result.nativePresetRegression.afterActualButtonFrames > beforeClick, "actual_original_preset_button_renders_regression_passed");
    check(errors.length === 0, "generic_native_runtime_no_errors"); return result;
  } catch (error) {
    const progress = await page.evaluate(() => window.r138GenericProgress ?? null);
    fs.writeFileSync(path.join(output, "GENERIC_FAILURE_DETAIL.json"), JSON.stringify({ error: String(error), progress }, null, 2) + "\n", { flag: "wx" });
    throw error;
  } finally { await context.close(); }
}
function compare(a, b) {
  check(a.captures.length === 124 && b.captures.length === 124, "complete124_frames_no_removed_samples");
  for (let i = 0; i < a.captures.length; i++) {
    const left = a.captures[i], right = b.captures[i];
    for (const key of ["name", "width", "height", "rgba_sha256", "projection", "position", "direction", "count", "mode"])
      check(JSON.stringify(left[key]) === JSON.stringify(right[key]), "pair_" + a.trial + "_" + i + "_" + key);
  }
  const reduction = 1 - b.final.draw.triangles / a.final.draw.triangles;
  return { trial: a.trial, nativeTriangles: a.final.draw.triangles, candidateTriangles: b.final.draw.triangles,
    reduction, nativeSeconds: a.elapsedSeconds, candidateSeconds: b.elapsedSeconds,
    rawWholeNotIncreased: b.elapsedSeconds <= a.elapsedSeconds, candidateTrace: b.final.culling };
}
(async () => {
  const started = performance.now(), gpuBefore = gpuState();
  const browser = await chromium.launch({ headless: true, args: ["--use-angle=d3d11"] });
  let result;
  try {
    if (mode === "generic") result = { status: "generic_passed", generic: await generic(browser) };
    else if (mode === "recover") {
      let caught = false;
      try { await episode(browser, true, "injected", true); } catch (error) { caught = String(error).includes("registered_after_actual_back_view_injection"); failures.push({ scope: "registered_injection", error: String(error) }); }
      check(caught, "actual_after_back_view_failure_propagated_and_context_closed");
      const a = await episode(browser, false, "recovery"), b = await episode(browser, true, "recovery");
      episodes.push(a, b); result = { status: "actual_new_context_recovery_passed", pairs: [compare(a, b)] };
    } else {
      const pairs = [];
      for (let trial = 0; trial < (mode === "formal" ? 3 : 1); trial++) {
        const order = trial % 2 ? [true, false] : [false, true], rows = new Map();
        for (const enabled of order) { gpuState(); const row = await episode(browser, enabled, trial); rows.set(enabled, row); episodes.push(row); }
        const pair = compare(rows.get(false), rows.get(true)); pairs.push(pair);
        if (pair.reduction < 0.2 || !pair.rawWholeNotIncreased || pair.candidateSeconds > 1.1 * pair.nativeSeconds) break;
      }
      result = { status: pairs.length === (mode === "formal" ? 3 : 1) && pairs.every(row => row.reduction >= 0.2 && row.rawWholeNotIncreased)
        ? "measured_protection_passed_acceptance_pending" : "screen_negative_no_promotion", pairs };
    }
    check([...sources, ...inputIdentities].every(row => identity(row.path).sha256 === row.sha256), "all_current_sources_and_complete_input_files_unchanged");
    Object.assign(result, { mode, sources, inputIdentities, checks, episodes, failures, gpuBefore, gpuAfter: gpuState(),
      browserVersion: browser.version(), elapsedSeconds: (performance.now() - started) / 1000,
      newOptimizationCount: 0, noUserTabsOrListeningService: true });
  } catch (error) {
    result = { status: "failed_verification", mode, error: String(error), stack: error.stack, sources, inputIdentities,
      checks, episodes, failures, elapsedSeconds: (performance.now() - started) / 1000, newOptimizationCount: 0 };
    publish(result); throw error;
  } finally { await browser.close(); }
  publish(result); console.log(JSON.stringify({ status: result.status, mode, checks: result.checks.length, pairs: result.pairs, elapsedSeconds: result.elapsedSeconds }));
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
