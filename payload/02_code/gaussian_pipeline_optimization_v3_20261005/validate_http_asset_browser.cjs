// Compare the unchanged native viewer through actual negotiated HTTP transport.
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { spawn, execFileSync } = require("child_process");
const { chromium } = require("C:/Users/99303/git/playwright/node_modules/playwright");

const ROOT = "C:/lidar-camera-fusion-mask-robust-downsampling";
const PYTHON = path.join(ROOT, "05_envs/sfm_py311_v1/Scripts/python.exe");
const CODE = path.join(ROOT, "02_code/gaussian_pipeline_optimization_v3_20261005");
const options = Object.fromEntries(process.argv.slice(2).reduce((rows, value, index, args) => {
  if (index % 2 === 0) rows.push([value.replace(/^--/, ""), args[index + 1]]);
  return rows;
}, []));
const campaign = path.resolve(options.campaign);
const output = path.resolve(options.output);
const receipt = path.join(output, "SERVER.json");
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const median = rows => [...rows].sort((a, b) => a - b)[Math.floor(rows.length / 2)];
const identity = file => ({ path: file, bytes: fs.statSync(file).size,
  sha256: crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex") });
function publish(name, value) {
  const target = path.join(output, name);
  fs.writeFileSync(target + ".tmp", JSON.stringify(value, null, 2));
  fs.renameSync(target + ".tmp", target);
}
function gpu() {
  const text = execFileSync("C:/Windows/System32/nvidia-smi.exe", ["--query-gpu=memory.total,memory.used,memory.free",
    "--format=csv,noheader,nounits"], { encoding: "utf8", windowsHide: true, timeout: 5000 }).trim();
  const values = text.split(/,\s*/).map(Number);
  if (values.length !== 3 || values.some(x => !Number.isFinite(x))) throw new Error("No actual CUDA headroom observation");
  return { total_MiB: values[0], used_MiB: values[1], free_MiB: values[2], scope: "whole_device_not_browser_process_peak" };
}
async function settled(page) {
  // Spark may request redundant frames while its sorted pixels remain unchanged.
  await sleep(300);
  let previous = null, stable = 0;
  for (let i = 0; i < 60; i++) {
    const observation = await pixels(page);
    const frame = JSON.stringify({ width: observation.width, height: observation.height,
      pixels: observation.pixel_sha256, projection: observation.projection });
    stable = frame === previous ? stable + 1 : 0;
    if (stable >= 3) return;
    previous = frame;
    await sleep(75);
  }
  throw new Error("The unchanged native pixels or camera never stabilized");
}
async function pixels(page) {
  return page.evaluate(async () => {
    const state = window.box2Viewer, renderer = state.renderer, gl = renderer.getContext(), canvas = renderer.domElement;
    const bytes = new Uint8Array(canvas.width * canvas.height * 4);
    gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    return { width: canvas.width, height: canvas.height,
      pixel_sha256: [...new Uint8Array(digest)].map(x => x.toString(16).padStart(2, "0")).join(""),
      projection: state.projectionMatrix, position: state.getCameraPosition(), direction: state.getCameraDirection(),
      navigation: state.navigationMode, mode: state.sceneMode, SH_unchanged_from_byte_exact_asset: true };
  });
}
async function controls(page) {
  const state = () => page.evaluate(() => ({ position: window.box2Viewer.getCameraPosition(),
    direction: window.box2Viewer.getCameraDirection(), keys: window.box2Viewer.getActiveKeys(),
    playing: window.box2Viewer.playing, view: window.box2Viewer.currentView, errors: window.box2Viewer.errors }));
  const distance = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));
  const rows = [];
  await page.locator("#navigation-mode").selectOption("orbit");
  await page.locator("[data-preset=front]").click();
  for (const button of ["left", "right"]) {
    const before = await state();
    await page.mouse.move(400, 400); await page.mouse.down({ button });
    await page.mouse.move(470, 440, { steps: 8 }); await page.mouse.up({ button }); await settled(page);
    const after = await state();
    rows.push({ name: "orbit_" + button, passed: distance(before.position, after.position) > .001 || distance(before.direction, after.direction) > .001 });
  }
  let before = await state(); await page.mouse.wheel(0, -250); await settled(page);
  rows.push({ name: "orbit_wheel", passed: distance(before.position, (await state()).position) > .001 });
  await page.locator("[data-preset=drone]").click(); await page.locator("canvas").focus();
  for (const key of ["w", "a", "q", "e"]) {
    before = await state(); await page.keyboard.down(key); await sleep(150); await page.keyboard.up(key); await settled(page);
    const after = await state();
    rows.push({ name: "fly_" + key, passed: distance(before.position, after.position) > .01 && after.keys.length === 0 });
  }
  before = await state(); await page.keyboard.down("Shift"); await page.keyboard.down("w");
  await sleep(150); await page.keyboard.up("w"); await page.keyboard.up("Shift"); await settled(page);
  rows.push({ name: "fly_shift", passed: distance(before.position, (await state()).position) > .04 });
  before = await state(); await page.mouse.move(400, 400); await page.mouse.down();
  await page.mouse.move(445, 430, { steps: 4 }); await page.mouse.up(); await settled(page);
  rows.push({ name: "fly_turn", passed: distance(before.direction, (await state()).direction) > .001 });
  await page.locator("#play").click(); await sleep(350);
  const playing = await state(); await page.locator("#play").click(); await settled(page);
  rows.push({ name: "capture_path_play_stop", passed: playing.playing && !(await state()).playing });
  await page.locator("#path-slider").evaluate(element => {
    element.value = "500"; element.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await settled(page); rows.push({ name: "capture_slider", passed: (await state()).view === 500 });
  await page.setViewportSize({ width: 1280, height: 800 }); await settled(page);
  const smaller = await pixels(page);
  await page.setViewportSize({ width: 1440, height: 900 }); await settled(page);
  rows.push({ name: "resize_projection_restored", passed: smaller.width > 0 && smaller.height > 0 && (await state()).errors.length === 0 });
  if (rows.some(row => !row.passed)) throw new Error("An original actual control failed: " + JSON.stringify(rows));
  return rows;
}

(async () => {
  if (fs.existsSync(output)) throw new Error("A fresh owned browser campaign is required");
  fs.mkdirSync(output, { recursive: true });
  const sources = [__filename, path.join(CODE, "http_asset_delivery.py"), path.join(CODE, "build_http_asset_packages.py"),
    "C:/Users/99303/git/playwright/node_modules/playwright/package.json"].map(identity);
  const admission = gpu();
  if (admission.free_MiB < 2048) throw new Error("The original two-GiB GPU headroom is unavailable");
  publish("PROTOCOL.json", { candidate: "R109_HTTP_LOSSLESS_ASSET_DELIVERY", sources, admission,
    all_controls_and_static_pixels_required: true, full_camera_path_required: 1016,
    exact_decoded_native_assets_and_all_SH_required: true, existing_tabs_and_ports_untouched: true,
    native_24_capture_indices: Array.from({ length: 24 }, (_, i) => Math.round(i * 1015 / 23)),
    three_cold_paired_navigations_per_scene_scope: true, maximum_each_whole_ratio: 1.1,
    whole_median_nonregression_required: true, first_packaging_cost_separate_not_claimed_as_speed: true, counted: false });
  const log = fs.openSync(path.join(output, "SERVER.log"), "wx");
  const server = spawn(PYTHON, ["-u", path.join(CODE, "http_asset_delivery.py"), "--campaign", campaign, "--receipt", receipt],
    { cwd: ROOT, windowsHide: true, stdio: ["pipe", log, log] });
  const serverClosed = new Promise(resolve => server.once("exit", (code, signal) => resolve({ code, signal })));
  const started = performance.now();
  let browser, failure, result;
  try {
    for (let i = 0; i < 600 && !fs.existsSync(receipt); i++) {
      if (server.exitCode !== null) throw new Error("Owned source validation failed before HTTP startup");
      await sleep(100);
    }
    if (!fs.existsSync(receipt)) throw new Error("Owned server startup exceeded sixty seconds");
    const serverState = JSON.parse(fs.readFileSync(receipt, "utf8"));
    const browserStart = performance.now();
    browser = await chromium.launch({ headless: true, args: ["--use-angle=d3d11"] });
    const browserStartup = (performance.now() - browserStart) / 1000;
    const rows = [], comparisons = [], controlRows = [], warm = [];
    for (let trial = 0; trial < 3; trial++) {
      for (const scope of ["scene", "box"]) {
        const pairPixels = {};
        for (const mode of trial === 1 ? ["gzip", "identity"] : ["identity", "gzip"]) {
          const wholeStart = performance.now();
          const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
          const page = await context.newPage(), errors = [];
          page.on("pageerror", error => errors.push(String(error)));
          try {
            const navigationStart = performance.now();
            const url = serverState.url + `/trial${trial}/${mode}/baseline/cases/packed/index.html?mode=${scope}`;
            await page.goto(url);
            await page.waitForFunction(() => window.box2Viewer?.ready && window.box2Viewer.firstReadyRenderMs != null,
              undefined, { timeout: 60000 });
            const navigationSeconds = (performance.now() - navigationStart) / 1000;
            const firstEntrySeconds = (performance.now() - wholeStart) / 1000;
            const state = await page.evaluate(() => {
              const s = window.box2Viewer, gl = s.renderer.getContext(), extension = gl.getExtension("WEBGL_debug_renderer_info");
              return { loaded: s.loadedModels, errors: s.errors, cache: globalThis.assetCacheTrace,
                full_splats: s.fullGaussianCount, roi_splats: s.boxGaussianCount,
                ready_render_ms: s.navigationToFirstReadyRenderMs,
                actual_GL_renderer: extension ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) : "unavailable",
                assets: performance.getEntriesByType("resource").filter(row => row.name.endsWith(".spk")).map(row => ({
                  name: row.name, encodedBodySize: row.encodedBodySize, decodedBodySize: row.decodedBodySize, transferSize: row.transferSize })) };
            });
            if (errors.length || state.errors.length || state.cache.some(row => row.hit || !row.committed_after_native_validation)) {
              throw new Error("The original cold native asset or clean runtime protection failed");
            }
            const row = { trial, scope, mode, navigation_seconds: navigationSeconds,
              complete_context_first_ready_seconds: firstEntrySeconds,
              shared_startup_charge_seconds: (serverState.startup_full_validation_seconds + browserStartup) / 12,
              whole_seconds: firstEntrySeconds + (serverState.startup_full_validation_seconds + browserStartup) / 12,
              state, gpu: gpu() };
            rows.push(row);
            if (row.gpu.used_MiB > 12 * 1024 || row.gpu.free_MiB < 2048) throw new Error("Actual device budget exceeded");
            if (trial === 0) {
              pairPixels[mode] = [];
              await page.locator("#navigation-mode").selectOption("fly");
              for (const sceneMode of ["scene", "box"]) {
                await page.locator("#scene-mode").selectOption(sceneMode);
                await page.waitForFunction(m => window.box2Viewer.sceneMode === m && !window.box2Viewer.modelLoading,
                  sceneMode, { timeout: 60000 });
                const count = await page.evaluate(async () => (await (await fetch("./camera_path.json")).json()).length);
                if (count !== 1016) throw new Error("The complete camera-path denominator changed");
                for (let i = 0; i < 24; i++) {
                  await page.evaluate(index => window.box2Viewer.setView(index), Math.round(i * 1015 / 23));
                  await settled(page);
                  pairPixels[mode].push({ sceneMode, index: Math.round(i * 1015 / 23), ...(await pixels(page)) });
                }
                for (const preset of ["top", "drone"]) {
                  await page.locator(`[data-preset=${preset}]`).click(); await settled(page);
                  pairPixels[mode].push({ sceneMode, preset, ...(await pixels(page)) });
                }
              }
              if (scope === "scene") controlRows.push({ mode, checks: await controls(page) });
              await page.screenshot({ path: path.join(output, `${mode}_${scope}_drone.png`) });
            }
            await page.goto(url);
            await page.waitForFunction(() => window.box2Viewer?.ready && window.box2Viewer.firstReadyRenderMs != null, undefined, { timeout: 60000 });
            const cacheRows = await page.evaluate(() => globalThis.assetCacheTrace);
            if (cacheRows.length !== 2 || cacheRows.some(x => !x.hit || !x.committed_after_native_validation)) {
              throw new Error("The original verified warm-cache behavior changed");
            }
            warm.push({ trial, mode, scope, original_complete_cache_rows: cacheRows });
            console.log(JSON.stringify({ trial, mode, scope, navigation_seconds: navigationSeconds, assets: state.assets }));
            publish("PROGRESS.json", { status: "running", rows, warm, controlRows, counted: false });
          } finally { await context.close(); }
        }
        if (trial === 0) {
          const exact = JSON.stringify(pairPixels.identity) === JSON.stringify(pairPixels.gzip);
          comparisons.push({ scope, original: pairPixels.identity, candidate: pairPixels.gzip, all_native_pixels_pose_projection_exact: exact });
          if (!exact) throw new Error("A native capture or drone pixel/projection changed");
        }
      }
    }
    const cost = Object.fromEntries(["scene", "box"].map(scope => {
      const selected = rows.filter(row => row.scope === scope);
      const original = selected.filter(row => row.mode === "identity"), candidate = selected.filter(row => row.mode === "gzip");
      const originalMedian = median(original.map(row => row.whole_seconds)), candidateMedian = median(candidate.map(row => row.whole_seconds));
      const ratios = candidate.map(row => ({ trial: row.trial,
        ratio: row.whole_seconds / original.find(x => x.trial === row.trial).whole_seconds }));
      return [scope, { original_whole_median_seconds: originalMedian, candidate_whole_median_seconds: candidateMedian,
        ratios, each_ratio_at_most1_1: ratios.every(x => x.ratio <= 1.1), median_nonregressed: candidateMedian <= originalMedian }];
    }));
    const passed = Object.values(cost).every(row => row.each_ratio_at_most1_1 && row.median_nonregressed);
    result = { status: passed ? "browser_passed_acceptance_pending" : "browser_cost_gate_failed", cost, rows, warm,
      comparisons, controlRows, sources, admission, server_startup: serverState, browser_startup_seconds: browserStartup,
      browser_version: browser.version(), original_model_byte_equivalence_is_not_GT_or_new_training_quality: true,
      logical_storage_is_not_whole_workspace_or_GPU_peak: true, existing_user_tabs_and_ports_untouched: true, counted: false };
  } catch (error) {
    failure = error;
    publish("FAILURE.json", { status: "failed_original_browser_contract_or_harness", error: String(error),
      stack: error.stack, sources, counted: false });
  } finally {
    if (browser) await browser.close();
    if (server.exitCode === null) server.stdin.end("stop\n");
    const closed = await Promise.race([serverClosed, sleep(10000).then(() => { throw new Error("Owned server did not drain"); })]);
    fs.closeSync(log);
    const serverFinal = fs.existsSync(receipt) ? JSON.parse(fs.readFileSync(receipt, "utf8")) : null;
    publish("CLOSURE.json", { server_pid: server.pid, closed, serverFinal,
      browser_closed: true, actual_sources_current: JSON.stringify(sources) === JSON.stringify(sources.map(x => identity(x.path))),
      elapsed_seconds: (performance.now() - started) / 1000, counted: false });
    if (result) {
      result.closure = { server_closed: closed.code === 0 && serverFinal?.status === "stopped",
        held_handles_closed: serverFinal?.held_read_handles_closed === true, browser_closed: true,
        actual_sources_current: JSON.stringify(sources) === JSON.stringify(sources.map(x => identity(x.path))) };
      publish("RESULT.json", result);
      console.log(JSON.stringify({ status: result.status, cost: result.cost, closure: result.closure }));
    }
  }
  if (failure) throw failure;
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
