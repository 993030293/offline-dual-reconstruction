// Profile two fresh original preview contexts and preserve all fixed-view pixel results.
const fs = require("fs"), path = require("path"), crypto = require("crypto");
const { chromium } = require("C:/Users/99303/git/playwright/node_modules/playwright");
const { execFileSync } = require("child_process");
const ROOT = path.resolve(__dirname, "../..");
const OUT = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v3_20261005/viewer_cost_005");
const RELEASE = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release");
const PAGE = path.join(RELEASE, "baseline/cases/packed");
const output = path.resolve(process.argv[2] ?? "");
const traceController = process.argv[3] === "--trace-controller";
if (process.argv[3] && !traceController) throw new Error("The diagnostic mode argument is unsupported");
if (!output.startsWith(OUT + path.sep) || fs.existsSync(output)) throw new Error("A fresh bounded private viewer cost output is required");
fs.mkdirSync(output, { recursive: true });
const identity = file => { const bytes = fs.readFileSync(file); return { path: file, bytes: bytes.length, sha256: crypto.createHash("sha256").update(bytes).digest("hex") }; };
const save = (name, value) => fs.writeFileSync(path.join(output, name), JSON.stringify(value, null, 2) + "\n", { flag: "wx" });
const helper = path.join(__dirname, "viewer_cost_probe.js");
const traceHelper = path.join(__dirname, "viewer_controller_trace.js");
const sources = [__filename, helper, ...(traceController ? [traceHelper] : []), path.join(PAGE, "viewer.js")].map(identity);
const inputs = [...fs.readdirSync(PAGE).filter(name => /\.(js|json|spk|html)$/.test(name) && !name.startsWith("export")).map(name => path.join(PAGE, name)),
  path.join(RELEASE, "baseline/ASSET_MANIFEST.json"), ...fs.readdirSync(path.join(RELEASE, "shared-build/vendor")).filter(name => /\.(js|txt)$/.test(name)).map(name => path.join(RELEASE, "shared-build/vendor", name))].map(identity);
fs.mkdirSync(path.join(output, "sources"));
for (const item of sources) fs.copyFileSync(item.path, path.join(output, "sources", path.basename(item.path)));
const helperText = fs.readFileSync(helper, "utf8"), nativeViewer = fs.readFileSync(path.join(PAGE, "viewer.js"), "utf8");
const traceText = traceController ? fs.readFileSync(traceHelper, "utf8") : null;
function once(text, before, after) { if (text.split(before).length !== 2) throw new Error("The complete original viewer source boundary changed"); return text.replace(before, after); }
let viewer = 'import { installViewerCostProbe } from "./viewer_cost_probe.js";\n' + nativeViewer;
viewer = once(viewer, "  scene.add(new SparkRenderer({ renderer, onDirty: requestRender, ...filterOptions }));", "  const costSpark = new SparkRenderer({ renderer, onDirty: requestRender, ...filterOptions });\n  scene.add(costSpark);");
viewer = once(viewer, "  const modelLoadStarted = performance.now();", "  state.costProbe = installViewerCostProbe({ state, scene, camera, renderer, spark: costSpark, SplatMesh, preset, projection, setMode, syncFlightAngles, requestRender });\n  if (new URLSearchParams(location.search).get('hooks') === '1') state.costProbe.enable();\n  const modelLoadStarted = performance.now();");
if (traceController) {
  viewer = 'import { installViewerControllerTrace } from "./viewer_controller_trace.js";\n' + viewer;
  viewer = once(viewer, "  const modelLoadStarted = performance.now();", "  if (new URLSearchParams(location.search).get('hooks') === '1') state.controllerTrace = installViewerControllerTrace({ state, scene, camera, controls, renderer, spark: costSpark, SplatMesh });\n  const modelLoadStarted = performance.now();");
}
fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });
const prefix = "http://127.0.0.1:28231", episodes = [], failures = [];
const started = performance.now();
function gpu() {
  const values = execFileSync("nvidia-smi", ["--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"], { encoding: "utf8" }).trim().split(",").map(Number);
  if (values.length !== 3 || values.some(x => !Number.isFinite(x)) || values[1] > 12288 || values[2] < 2048) throw new Error("The registered device resource reserve changed");
  return { totalMiB: values[0], usedMiB: values[1], freeMiB: values[2], scope: "whole_device_not_owned_browser_peak" };
}
async function settle(page, previous = -1) {
  await page.waitForFunction(prior => {
    const s = window.box2Viewer, p = s?.costProbe?.snapshot();
    return s?.ready && !s.modelLoading && s.frames > prior && !s.isRenderPending() && p && !p.sorting && !p.sortDirty && !p.sortTimeout && p.active === 0;
  }, previous, { timeout: 30000, polling: "raf" });
}
async function capture(page) { return page.evaluate(() => window.box2Viewer.costProbe.capture()); }
async function step(page, name, action) {
  const before = await page.evaluate(() => ({ at: performance.now(), snapshot: window.box2Viewer.costProbe.snapshot() }));
  const wall = performance.now();
  await action(); await settle(page, before.snapshot.frame);
  const stable = await page.evaluate(() => ({ at: performance.now(), snapshot: window.box2Viewer.costProbe.snapshot() }));
  const result = await capture(page);
  return { name, Node_action_to_stable_and_RGBA_ms: performance.now() - wall, browser_action_boundary_to_stable_ms: stable.at - before.at,
    before: before.snapshot, stable: stable.snapshot, capture: result };
}
async function episode(browser, hooks) {
  const wall = performance.now(), context = await browser.newContext({ viewport: { width: 960, height: 600 }, deviceScaleFactor: 1 });
  const page = await context.newPage(), errors = [], views = [], requests = [];
  page.on("pageerror", e => errors.push(String(e)));
  await context.route("**/*", async route => {
    const url = new URL(route.request().url());
    if (url.origin !== prefix) throw new Error("The diagnostic page requested an external origin");
    const relative = decodeURIComponent(url.pathname.replace(/^\/(native|profile)\//, ""));
    const filename = path.resolve(RELEASE, relative);
    if (!filename.startsWith(RELEASE + path.sep)) throw new Error("A private request escaped the frozen release");
    const body = relative.endsWith("/viewer_cost_probe.js") ? helperText : traceController && relative.endsWith("/viewer_controller_trace.js") ? traceText : relative.endsWith("/viewer.js") ? viewer : fs.readFileSync(filename);
    const extension = path.extname(filename);
    requests.push({ relative, bytes: Buffer.byteLength(body), source: relative.endsWith("/viewer.js") ? "diagnostic_viewer_only" : "original_asset_or_private_helper" });
    await route.fulfill({ status: 200, body, contentType: extension === ".js" ? "text/javascript" : extension === ".json" ? "application/json" : extension === ".html" ? "text/html" : "application/octet-stream" });
  });
  let result;
  try {
    const loadStart = performance.now();
    await page.goto(prefix + "/" + (hooks ? "profile" : "native") + "/baseline/cases/packed/index.html?hooks=" + Number(hooks), { waitUntil: "domcontentloaded" });
    await settle(page);
    const first = await capture(page), completeColdToStableAndRGBA_ms = performance.now() - loadStart;
    const hardware = await page.evaluate(() => {
      const gl = window.box2Viewer.renderer.getContext(), ext = gl.getExtension("WEBGL_debug_renderer_info");
      return { renderer: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : null, version: gl.getParameter(gl.VERSION), timerQueryAvailable: Boolean(gl.getExtension("EXT_disjoint_timer_query_webgl2")) };
    });
    if (!hardware.renderer?.includes("RTX 5060 Ti") || first.glError !== 0 || errors.length) throw new Error("The actual complete preview renderer or initial frame failed");
    await page.evaluate(() => window.box2Viewer.costProbe.attachWorker());
    for (const range of ["scene", "box"]) {
      if (range === "box") {
        const before = await page.evaluate(() => window.box2Viewer.frames), switched = performance.now();
        await page.getByLabel("展示范围").selectOption(range); await settle(page, before);
        views.push({ name: "switch_box", Node_complete_switch_ms: performance.now() - switched, capture: await capture(page) });
      }
      const count = await page.evaluate(() => Number(document.getElementById("path-slider").max) + 1);
      for (let i = 0; i < 6; ++i) {
        const index = Math.floor(i * (count - 1) / 5);
        views.push(await step(page, range + "_capture_" + index, () => page.evaluate(index => window.box2Viewer.setView(index), index)));
      }
      for (const preset of ["top", "drone"]) views.push(await step(page, range + "_" + preset, () => page.locator('button[data-preset="' + preset + '"]').click()));
      const bounds = await page.evaluate(() => window.box2Viewer.costProbe.bounds());
      for (let i = 0; i < 2; ++i) {
        const position = bounds.center.map((value, axis) => value + bounds.radius * (axis === 0 ? 2 + i * .2 : axis === 1 ? .4 + i * .2 : .6));
        const direction = bounds.center.map((value, axis) => value - position[axis]);
        views.push(await step(page, range + "_free_" + i, () => page.evaluate(({ position, direction }) => window.box2Viewer.costProbe.freeCamera(position, direction), { position, direction })));
      }
    }
    await page.getByLabel("移动方式").selectOption("fly");
    await page.locator("canvas").focus();
    const flyBegin = performance.now(), beforeFly = await page.evaluate(() => window.box2Viewer.costProbe.snapshot());
    await page.keyboard.down("w"); await page.waitForTimeout(1000); await page.keyboard.up("w");
    await settle(page, beforeFly.frame);
    const flight = { actual_begin_to_release_stable_ms: performance.now() - flyBegin, before: beforeFly, after: await page.evaluate(() => window.box2Viewer.costProbe.snapshot()), capture: await capture(page), fixed_pose_comparison: false };
    const controllerTrace = traceController && hooks ? await page.evaluate(() => window.box2Viewer.controllerTrace.dispose()) : null;
    if (controllerTrace) save("CONTROLLER_TRACE.json", controllerTrace);
    const disposed = await page.evaluate(() => window.box2Viewer.costProbe.dispose());
    result = { hooks, completeColdToStableAndRGBA_ms, first, hardware, views, flight, disposed, errors, requests,
      controllerTrace: controllerTrace ? { eventCount: controllerTrace.eventCount, disposed: controllerTrace.disposed,
        originalUpdate: controllerTrace.originalUpdate, originalRender: controllerTrace.originalRender } : null,
      totals_include_nested_not_additive: true, first_worker_constructor_sort_not_observed_by_worker_call_hook: true };
    if (errors.length || views.some(view => view.capture.glError !== 0) || !views.some(view => view.capture.nonuniformPixels > 0) || !Object.entries(disposed).every(([k, v]) => k === "observedWorker" || Boolean(v) || k === "active" && v === 0)) throw new Error("The original stable-view pixels, errors, or hook drain failed");
  } catch (error) {
    let actualState;
    try { actualState = await page.evaluate(() => window.box2Viewer?.costProbe?.snapshot()); }
    catch (snapshotError) { actualState = { snapshotError: String(snapshotError) }; }
    save("FAILED_EPISODE_" + Number(hooks) + ".json", { hooks, completed_views: views, errors, requests, actualState, primaryError: String(error) });
    throw error;
  } finally { await context.close(); }
  result.complete_context_to_actual_close_ms = performance.now() - wall;
  episodes.push(result); save("EPISODE_" + Number(hooks) + ".json", result);
  console.log(JSON.stringify({ episode: hooks ? "profile" : "native", wholeMs: result.complete_context_to_actual_close_ms, coldMs: result.completeColdToStableAndRGBA_ms, frames: result.flight.after.frame }), null, 0);
}
(async () => {
  save("PROTOCOL.json", { sources, inputs, viewport: [960, 600], deviceScaleFactor: 1, original_angle_d3d11: true,
    registered_context_orders: ["original_no_hooks", "same_original_timing_hooks"], traceController,
    complete_entry_not_GPU_kernel_time: true, new_optimization_count: 0 });
  const gpuBefore = gpu(), browser = await chromium.launch({ headless: true, args: ["--use-angle=d3d11"] });
  try {
    await episode(browser, false); await episode(browser, true);
    const left = episodes[0], right = episodes[1];
    const keys = ["rgba_sha256", "position", "direction", "projection", "dimensions", "count", "mode", "activeSplats", "currentSplats"];
    const fixedExact = left.views.length === right.views.length && left.views.every((view, i) => view.name === right.views[i].name && keys.every(key => JSON.stringify(view.capture[key]) === JSON.stringify(right.views[i].capture[key])));
    const checks = { fixed_all21_view_results_model_camera_and_RGBA_exact: fixedExact,
      initial_stable_capture_exact: keys.every(key => JSON.stringify(left.first[key]) === JSON.stringify(right.first[key])),
      actual_hardware_renderer_and_nonempty_original_views: episodes.every(e => e.hardware.renderer.includes("RTX 5060 Ti") && e.views.some(v => v.capture.nonuniformPixels > 0)),
      original_vendor_and_all_inputs_unchanged: JSON.stringify(inputs) === JSON.stringify(inputs.map(row => identity(row.path))),
      own_original_sources_unchanged: JSON.stringify(sources) === JSON.stringify(sources.map(row => identity(row.path))),
      original_aliases_restored_and_native_calls_drained: episodes.every(e => e.disposed.active === 0 && e.disposed.originalRender && e.disposed.originalReadback && e.disposed.originalDriveSort),
      observed_current_native_worker_calls_present_in_profile: right.flight.after.totals.observedWorkerCalls > 0,
      no_original_page_or_GL_errors: episodes.every(e => !e.errors.length && e.first.glError === 0 && e.views.every(v => v.capture.glError === 0)),
      original_controller_trace_methods_restored: !traceController || right.controllerTrace.disposed && right.controllerTrace.originalUpdate && right.controllerTrace.originalRender };
    save("RESULT.json", { status: Object.values(checks).every(Boolean) ? "complete_preview_cost_diagnostic_finished0" : "failed_diagnostic_verification", checks, sources, inputs,
      gpuBefore, gpuAfter: gpu(), episode_files: ["EPISODE_0.json", "EPISODE_1.json"], episodes, elapsed_seconds_before_browser_close: (performance.now() - started) / 1000,
      root_accepted_count: 33, new_optimization_count: 0, source_only_hypothesis_no_candidate_or_formal: true });
    if (!Object.values(checks).every(Boolean)) throw new Error("The complete original instrumentation comparison failed");
    console.log(JSON.stringify({ checks, newCount: 0 }));
  } catch (error) {
    failures.push({ name: error.name, message: error.message, stack: error.stack });
    save("FAILURE.json", { status: "failed_complete_preview_diagnostic0", failures, completed_episodes: episodes.length, sources, inputs, new_optimization_count: 0 });
    throw error;
  } finally { await browser.close(); }
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
