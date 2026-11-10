// Profile two fresh original preview contexts and preserve all fixed-view pixel results.
const fs = require("fs"), path = require("path"), crypto = require("crypto");
const { chromium } = require("C:/Users/99303/git/playwright/node_modules/playwright");
const { execFileSync } = require("child_process");
const ROOT = path.resolve(__dirname, "../..");
const OUT = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup012/R167");
const RELEASE = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release");
const PAGE = path.join(RELEASE, "baseline/cases/packed");
const output = path.resolve(process.argv[2] ?? "");
const mode = process.argv[3] ?? "AB"; const traceController = false;
if (!["AB", "BA"].includes(mode)) throw new Error("The preregistered R167 pair order is unsupported");
if (!output.startsWith(OUT + path.sep) || fs.existsSync(output)) throw new Error("A fresh bounded private viewer cost output is required");
fs.mkdirSync(output, { recursive: true });
const identity = file => { const bytes = fs.readFileSync(file); return { path: file, bytes: bytes.length, sha256: crypto.createHash("sha256").update(bytes).digest("hex") }; };
const save = (name, value) => fs.writeFileSync(path.join(output, name), JSON.stringify(value, null, 2) + "\n", { flag: "wx" });
const helper = path.join(__dirname, "viewer_cost_probe.js");
const traceHelper = path.join(__dirname, "viewer_controller_trace.js");
const dagHelper = "C:\\lidar-camera-fusion-mask-robust-downsampling\\02_code\\gaussian_pipeline_optimization_v3_20261005\\orbit_convergence_dag_v2.js", genericHelper = "C:\\lidar-camera-fusion-mask-robust-downsampling\\02_code\\gaussian_pipeline_optimization_v3_20261005\\orbit_convergence_generic.js";
const originalDonor = "C:\\lidar-camera-fusion-mask-robust-downsampling\\04_outputs\\gaussian_pipeline_optimization_v3_20261005\\oup007\\p3\\sources\\COMPILED_DRIVER_BODY.cjs", radialHelper = "C:\\lidar-camera-fusion-mask-robust-downsampling\\02_code\\gaussian_pipeline_optimization_v3_20261005\\radial_dependency_observer_012.js", glHelper = "C:\\lidar-camera-fusion-mask-robust-downsampling\\02_code\\gaussian_pipeline_optimization_v3_20261005\\gl_resource_generation_v2.js";
const candidateHelper = "C:\\lidar-camera-fusion-mask-robust-downsampling\\02_code\\gaussian_pipeline_optimization_v3_20261005\\radial_translation_dependency_v3.js", candidateDonor = "C:\\lidar-camera-fusion-mask-robust-downsampling\\04_outputs\\gaussian_pipeline_optimization_v3_20261005\\oup012\\p0\\sources\\COMPILED_DRIVER_BODY.cjs", license = "C:\\lidar-camera-fusion-mask-robust-downsampling\\04_outputs\\gaussian_pipeline_optimization_v2_20261005\\viewer_execution_001\\release\\shared-build\\vendor\\SPARK-LICENSE.txt";
const nativeGenericHelper = "C:/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/radial_dependency_generic_001.js";
const sources = [__filename, "C:/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/oup012/R167/p0/sources/COMPILED_DRIVER_BODY.cjs", nativeGenericHelper, "C:\\lidar-camera-fusion-mask-robust-downsampling\\02_code\\gaussian_pipeline_optimization_v3_20261005\\run_radial_translation_dependency_001.cjs", candidateHelper, candidateDonor, license, originalDonor, radialHelper, glHelper, "C:\\lidar-camera-fusion-mask-robust-downsampling\\02_code\\gaussian_pipeline_optimization_v3_20261005\\profile_ordering_upload_cost_006.cjs", "C:\\lidar-camera-fusion-mask-robust-downsampling\\02_code\\gaussian_pipeline_optimization_v3_20261005\\ordering_allocation_cost_v2.js",  "C:\\lidar-camera-fusion-mask-robust-downsampling\\02_code\\gaussian_pipeline_optimization_v3_20261005\\profile_ordering_allocation_cost_007.cjs",  "C:\\lidar-camera-fusion-mask-robust-downsampling\\02_code\\gaussian_pipeline_optimization_v3_20261005\\run_orbit_convergence_dag.cjs", "C:\\lidar-camera-fusion-mask-robust-downsampling\\02_code\\gaussian_pipeline_optimization_v3_20261005\\ordering_upload_cost.js",  "C:\\lidar-camera-fusion-mask-robust-downsampling\\02_code\\gaussian_pipeline_optimization_v3_20261005\\profile_viewer_complete_cost.cjs", dagHelper, genericHelper, helper, ...(traceController ? [traceHelper] : []), path.join(PAGE, "viewer.js")].map(identity);
const inputs = [...fs.readdirSync(PAGE).filter(name => /\.(js|json|spk|html)$/.test(name) && !name.startsWith("export")).map(name => path.join(PAGE, name)),
  path.join(RELEASE, "baseline/ASSET_MANIFEST.json"), ...fs.readdirSync(path.join(RELEASE, "shared-build/vendor")).filter(name => /\.(js|txt)$/.test(name)).map(name => path.join(RELEASE, "shared-build/vendor", name))].map(identity);
fs.mkdirSync(path.join(output, "sources"));
for (const item of sources) fs.copyFileSync(item.path, path.join(output, "sources", item.path === originalDonor ? "DONOR_007_COMPILED_DRIVER_BODY.cjs" : item.path === candidateDonor ? "DONOR_012_COMPILED_DRIVER_BODY.cjs" : path.basename(item.path) === "COMPILED_DRIVER_BODY.cjs" ? "DONOR_NATIVE_COMPONENT_DRIVER.cjs" : path.basename(item.path)));
let helperText = fs.readFileSync(helper, "utf8"), nativeViewer = fs.readFileSync(path.join(PAGE, "viewer.js"), "utf8");
const traceText = traceController ? fs.readFileSync(traceHelper, "utf8") : null;
function once(text, before, after) { if (text.split(before).length !== 2) throw new Error("The complete original viewer source boundary changed"); return text.replace(before, after); }
helperText = once(helperText, "      count: [state.fullGaussianCount, state.boxGaussianCount],", "      quaternion: camera.quaternion.toArray(), world: camera.matrixWorld.toArray(),\n      count: [state.fullGaussianCount, state.boxGaussianCount],");
let viewer = 'import { installViewerCostProbe } from "./viewer_cost_probe.js";\n' + nativeViewer;
viewer = once(viewer, "  scene.add(new SparkRenderer({ renderer, onDirty: requestRender, ...filterOptions }));", "  const costSpark = new SparkRenderer({ renderer, onDirty: requestRender, ...filterOptions });\n  scene.add(costSpark);");
viewer = once(viewer, "  const modelLoadStarted = performance.now();", "  state.costProbe = installViewerCostProbe({ state, scene, camera, renderer, spark: costSpark, SplatMesh, preset, projection, setMode, syncFlightAngles, requestRender });\n  if (new URLSearchParams(location.search).get('hooks') === '1') state.costProbe.enable();\n  const modelLoadStarted = performance.now();");
if (traceController) {
  viewer = 'import { installViewerControllerTrace } from "./viewer_controller_trace.js";\n' + viewer;
  viewer = once(viewer, "  const modelLoadStarted = performance.now();", "  if (new URLSearchParams(location.search).get('hooks') === '1') state.controllerTrace = installViewerControllerTrace({ state, scene, camera, controls, renderer, spark: costSpark, SplatMesh });\n  const modelLoadStarted = performance.now();");
}

viewer = 'import { createOrbitConvergenceDag } from "./orbit_convergence_dag.js";\n' + viewer;
viewer = once(viewer, "  const radius = Math.max", "  const useOrbitDag = new URLSearchParams(location.search).get('hooks') === '1';\n  const orbitDag = createOrbitConvergenceDag({ controls, camera, state });\n  state.orbitDag = orbitDag; state.orbitControls = controls; state.orbitCamera = camera;\n  const radius = Math.max");
viewer = once(viewer, "  function applyView(index, fraction = 0) {", "  function applyView(index, fraction = 0) {\n    if (useOrbitDag) orbitDag.markDiscrete(fraction);");
viewer = once(viewer, "      orbitChanged = controls.update();", "      orbitChanged = useOrbitDag ? orbitDag.update() : controls.update();");
viewer = once(viewer, "  if (new URLSearchParams(location.search).get('hooks') === '1') state.costProbe.enable();", "  state.costProbe.enable();");

viewer = once(viewer, "  const useOrbitDag = new URLSearchParams(location.search).get('hooks') === '1';", "  const useOrbitDag = true;");
viewer = once(viewer, "  function stopPlayback() {", "  function stopPlayback() {\n    if (useOrbitDag) orbitDag.cancelDiscrete();");
helperText = 'import { installOrderingUploadCost } from "./ordering_upload_cost.js";\n' + helperText;
helperText = once(helperText, "  const originalRender = renderer.render;", "  const orderingCost = installOrderingUploadCost({ renderer, spark });\n  const originalRender = renderer.render;");
helperText = once(helperText, "      frame: state.frames, pendingRAF: state.isRenderPending() };", "      frame: state.frames, pendingRAF: state.isRenderPending(), orderingUpload: orderingCost.snapshot() };");
helperText = once(helperText, "      disposed = true;", "      const orderingUpload = orderingCost.dispose();\n      disposed = true;");
helperText = once(helperText, "      return { active, disposed, originalRender:", "      return { active, disposed, orderingUpload, originalRender:");
helperText = once(helperText, "originalReadback: spark.readbackDepth === originalReadback", "originalReadback: orderingAllocation.originalAliasRestored");
helperText = 'import { installOrderingAllocationCostV2 } from "./ordering_allocation_cost_v2.js";\n' + helperText;
helperText = once(helperText, "  const originalReadback = spark.readbackDepth;", "  const allocationCost = installOrderingAllocationCostV2({ spark });\n  const originalReadback = spark.readbackDepth;");
helperText = once(helperText, "orderingUpload: orderingCost.snapshot()", "orderingUpload: orderingCost.snapshot(), orderingAllocation: allocationCost.snapshot()");
helperText = once(helperText, "      const orderingUpload = orderingCost.dispose();", "      const orderingUpload = orderingCost.dispose();\n      const orderingAllocation = allocationCost.dispose();");
helperText = once(helperText, "      return { active, disposed, orderingUpload, originalRender:", "      return { active, disposed, orderingUpload, orderingAllocation, originalRender:");

viewer = 'import { installGlResourceGenerationV2 } from "./gl_resource_generation_v2.js";\nimport { installRadialTranslationDependency } from "./radial_translation_dependency_v3.js";\n' + viewer;
viewer = once(viewer, "  const costSpark = new SparkRenderer", "  state.glGeneration = installGlResourceGenerationV2({ renderer });\n  const costSpark = new SparkRenderer");
viewer = once(viewer, "  const modelLoadStarted = performance.now();", "  state.radialCache = installRadialTranslationDependency({ state, scene, camera, renderer, spark: costSpark, SplatMesh, glGeneration: state.glGeneration, enabled: new URLSearchParams(location.search).get('hooks') === '1' });\n  const modelLoadStarted = performance.now();");

viewer = 'import { runRadialDependencyGeneric } from "./radial_dependency_generic_001.js";\n' + viewer;
viewer = once(viewer, "  const modelLoadStarted = performance.now();", "  state.radialNativeGeneric = () => runRadialDependencyGeneric({ state, scene, camera, spark: costSpark, cache: state.radialCache });\n  const modelLoadStarted = performance.now();");
fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });
fs.writeFileSync(path.join(output, "sources/EXECUTED_PROBE.js"), helperText, { flag: "wx" });
const prefix = "http://127.0.0.1:28243", episodes = [], failures = [];
const freeDisk = fs.statfsSync(ROOT, { bigint: true });
if (freeDisk.bavail * freeDisk.bsize < 20n * 1024n ** 3n) throw new Error("The registered private-output disk reserve changed");
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
  await page.evaluate(name => window.box2Viewer.radialObserver?.setLabel(name), name);
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
    const body = relative.endsWith("/radial_dependency_generic_001.js") ? fs.readFileSync(nativeGenericHelper) : relative.endsWith("/radial_translation_dependency_v3.js") ? fs.readFileSync(candidateHelper) : relative.endsWith("/radial_dependency_observer_012.js") ? fs.readFileSync(radialHelper) : relative.endsWith("/gl_resource_generation_v2.js") ? fs.readFileSync(glHelper) : relative.endsWith("/ordering_allocation_cost_v2.js") ? fs.readFileSync("C:\\lidar-camera-fusion-mask-robust-downsampling\\02_code\\gaussian_pipeline_optimization_v3_20261005\\ordering_allocation_cost_v2.js") : relative.endsWith("/ordering_upload_cost.js") ? fs.readFileSync("C:\\lidar-camera-fusion-mask-robust-downsampling\\02_code\\gaussian_pipeline_optimization_v3_20261005\\ordering_upload_cost.js") : relative.endsWith("/orbit_convergence_dag.js") ? fs.readFileSync(dagHelper) : relative.endsWith("/orbit_convergence_generic.js") ? fs.readFileSync(genericHelper) : relative.endsWith("/viewer_cost_probe.js") ? helperText : traceController && relative.endsWith("/viewer_controller_trace.js") ? traceText : relative.endsWith("/viewer.js") ? viewer : fs.readFileSync(filename);
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

      const originView = views.find(view => view.name === range + "_capture_0").capture;
      for (let k = 1; k <= 6; ++k) {
        const angle = k * Math.PI / 12, d = originView.direction;
        const direction = [Math.cos(angle) * d[0] + Math.sin(angle) * d[2], d[1], -Math.sin(angle) * d[0] + Math.cos(angle) * d[2]];
        const position = [...originView.position];
        views.push(await step(page, range + "_pure_look_" + k, () => page.evaluate(({ position, direction }) => window.box2Viewer.costProbe.freeCamera(position, direction), { position, direction })));
      }
    }
    await page.getByLabel("移动方式").selectOption("fly");
    await page.locator("canvas").focus();
    await page.evaluate(() => window.box2Viewer.radialObserver?.setLabel("continuous_W"));
    const flyBegin = performance.now(), beforeFly = await page.evaluate(() => window.box2Viewer.costProbe.snapshot());
    await page.keyboard.down("w"); await page.waitForTimeout(1000); await page.keyboard.up("w");
    await settle(page, beforeFly.frame);
    const flight = { actual_begin_to_release_stable_ms: performance.now() - flyBegin, before: beforeFly, after: await page.evaluate(() => window.box2Viewer.costProbe.snapshot()), capture: await capture(page), fixed_pose_comparison: false };
    await page.evaluate(origin => { window.box2Viewer.radialGenericOrigin = { position: origin.position, direction: origin.direction }; }, views.find(v => v.name === "box_capture_0").capture);
    const nativeComponents = await page.evaluate(() => window.box2Viewer.radialNativeGeneric());
    const controllerTrace = traceController && hooks ? await page.evaluate(() => window.box2Viewer.controllerTrace.dispose()) : null;
    if (controllerTrace) save("CONTROLLER_TRACE.json", controllerTrace);
    const orbitDag = await page.evaluate(() => window.box2Viewer.orbitDag.dispose());
    const radialCache = await page.evaluate(() => window.box2Viewer.radialCache.dispose());
    const disposed = await page.evaluate(() => window.box2Viewer.costProbe.dispose());
    const radial = await page.evaluate(() => window.box2Viewer.radialObserver?.dispose() ?? null);
    const glGeneration = await page.evaluate(() => window.box2Viewer.glGeneration.dispose());
    if (radial) save("RADIAL_DEPENDENCY_TRACE.json", radial);
    if (!glGeneration.originalAliasesRestored || radial && !radial.originalAliasesRestored) throw new Error("The original radial observer or common GL aliases were not restored");
    if (!orbitDag.nativeUpdateUnchanged || !orbitDag.disposed || orbitDag.pending) throw new Error("The owned native controller lifecycle changed");
    result = { hooks, nativeComponents, radialCache, radial: radial ? { totals: radial.totals, eventCount: radial.eventCount, originalAliasesRestored: radial.originalAliasesRestored, active: radial.active } : null, glGeneration, orbitDag, completeColdToStableAndRGBA_ms, first, hardware, views, flight, disposed, errors, requests,
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

async function runGeneric(browser) {
  const pubFile = "C:/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/ocd001/public_inputs001/RESULT.json", pub = JSON.parse(fs.readFileSync(pubFile, "utf8"));
  const viewFile = path.join(PAGE, "camera_path.json"), metadataFile = path.join(PAGE, "display_manifest.json");
  const values = { views: JSON.parse(fs.readFileSync(viewFile, "utf8")), publicViews: pub.poses,
    focus: JSON.parse(fs.readFileSync(metadataFile, "utf8")).focus };
  if (values.views.length !== 1016 || values.publicViews.length !== 360) throw new Error("The original whole pose denominators changed");
  const context = await browser.newContext({ viewport: { width: 960, height: 600 }, deviceScaleFactor: 1 }), page = await context.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(String(error)));
  await context.route("**/*", async route => {
    const url = new URL(route.request().url());
    if (url.origin !== prefix) throw new Error("The generic diagnostic requested an external origin");
    const relative = decodeURIComponent(url.pathname.replace(/^\/(native|profile)\//, "")), filename = path.resolve(RELEASE, relative);
    if (!filename.startsWith(RELEASE + path.sep)) throw new Error("A private generic request escaped the original release");
    const body = relative.endsWith("/radial_dependency_generic_001.js") ? fs.readFileSync(nativeGenericHelper) : relative.endsWith("/radial_translation_dependency_v3.js") ? fs.readFileSync(candidateHelper) : relative.endsWith("/radial_dependency_observer_012.js") ? fs.readFileSync(radialHelper) : relative.endsWith("/gl_resource_generation_v2.js") ? fs.readFileSync(glHelper) : relative.endsWith("/ordering_allocation_cost_v2.js") ? fs.readFileSync("C:\\lidar-camera-fusion-mask-robust-downsampling\\02_code\\gaussian_pipeline_optimization_v3_20261005\\ordering_allocation_cost_v2.js") : relative.endsWith("/ordering_upload_cost.js") ? fs.readFileSync("C:\\lidar-camera-fusion-mask-robust-downsampling\\02_code\\gaussian_pipeline_optimization_v3_20261005\\ordering_upload_cost.js") : relative.endsWith("/orbit_convergence_dag.js") ? fs.readFileSync(dagHelper) : relative.endsWith("/orbit_convergence_generic.js") ? fs.readFileSync(genericHelper) : relative.endsWith("/viewer_cost_probe.js") ? helperText : relative.endsWith("/viewer.js") ? viewer : fs.readFileSync(filename);
    const extension = path.extname(filename);
    await route.fulfill({ status: 200, body, contentType: extension === ".js" ? "text/javascript" : extension === ".json" ? "application/json" : extension === ".html" ? "text/html" : "application/octet-stream" });
  });
  try {
    await page.goto(prefix + "/native/baseline/cases/packed/index.html?hooks=0", { waitUntil: "domcontentloaded" });
    await settle(page);
    const result = await page.evaluate(async values => (await import("./orbit_convergence_generic.js")).runOrbitConvergenceGeneric(values), values);
    result.checks.original_vendor_and_all_assets_unchanged = JSON.stringify(inputs) === JSON.stringify(inputs.map(row => identity(row.path)));
    result.checks.original_complete_public_inputs_unchanged = pub.inputs.every(row => JSON.stringify(row) === JSON.stringify(identity(row.path)));
    result.checks.generic_helper_and_complete_primary_sources_unchanged = JSON.stringify(sources) === JSON.stringify(sources.map(row => identity(row.path)));
    result.checks.actual_native_page_errors_empty = errors.length === 0;
    result.inputs = [...inputs, ...pub.inputs, identity(pubFile)]; result.sources = sources;
    result.errors = errors; result.new_optimization_count = 0;
    save("CONTROLLER_GENERIC_RESULT.json", result);
    console.log(JSON.stringify({ checks: result.checks, summaries: result.summaries, routes: result.routes }));
    if (!Object.values(result.checks).every(Boolean)) throw new Error("The full original pose or generic contract failed");
    await page.evaluate(() => { window.box2Viewer.radialCache.dispose(); window.box2Viewer.orbitDag.dispose(); window.box2Viewer.costProbe.dispose(); window.box2Viewer.glGeneration.dispose(); });
  } finally { await context.close(); }
}

(async () => {
  save("PROTOCOL.json", { sources, inputs, viewport: [960, 600], deviceScaleFactor: 1, original_angle_d3d11: true,
    registered_context_orders: ["baseline","candidate"], registered_extra_pure_looks: { angles_degrees: [15,30,45,60,75,90], origin_per_range: "original_capture_0", axis: "world_Y", all_original21_plus12: true }, symmetric_original_cost_hooks: true, mode, traceController,
    complete_entry_not_GPU_kernel_time: true, new_optimization_count: 0, scope: "public_controller_interface_only_not_formal_speed" });
  const gpuBefore = gpu(), browser = await chromium.launch({ headless: true, args: ["--use-angle=d3d11"] });
  const gpuSamples = [gpuBefore], gpuFailures = [];
  const gpuMonitor = setInterval(() => { try { gpuSamples.push(gpu()); } catch (error) { gpuFailures.push(String(error)); void browser.close(); } }, 2000);
  try {
    await runGeneric(browser);
    const component = JSON.parse(fs.readFileSync(path.join(output, "CONTROLLER_GENERIC_RESULT.json"), "utf8"));
    save("RESULT.json", { ...component, status: "R167_public_controller_interface_component0", gpuBefore, gpuAfter: gpu(), gpuSamples, gpuFailures, root_accepted_count: 36, new_optimization_count: 0, standalone_native_context_closed_by_runGeneric: true });
    if (!Object.values(component.checks).every(Boolean)) throw new Error("The original complete public controller component failed");
  } catch (error) {
    failures.push({ name: error.name, message: error.message, stack: error.stack });
    save("FAILURE.json", { status: "failed_complete_preview_diagnostic0", failures, completed_episodes: episodes.length, sources, inputs, new_optimization_count: 0 });
    throw error;
  } finally { clearInterval(gpuMonitor); await browser.close(); save("GPU_RESOURCE_SAMPLES.json", { gpuSamples, gpuFailures }); }
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
