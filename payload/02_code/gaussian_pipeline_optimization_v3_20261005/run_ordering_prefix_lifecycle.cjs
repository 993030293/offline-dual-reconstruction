// Exercise complete public controller components and actual preview context and callback lifecycles.
const fs = require("fs"), path = require("path"), crypto = require("crypto");
const { chromium } = require("C:/Users/99303/git/playwright/node_modules/playwright");
const { execFileSync } = require("child_process");
const ROOT = path.resolve(__dirname, "../.."), OUT = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup006/R163");
const RELEASE = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release");
const PAGE = path.join(RELEASE, "baseline/cases/packed"), output = path.resolve(process.argv[2] ?? ""), mode = process.argv[3];
if (!output.startsWith(OUT + path.sep) || fs.existsSync(output) || !["public", "context", "lifecycle"].includes(mode)) throw new Error("A fresh bounded lifecycle mode and output are required");
fs.mkdirSync(path.join(output, "sources"), { recursive: true });
const save = (name, value) => fs.writeFileSync(path.join(output, name), JSON.stringify(value, null, 2) + "\n", { flag: "wx" });
const identity = file => { const bytes = fs.readFileSync(file); return { path: file, bytes: bytes.length, sha256: crypto.createHash("sha256").update(bytes).digest("hex") }; };
const prior = JSON.parse(fs.readFileSync(path.join(OUT, "c0/RESULT.json"), "utf8"));
const viewerFile = path.join(OUT, "c0/sources/EXECUTED_VIEWER.js"), probeFile = path.join(OUT, "c0/sources/EXECUTED_PROBE.js");
const helperFile = path.join(__dirname, "ordering_active_prefix.js"), orbitFile = path.join(__dirname, "orbit_convergence_dag_v2.js"), genericFile = path.join(__dirname, "orbit_convergence_generic.js");
const publicFile = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v3_20261005/ocd001/public_inputs001/RESULT.json");
const sources = [__filename, helperFile, orbitFile, genericFile].map(identity);
const inputs = [...prior.inputs, identity(viewerFile), identity(probeFile), identity(publicFile)];
for (const row of sources) fs.copyFileSync(row.path, path.join(output, "sources", path.basename(row.path)));
const viewer = fs.readFileSync(viewerFile), originalProbe = fs.readFileSync(probeFile, "utf8");
function once(source, before, after) { if (source.split(before).length !== 2) throw new Error("The actual lifecycle source boundary changed"); return source.replace(before, after); }
const prefix = "http://127.0.0.1:28234";
let probe = once(originalProbe, "  const originalRender = renderer.render;", "  state.orderingDiagnostics = { helper: orderingCost, spark, renderer };\n  const originalRender = renderer.render;");
if (mode === "lifecycle") {
  const fault = `  const glFault = renderer.getContext(), originalFaultUpload = glFault.texSubImage2D;
  const faultUpload = function (...args) {
    if (state.uploadFault?.armed && args.length === 9 && args[4] === 4096 && args[6] === glFault.RGBA_INTEGER && args[7] === glFault.UNSIGNED_INT && args[8] instanceof Uint32Array) {
      state.uploadFault.calls++;
      if (state.uploadFault.calls === 2) throw new Error("registered actual original-WebGL-upload callback failure");
    }
    return originalFaultUpload.apply(this, args);
  };
  glFault.texSubImage2D = faultUpload;
  state.restoreUploadFault = () => { if (glFault.texSubImage2D !== faultUpload) throw new Error("The owned WebGL fault alias changed"); glFault.texSubImage2D = originalFaultUpload; return glFault.texSubImage2D === originalFaultUpload; };
`;
  probe = once(probe, "  const orderingCost = installOrderingActivePrefix", fault + "  const orderingCost = installOrderingActivePrefix");
}
fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });
fs.writeFileSync(path.join(output, "sources/EXECUTED_PROBE.js"), probe, { flag: "wx" });
function gpu() {
  const values = execFileSync("nvidia-smi", ["--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"], { encoding: "utf8" }).trim().split(",").map(Number);
  if (values.length !== 3 || values[1] > 12288 || values[2] < 2048) throw new Error("The original GPU resource reserve changed");
  return { usedMiB: values[1], freeMiB: values[2], wholeDeviceNotOwnedPeak: true };
}
async function setup(browser, enabled) {
  const context = await browser.newContext({ viewport: { width: 960, height: 600 }, deviceScaleFactor: 1 }), page = await context.newPage(), errors = [];
  page.on("pageerror", error => errors.push(String(error)));
  await context.route("**/*", async route => {
    const url = new URL(route.request().url());
    if (url.origin !== prefix) throw new Error("The lifecycle requested an external origin");
    const relative = decodeURIComponent(url.pathname.replace(/^\/(native|profile)\//, "")), filename = path.resolve(RELEASE, relative);
    if (!filename.startsWith(RELEASE + path.sep)) throw new Error("The lifecycle escaped original release inputs");
    const body = relative.endsWith("/ordering_active_prefix.js") ? fs.readFileSync(helperFile) : relative.endsWith("/orbit_convergence_dag.js") ? fs.readFileSync(orbitFile) : relative.endsWith("/orbit_convergence_generic.js") ? fs.readFileSync(genericFile) : relative.endsWith("/viewer_cost_probe.js") ? probe : relative.endsWith("/viewer.js") ? viewer : fs.readFileSync(filename);
    const extension = path.extname(filename);
    await route.fulfill({ status: 200, body, contentType: extension === ".js" ? "text/javascript" : extension === ".json" ? "application/json" : extension === ".html" ? "text/html" : "application/octet-stream" });
  });
  await page.goto(prefix + "/" + (enabled ? "profile" : "native") + "/baseline/cases/packed/index.html?hooks=" + Number(enabled), { waitUntil: "domcontentloaded" });
  await settle(page);
  return { context, page, errors };
}
async function settle(page, previous = -1) {
  await page.waitForFunction(priorFrame => { const s = window.box2Viewer, p = s?.costProbe?.snapshot(); return s?.ready && !s.modelLoading && s.frames > priorFrame && !s.isRenderPending() && p && !p.sorting && !p.sortDirty && !p.sortTimeout && p.active === 0; }, previous, { timeout: 30000, polling: "raf" });
}
async function capture(page) { return page.evaluate(() => window.box2Viewer.costProbe.capture()); }
async function step(page, name, action) { const frame = await page.evaluate(() => window.box2Viewer.frames); await action(); await settle(page, frame); return { name, capture: await capture(page) }; }
async function closeAliases(page) { return page.evaluate(() => { const s = window.box2Viewer; return { dag: s.orbitDag.dispose(), timing: s.costProbe.dispose() }; }); }
const keys = ["rgba_sha256", "position", "direction", "quaternion", "world", "projection", "dimensions", "count", "mode", "activeSplats", "currentSplats"];
const exact = (a, b) => keys.every(key => JSON.stringify(a[key]) === JSON.stringify(b[key]));
async function allFixed(page) {
  const views = [];
  if (await page.evaluate(() => window.box2Viewer.sceneMode !== "scene")) { const frame = await page.evaluate(() => window.box2Viewer.frames); await page.getByLabel("展示范围").selectOption("scene"); await settle(page, frame); }
  for (const range of ["scene", "box"]) {
    if (range === "box") views.push(await step(page, "switch_box", () => page.getByLabel("展示范围").selectOption("box")));
    const count = await page.evaluate(() => Number(document.getElementById("path-slider").max) + 1);
    for (let i = 0; i < 6; ++i) { const index = Math.floor(i * (count - 1) / 5); views.push(await step(page, range + "_capture_" + index, () => page.evaluate(index => window.box2Viewer.setView(index), index))); }
    for (const preset of ["top", "drone"]) views.push(await step(page, range + "_" + preset, () => page.locator('button[data-preset="' + preset + '"]').click()));
    const bounds = await page.evaluate(() => window.box2Viewer.costProbe.bounds());
    for (let i = 0; i < 2; ++i) {
      const position = bounds.center.map((value, axis) => value + bounds.radius * (axis === 0 ? 2 + i * .2 : axis === 1 ? .4 + i * .2 : .6));
      const direction = bounds.center.map((value, axis) => value - position[axis]);
      views.push(await step(page, range + "_free_" + i, () => page.evaluate(values => window.box2Viewer.costProbe.freeCamera(values.position, values.direction), { position, direction })));
    }
  }
  return views;
}
async function contextCase(browser, enabled) {
  const { context, page, errors } = await setup(browser, enabled); let result;
  try {
    await step(page, "preloss_switch_box", () => page.getByLabel("展示范围").selectOption("box"));
    await step(page, "preloss_box203", () => page.evaluate(() => window.box2Viewer.setView(203)));
    const before = await page.evaluate(async () => {
      const s = window.box2Viewer, d = s.orderingDiagnostics, data = d.spark.orderingTexture.image.data;
      s.contextAlias = { texture: d.spark.orderingTexture, data };
      const hash = await crypto.subtle.digest("SHA-256", data);
      s.contextDataHash = Array.from(new Uint8Array(hash));
      s.renderer.domElement.addEventListener("webglcontextlost", () => { s.actualContextLost = true; }, { once: true });
      s.renderer.domElement.addEventListener("webglcontextrestored", () => { s.actualContextRestored = true; }, { once: true });
      const snapshot = s.costProbe.snapshot(); s.renderer.forceContextLoss(); return snapshot;
    });
    await page.waitForFunction(() => window.box2Viewer.actualContextLost && window.box2Viewer.renderer.getContext().isContextLost(), undefined, { timeout: 10000 });
    await page.evaluate(() => window.box2Viewer.renderer.forceContextRestore());
    await page.waitForFunction(() => window.box2Viewer.actualContextRestored && !window.box2Viewer.renderer.getContext().isContextLost(), undefined, { timeout: 10000 });
    const restoredFirst = await step(page, "first_restored_stable_box0", () => page.evaluate(() => window.box2Viewer.setView(0)));
    const restoration = await page.evaluate(async () => {
      const s = window.box2Viewer, d = s.orderingDiagnostics, data = d.spark.orderingTexture.image.data;
      const hash = await crypto.subtle.digest("SHA-256", data);
      return { actualContextLost: s.actualContextLost, actualContextRestored: s.actualContextRestored,
        sameTexture: s.contextAlias.texture === d.spark.orderingTexture, sameOriginalImageData: s.contextAlias.data === data,
        originalImageDataHashExact: JSON.stringify(s.contextDataHash) === JSON.stringify(Array.from(new Uint8Array(hash))), snapshot: s.costProbe.snapshot() };
    });
    const views = await allFixed(page), reference = prior.episodes.find(row => !row.hooks);
    const fixedExact = views.length === 21 && views.every((row, i) => row.name === reference.views[i].name && exact(row.capture, reference.views[i].capture));
    const firstExpected = reference.views.find(row => row.name === "box_capture_0").capture;
    const disposed = await closeAliases(page);
    result = { enabled, before, restoredFirst, restoration, views, disposed, errors, checks: {
      actual_loss_and_restore_events: restoration.actualContextLost && restoration.actualContextRestored,
      original_texture_and_image_alias_hash_preserved: restoration.sameTexture && restoration.sameOriginalImageData && restoration.originalImageDataHashExact,
      first_restored_stable_original_sort_camera_RGBA_exact: exact(restoredFirst.capture, firstExpected),
      complete21_after_restore_camera_and_RGBA_exact: fixedExact,
      actual_restoration_initial_upload_full_image_alias: restoration.snapshot.orderingUpload.events.slice(before.orderingUpload.events.length).some(row => row.initialImageAlias && row.uploadedRows === row.requestedRows),
      native_GL_errors_and_page_errors_empty: !errors.length && restoredFirst.capture.glError === 0 && views.every(row => row.capture.glError === 0),
      owned_original_aliases_drained_restored: disposed.timing.active === 0 && disposed.timing.orderingUpload.originalAliasRestored && disposed.dag.nativeUpdateUnchanged,
    } };
    save("CONTEXT_" + Number(enabled) + ".json", result);
    if (!Object.values(result.checks).every(Boolean)) throw new Error("The actual original context restoration contract failed");
  } catch (error) {
    save("FAILED_CONTEXT_" + Number(enabled) + ".json", { error: String(error), result, errors });
    throw error;
  } finally { await context.close(); }
  result.actualContextClosed = true; return result;
}
async function lifecycleCase(browser, enabled, kind) {
  const { context, page, errors } = await setup(browser, enabled), completedViews = []; let result;
  try {
    completedViews.push(await step(page, "prior_switch_box", () => page.getByLabel("展示范围").selectOption("box")));
    completedViews.push(await step(page, "prior_box0", () => page.evaluate(() => window.box2Viewer.setView(0))));
    const before = await page.evaluate(() => window.box2Viewer.costProbe.snapshot());
    if (kind === "fault") {
      await page.evaluate(() => { window.box2Viewer.uploadFault = { armed: true, calls: 0 }; });
      completedViews.push(await step(page, "actual_first_fault_callback_box203", () => page.evaluate(() => window.box2Viewer.setView(203))));
      await page.evaluate(() => window.box2Viewer.setView(406));
      await page.waitForFunction(() => window.box2Viewer.errors.some(value => value.includes("registered actual original-WebGL-upload callback failure")), undefined, { timeout: 10000, polling: "raf" });
      const observed = await page.evaluate(() => ({ fault: { ...window.box2Viewer.uploadFault }, uiErrors: window.box2Viewer.errors.slice(), snapshot: window.box2Viewer.costProbe.snapshot() }));
      await page.evaluate(() => { window.box2Viewer.uploadFault.armed = false; });
      await page.waitForFunction(() => window.box2Viewer.costProbe.snapshot().active === 0, undefined, { timeout: 10000 });
      const restored = await closeAliases(page), faultAliasRestored = await page.evaluate(() => window.box2Viewer.restoreUploadFault());
      if (observed.fault.calls !== 2 || !errors.some(value => value.includes("registered actual original-WebGL-upload callback failure")) || !faultAliasRestored || !restored.timing.orderingUpload.originalAliasRestored || restored.timing.active !== 0) throw new Error("The actual upload callback error propagation or alias restoration failed");
      result = { enabled, kind, completedViews, before, observed, errors, restored, faultAliasRestored,
        failed_original_context_not_repaired_or_claimed_reusable: true };
    } else {
      await page.evaluate(() => window.box2Viewer.setView(406));
      await page.getByLabel("移动方式").selectOption("fly"); await page.locator("canvas").focus(); await page.keyboard.down("w");
      const pending = settle(page, before.frame).then(() => ({ resolved: true }), error => ({ resolved: false, message: String(error) }));
      await page.waitForFunction(frame => { const s = window.box2Viewer, p = s.costProbe.snapshot(); return s.frames > frame && p.pendingRAF && p.orderingUpload.totals.orderingCalls > 3; }, before.frame, { timeout: 10000, polling: "raf" });
      const observed = await page.evaluate(() => window.box2Viewer.costProbe.snapshot());
      await page.close(); const cancelled = await pending;
      if (cancelled.resolved || !cancelled.message.includes("closed")) throw new Error("The actual original continuous-view wait was not cancelled by page close");
      result = { enabled, kind, completedViews, before, observed, cancelled, actualPageClosed: page.isClosed(), actualOriginalWInputStillHeldAtClose: true };
    }
  } finally { await context.close(); }
  result.actualContextClosed = true;
  save("LIFECYCLE_" + kind + "_" + Number(enabled) + ".json", result); return result;
}
(async () => {
  const gpuBefore = gpu(), browser = await chromium.launch({ headless: true, args: ["--use-angle=d3d11"] });
  try {
    let result;
    if (mode === "public") {
      const { context, page, errors } = await setup(browser, true);
      try {
        const pub = JSON.parse(fs.readFileSync(publicFile, "utf8")), views = JSON.parse(fs.readFileSync(path.join(PAGE, "camera_path.json"), "utf8")), metadata = JSON.parse(fs.readFileSync(path.join(PAGE, "display_manifest.json"), "utf8"));
        const values = await page.evaluate(async input => (await import("./orbit_convergence_generic.js")).runOrbitConvergenceGeneric(input), { views, publicViews: pub.poses, focus: metadata.focus });
        const snapshot = await page.evaluate(() => window.box2Viewer.costProbe.snapshot()), disposed = await closeAliases(page);
        result = { status: "complete_original_public_controller_component0", checks: { all_original_controller_component_checks: Object.values(values.checks).every(Boolean),
          full_original1016_public360_generic128_denominators: JSON.stringify(values.summaries.map(row => row.denominator)) === JSON.stringify([1016, 360, 128]),
          all_complete_pose_values_exact: values.summaries.every(row => row.exact === row.denominator),
          original_public_source_model_inputs_unchanged: pub.inputs.every(row => JSON.stringify(row) === JSON.stringify(identity(row.path))),
          normal_GL_aliases_drained_and_no_page_errors: disposed.timing.active === 0 && disposed.timing.orderingUpload.originalAliasRestored && !errors.length }, values, snapshot, disposed, errors,
          component_only_not_1504_Gaussian_GPU_views_or_GT: true };
      } finally { await context.close(); }
    } else {
      const cases = [];
      if (mode === "context") for (const enabled of [false, true]) cases.push(await contextCase(browser, enabled));
      else for (const kind of ["fault", "cancel"]) for (const enabled of [false, true]) cases.push(await lifecycleCase(browser, enabled, kind));
      result = { status: "actual_" + mode + "_cases_passed0", cases, checks: {
        all_owned_contexts_closed: cases.every(row => row.actualContextClosed),
        all_actual_context_cases_passed: mode !== "context" || cases.every(row => Object.values(row.checks).every(Boolean)),
        actual_two_GL_callback_errors_propagated_and_restored: mode !== "lifecycle" || cases.filter(row => row.kind === "fault").length === 2 && cases.filter(row => row.kind === "fault").every(row => row.faultAliasRestored && row.observed.fault.calls === 2),
        actual_two_page_closes_cancelled_real_W_pending_waits: mode !== "lifecycle" || cases.filter(row => row.kind === "cancel").length === 2 && cases.filter(row => row.kind === "cancel").every(row => row.actualPageClosed && !row.cancelled.resolved),
      } };
    }
    result.checks.original_assets_and_executed_sources_unchanged = [...inputs, ...sources].every(row => JSON.stringify(row) === JSON.stringify(identity(row.path)));
    result.sources = sources; result.inputs = inputs; result.gpuBefore = gpuBefore; result.gpuAfter = gpu(); result.new_optimization_count = 0;
    save("RESULT.json", result);
    console.log(JSON.stringify({ status: result.status, checks: result.checks }));
    if (!Object.values(result.checks).every(Boolean)) throw new Error("The original lifecycle or component checks failed");
  } finally { await browser.close(); }
})().catch(error => { save("FAILURE.json", { error: String(error), mode, sources, inputs, new_optimization_count: 0 }); console.error(error); process.exitCode = 1; });
