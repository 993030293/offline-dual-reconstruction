// Exercise actual controller failures and page cancellation before fresh complete recovery.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "run_orbit_convergence_dag.cjs");
process.argv[3] = "AB";
function once(source, before, after) {
  if (source.split(before).length !== 2) throw new Error("The complete convergence recovery source boundary changed");
  return source.replace(before, after);
}
let source = fs.readFileSync(original, "utf8");
const setupFault = `
viewer = once(viewer, "  const orbitDag = createOrbitConvergenceDag({ controls, camera, state });", "  const originalFaultUpdate = controls.update;\\n  const faultUpdate = function (...args) { if (this === controls && state.orbitFault?.armed) { state.orbitFault.calls++; if (state.orbitFault.calls === 2) throw new Error('registered actual original-controller failure'); } return originalFaultUpdate.apply(this, args); };\\n  controls.update = faultUpdate;\\n  state.restoreOrbitFault = () => { if (controls.update !== faultUpdate) throw new Error('The owned fault alias changed'); controls.update = originalFaultUpdate; return controls.update === originalFaultUpdate; };\\n  const orbitDag = createOrbitConvergenceDag({ controls, camera, state });");
`;
const lifecycleFunction = `
async function lifecycleCase(browser, enabled, kind) {
  const context = await browser.newContext({ viewport: { width: 960, height: 600 }, deviceScaleFactor: 1 }), page = await context.newPage();
  const errors = [], completedViews = [];
  page.on("pageerror", error => errors.push(String(error)));
  await context.route("**/*", async route => {
    const url = new URL(route.request().url());
    if (url.origin !== prefix) throw new Error("A lifecycle diagnostic requested an external origin");
    const relative = decodeURIComponent(url.pathname.replace(/^\\/(native|profile)\\//, "")), filename = path.resolve(RELEASE, relative);
    if (!filename.startsWith(RELEASE + path.sep)) throw new Error("A lifecycle request escaped the original release");
    const body = relative.endsWith("/orbit_convergence_dag.js") ? fs.readFileSync(dagHelper) : relative.endsWith("/orbit_convergence_generic.js") ? fs.readFileSync(genericHelper) : relative.endsWith("/viewer_cost_probe.js") ? helperText : relative.endsWith("/viewer.js") ? viewer : fs.readFileSync(filename);
    const extension = path.extname(filename);
    await route.fulfill({ status: 200, body, contentType: extension === ".js" ? "text/javascript" : extension === ".json" ? "application/json" : extension === ".html" ? "text/html" : "application/octet-stream" });
  });
  let evidence;
  try {
    await page.goto(prefix + "/" + (enabled ? "profile" : "native") + "/baseline/cases/packed/index.html?hooks=" + Number(enabled), { waitUntil: "domcontentloaded" });
    await settle(page);
    for (const index of [0, 203]) completedViews.push(await step(page, "prior_view_" + index, () => page.evaluate(index => window.box2Viewer.setView(index), index)));
    const before = await page.evaluate(() => ({ frame: window.box2Viewer.frames, snapshot: window.box2Viewer.costProbe.snapshot() }));
    if (kind === "fault") {
      await page.evaluate(() => { const s = window.box2Viewer; s.orbitFault = { armed: true, calls: 0 }; s.setView(406); });
      await page.waitForFunction(() => window.box2Viewer.errors.some(value => value.includes("registered actual original-controller failure")), undefined, { timeout: 30000, polling: "raf" });
      const observed = await page.evaluate(() => ({ originalUiErrors: window.box2Viewer.errors.slice(),
        fault: { ...window.box2Viewer.orbitFault }, snapshot: window.box2Viewer.costProbe.snapshot(), dag: window.box2Viewer.orbitDag.snapshot() }));
      if (observed.fault.calls < 2 || !errors.some(value => value.includes("registered actual original-controller failure"))) throw new Error("The actual original-controller exception did not reach both the browser and original UI");
      await page.evaluate(() => { window.box2Viewer.orbitFault.armed = false; });
      await settle(page, before.frame);
      const restored = await page.evaluate(() => { const s = window.box2Viewer, dag = s.orbitDag.dispose(), timing = s.costProbe.dispose();
        const nativeFaultAliasRestored = s.restoreOrbitFault(); return { dag, timing, nativeFaultAliasRestored }; });
      if (!restored.nativeFaultAliasRestored || restored.timing.active !== 0 || !restored.dag.nativeUpdateUnchanged) throw new Error("The original fault consumer did not drain and restore its owned aliases");
      evidence = { enabled, kind, completedViews, before, observed, restored, original_error_propagated: true };
    } else {
      await page.evaluate(() => window.box2Viewer.setView(406));
      const pending = settle(page, before.frame).then(() => ({ resolved: true }), error => ({ resolved: false, message: String(error) }));
      await page.waitForFunction(frame => { const s = window.box2Viewer, p = s.costProbe.snapshot(); return s.currentView === 406 && s.frames > frame && (p.sorting || p.sortDirty || p.pendingRAF); }, before.frame, { timeout: 30000, polling: "raf" });
      const observed = await page.evaluate(() => ({ currentView: window.box2Viewer.currentView, snapshot: window.box2Viewer.costProbe.snapshot(), dag: window.box2Viewer.orbitDag.snapshot() }));
      await page.close();
      const originalPendingWait = await pending;
      if (originalPendingWait.resolved || !originalPendingWait.message.includes("closed")) throw new Error("The real page close did not cancel an unfinished complete-view wait");
      evidence = { enabled, kind, completedViews, before, observed, originalPendingWait, actualPageClosed: page.isClosed(), original_error_propagated: true };
    }
  } finally { await context.close(); }
  evidence.actualContextClosed = true;
  save("LIFECYCLE_" + kind + "_" + Number(enabled) + ".json", evidence);
  console.log(JSON.stringify({ lifecycle: kind, candidate: enabled, completedViews: completedViews.length, actualContextClosed: true }));
  return evidence;
}
`;
const modifications = `
source = once(source, 'const sources = [__filename,', 'const sources = [__filename, ' + ${JSON.stringify(JSON.stringify(original))} + ' ,');
source = once(source, 'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });', ${JSON.stringify(setupFault)} + '\\nfs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });');
source = once(source, '(async () => {', ${JSON.stringify(lifecycleFunction)} + '\\n(async () => {');
source = once(source, 'try {\\n    await episode(browser, false);', 'try {\\n    const recoveryEvidence = [];\\n    for (const kind of ["fault", "cancel"]) for (const enabled of [false, true]) recoveryEvidence.push(await lifecycleCase(browser, enabled, kind));\\n    save("RECOVERY_EVENTS.json", { recoveryEvidence, new_optimization_count: 0 });\\n    await episode(browser, false);');
source = once(source, 'const checks = { fixed_all21_view_results_model_camera_and_RGBA_exact:', 'const checks = { actual_original_callback_errors_propagated_and_restored: recoveryEvidence.filter(e => e.kind === "fault").length === 2 && recoveryEvidence.filter(e => e.kind === "fault").every(e => e.original_error_propagated && e.restored.nativeFaultAliasRestored && e.restored.timing.active === 0), actual_page_close_cancelled_unfinished_waits: recoveryEvidence.filter(e => e.kind === "cancel").length === 2 && recoveryEvidence.filter(e => e.kind === "cancel").every(e => e.actualPageClosed && !e.originalPendingWait.resolved && e.observed.snapshot.frame > e.before.frame), all_owned_lifecycle_contexts_closed: recoveryEvidence.every(e => e.actualContextClosed), fresh_complete21_view_recovery_both_routes: episodes.length === 2 && episodes.every(e => e.views.length === 21), fixed_all21_view_results_model_camera_and_RGBA_exact:');
source = once(source, '"complete_fresh_orbit_dag_pair_exact_pending_acceptance0"', '"actual_original_fault_page_cancel_and_fresh_complete_recovery_passed0"');
`;
source = once(source, 'const compiled = source;', modifications + '\nconst compiled = source;');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
