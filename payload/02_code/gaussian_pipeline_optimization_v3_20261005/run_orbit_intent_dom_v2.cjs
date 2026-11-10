// Validate real navigation callbacks before paint and a new explicit fixed-view render.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "run_orbit_convergence_dag.cjs");
process.argv[3] = "AB";
function once(source, before, after) {
  if (source.split(before).length !== 2) throw new Error("The registered V2 DOM source boundary changed");
  return source.replace(before, after);
}
let source = fs.readFileSync(original, "utf8");
source = once(source, '"orbit_convergence_dag.js"', '"orbit_convergence_dag_v2.js"');
const intentSetup = 'viewer = once(viewer, ' + JSON.stringify("  function stopPlayback() {") + ', ' + JSON.stringify("  function stopPlayback() {\n    if (useOrbitDag) orbitDag.cancelDiscrete();") + ');\n' +
  'viewer = once(viewer, ' + JSON.stringify("  state.setView = index => { stopPlayback(); keys.clear(); applyView(index); };") + ', ' + JSON.stringify("  state.setView = index => { stopPlayback(); keys.clear(); applyView(index); };\n  state.applyOrbitFraction = applyView;") + ');\n';
const domFunction = `
async function domEpisode(browser, enabled) {
  const context = await browser.newContext({ viewport: { width: 960, height: 600 }, deviceScaleFactor: 1 }), page = await context.newPage(), errors = [], cases = [];
  page.on("pageerror", error => errors.push(String(error)));
  await context.route("**/*", async route => {
    const url = new URL(route.request().url()); if (url.origin !== prefix) throw new Error("A DOM diagnostic requested an external origin");
    const relative = decodeURIComponent(url.pathname.replace(/^\\/(native|profile)\\//, "")), filename = path.resolve(RELEASE, relative);
    if (!filename.startsWith(RELEASE + path.sep)) throw new Error("A DOM request escaped the frozen release");
    const body = relative.endsWith("/orbit_convergence_dag.js") ? fs.readFileSync(dagHelper) : relative.endsWith("/orbit_convergence_generic.js") ? fs.readFileSync(genericHelper) : relative.endsWith("/viewer_cost_probe.js") ? helperText : relative.endsWith("/viewer.js") ? viewer : fs.readFileSync(filename);
    const ext = path.extname(filename); await route.fulfill({ status: 200, body, contentType: ext === ".js" ? "text/javascript" : ext === ".json" ? "application/json" : ext === ".html" ? "text/html" : "application/octet-stream" });
  });
  try {
    await page.goto(prefix + "/" + (enabled ? "profile" : "native") + "/baseline/cases/packed/index.html?hooks=" + Number(enabled), { waitUntil: "domcontentloaded" }); await settle(page);
    for (const name of ["play_pause", "fly_orbit", "controller_start", "fractional_view"]) {
      await page.evaluate(() => window.box2Viewer.setView(0)); await settle(page);
      const transition = await page.evaluate(name => {
        const s = window.box2Viewer, before = s.orbitDag.snapshot(); s.setView(203);
        if (name === "play_pause") { document.getElementById("play").click(); document.getElementById("play").click(); }
        if (name === "fly_orbit") {
          const select = document.getElementById("navigation-mode");
          for (const mode of ["fly", "orbit"]) { select.value = mode; select.dispatchEvent(new Event("change", { bubbles: true })); }
        }
        if (name === "controller_start") s.orbitControls.dispatchEvent({ type: "start" });
        if (name === "fractional_view") s.applyOrbitFraction(0, .25);
        const after = s.orbitDag.snapshot(), immediate = s.costProbe.snapshot();
        const cleared = !after.pending && after.shadowUpdates === before.shadowUpdates && after.replayUpdates === before.replayUpdates;
        s.setView(0);
        return { name, before, after, immediate, cleared, new_explicit_discrete_request_after_record: true };
      }, name);
      if (!transition.cleared) throw new Error("A cancelled original navigation intent reached the next RAF");
      await settle(page); cases.push({ ...transition, finalCapture: await capture(page) });
    }
    const disposed = await page.evaluate(() => ({ dag: window.box2Viewer.orbitDag.dispose(), timing: window.box2Viewer.costProbe.dispose() }));
    if (errors.length || disposed.timing.active || !disposed.dag.nativeUpdateUnchanged) throw new Error("The actual DOM callbacks or native aliases failed");
    return { enabled, cases, errors, disposed };
  } finally { await context.close(); }
}
async function runDomCases(browser) {
  const native = await domEpisode(browser, false), candidate = await domEpisode(browser, true);
  const keys = ["position", "direction", "quaternion", "world", "projection", "dimensions", "count", "mode", "activeSplats", "currentSplats"];
  const exact = native.cases.every((left, i) => {
    const right = candidate.cases[i]; return left.name === right.name && keys.every(k => JSON.stringify(left.immediate[k]) === JSON.stringify(right.immediate[k])) && [...keys, "rgba_sha256"].every(k => JSON.stringify(left.finalCapture[k]) === JSON.stringify(right.finalCapture[k]));
  });
  const checks = { all4_real_original_callback_intents_cleared_before_next_RAF: [...native.cases, ...candidate.cases].every(x => x.cleared),
    immediate_full_camera_and_new_explicit_fixed_render_RGBA_exact: exact, original_assets_and_vendor_unchanged: JSON.stringify(inputs) === JSON.stringify(inputs.map(row => identity(row.path))),
    actual_sources_unchanged: JSON.stringify(sources) === JSON.stringify(sources.map(row => identity(row.path))),
    native_aliases_drained_and_restored: [native,candidate].every(x => !x.disposed.timing.active && x.disposed.dag.nativeUpdateUnchanged),
    all_original_GL_and_page_errors_empty: [native,candidate].every(x => !x.errors.length && x.cases.every(c => c.finalCapture.glError === 0)) };
  save("RESULT.json", { status: Object.values(checks).every(Boolean) ? "actual_V2_navigation_callbacks_and_fixed_recovery_exact0" : "failed_V2_navigation_callbacks0", checks, native, candidate, sources, inputs, new_optimization_count: 0 });
  console.log(JSON.stringify({ checks, new_optimization_count: 0 }));
  if (!Object.values(checks).every(Boolean)) throw new Error("The V2 real navigation contract failed");
}
`;
const modifications = `
source = once(source, 'const sources = [__filename,', 'const sources = [__filename, ' + ${JSON.stringify(JSON.stringify(original))} + ' ,');
source = once(source, 'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });', ${JSON.stringify(intentSetup)} + 'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });');
source = once(source, '(async () => {', ${JSON.stringify(domFunction)} + '\\n(async () => {');
source = once(source, 'try {\\n    await episode(browser, false);', 'try {\\n    await runDomCases(browser); return;\\n    await episode(browser, false);');
`;
source = once(source, 'const compiled = source;', modifications + '\nconst compiled = source;');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
