// Reuse the original complete preview diagnostic in symmetric private fresh contexts.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "profile_viewer_complete_cost.cjs");
const dagHelper = path.join(__dirname, "orbit_convergence_dag.js");
const genericHelper = path.join(__dirname, "orbit_convergence_generic.js");
const mode = process.argv[3] ?? "AB";
if (!["AB", "BA", "generic"].includes(mode)) throw new Error("The registered convergence diagnostic mode changed");
process.argv[3] = undefined;
function once(source, before, after) {
  if (source.split(before).length !== 2) throw new Error("The complete original diagnostic source boundary changed: " + before.slice(0, 80));
  return source.replace(before, after);
}
let source = fs.readFileSync(original, "utf8");
source = once(source, '"04_outputs/gaussian_pipeline_optimization_v3_20261005/viewer_cost_005"', '"04_outputs/gaussian_pipeline_optimization_v3_20261005/ocd001"');
source = once(source, 'const sources = [__filename, helper,', 'const dagHelper = ' + JSON.stringify(dagHelper) + ', genericHelper = ' + JSON.stringify(genericHelper) + ';\nconst sources = [__filename, ' + JSON.stringify(original) + ', dagHelper, genericHelper, helper,');
source = once(source, 'const helperText = fs.readFileSync(helper, "utf8"), nativeViewer', 'let helperText = fs.readFileSync(helper, "utf8"), nativeViewer');
source = once(source, 'let viewer =', 'helperText = once(helperText, "      count: [state.fullGaussianCount, state.boxGaussianCount],", "      quaternion: camera.quaternion.toArray(), world: camera.matrixWorld.toArray(),\\n      count: [state.fullGaussianCount, state.boxGaussianCount],");\nlet viewer =');
const injection = `
viewer = 'import { createOrbitConvergenceDag } from "./orbit_convergence_dag.js";\\n' + viewer;
viewer = once(viewer, "  const radius = Math.max", "  const useOrbitDag = new URLSearchParams(location.search).get('hooks') === '1';\\n  const orbitDag = createOrbitConvergenceDag({ controls, camera, state });\\n  state.orbitDag = orbitDag; state.orbitControls = controls; state.orbitCamera = camera;\\n  const radius = Math.max");
viewer = once(viewer, "  function applyView(index, fraction = 0) {", "  function applyView(index, fraction = 0) {\\n    if (useOrbitDag) orbitDag.markDiscrete(fraction);");
viewer = once(viewer, "      orbitChanged = controls.update();", "      orbitChanged = useOrbitDag ? orbitDag.update() : controls.update();");
viewer = once(viewer, "  if (new URLSearchParams(location.search).get('hooks') === '1') state.costProbe.enable();", "  state.costProbe.enable();");
`;
source = once(source, 'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });', injection + '\nfs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });\nfs.writeFileSync(path.join(output, "sources/EXECUTED_PROBE.js"), helperText, { flag: "wx" });');
source = once(source, '"http://127.0.0.1:28231"', '"http://127.0.0.1:28232"');
source = once(source, 'const body = relative.endsWith("/viewer_cost_probe.js") ?', 'const body = relative.endsWith("/orbit_convergence_dag.js") ? fs.readFileSync(dagHelper) : relative.endsWith("/orbit_convergence_generic.js") ? fs.readFileSync(genericHelper) : relative.endsWith("/viewer_cost_probe.js") ?');
source = once(source, 'const disposed = await page.evaluate(() => window.box2Viewer.costProbe.dispose());', 'const orbitDag = await page.evaluate(() => window.box2Viewer.orbitDag.dispose());\n    const disposed = await page.evaluate(() => window.box2Viewer.costProbe.dispose());\n    if (!orbitDag.nativeUpdateUnchanged || !orbitDag.disposed || orbitDag.pending) throw new Error("The owned native controller lifecycle changed");');
source = once(source, 'result = { hooks, completeColdToStableAndRGBA_ms,', 'result = { hooks, orbitDag, completeColdToStableAndRGBA_ms,');
source = once(source, 'registered_context_orders: ["original_no_hooks", "same_original_timing_hooks"], traceController,', 'registered_context_orders: ' + JSON.stringify(mode === "BA" ? ["candidate", "native"] : ["native", "candidate"]) + ', symmetric_original_cost_hooks: true, mode: ' + JSON.stringify(mode) + ', traceController,');
source = once(source, 'await episode(browser, false); await episode(browser, true);', mode === "BA" ? 'await episode(browser, true); await episode(browser, false);' : 'await episode(browser, false); await episode(browser, true);');
source = once(source, 'const left = episodes[0], right = episodes[1];', 'const left = episodes.find(e => !e.hooks), right = episodes.find(e => e.hooks);');
source = once(source, '["rgba_sha256", "position", "direction", "projection",', '["rgba_sha256", "position", "direction", "quaternion", "world", "projection",');
source = once(source, '"complete_preview_cost_diagnostic_finished0"', '"complete_fresh_orbit_dag_pair_exact_pending_acceptance0"');
source = once(source, 'source_only_hypothesis_no_candidate_or_formal: true', 'source_only_hypothesis_no_candidate_or_formal: false, independent_mechanism: "R162_original_discrete_orbit_CPU_convergence_before_GPU_generation"');
const genericFunction = `
async function runGeneric(browser) {
  const pubFile = path.join(OUT, "public_inputs001/RESULT.json"), pub = JSON.parse(fs.readFileSync(pubFile, "utf8"));
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
    const relative = decodeURIComponent(url.pathname.replace(/^\\/(native|profile)\\//, "")), filename = path.resolve(RELEASE, relative);
    if (!filename.startsWith(RELEASE + path.sep)) throw new Error("A private generic request escaped the original release");
    const body = relative.endsWith("/orbit_convergence_dag.js") ? fs.readFileSync(dagHelper) : relative.endsWith("/orbit_convergence_generic.js") ? fs.readFileSync(genericHelper) : relative.endsWith("/viewer_cost_probe.js") ? helperText : relative.endsWith("/viewer.js") ? viewer : fs.readFileSync(filename);
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
    save("RESULT.json", result);
    console.log(JSON.stringify({ checks: result.checks, summaries: result.summaries, routes: result.routes }));
    if (!Object.values(result.checks).every(Boolean)) throw new Error("The full original pose or generic contract failed");
    await page.evaluate(() => { window.box2Viewer.orbitDag.dispose(); window.box2Viewer.costProbe.dispose(); });
  } finally { await context.close(); }
}
`;
source = once(source, '(async () => {', genericFunction + '\n(async () => {');
if (mode === "generic") source = once(source, 'try {\n    await episode(browser, false);', 'try {\n    await runGeneric(browser); return;\n    await episode(browser, false);');
source = once(source, 'const started = performance.now();', 'const freeDisk = fs.statfsSync(ROOT, { bigint: true });\nif (freeDisk.bavail * freeDisk.bsize < 20n * 1024n ** 3n) throw new Error("The registered private-output disk reserve changed");\nconst started = performance.now();');
const compiled = source;
source = once(source, 'fs.mkdirSync(path.join(output, "sources"));', 'fs.mkdirSync(path.join(output, "sources"));\nfs.writeFileSync(path.join(output, "sources/COMPILED_DRIVER_BODY.cjs"), ' + JSON.stringify(compiled) + ', { flag: "wx" });');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
