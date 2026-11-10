// Reuse the frozen full viewer workload and add preregistered fixed-origin look directions.
const fs = require("fs"), path = require("path"), Module = require("module");
const root = path.resolve(__dirname, "../..");
const donor = path.join(root, "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup007/p3/sources/COMPILED_DRIVER_BODY.cjs");
const observerHelper = path.join(__dirname, "radial_dependency_observer_012.js");
const glHelper = path.join(__dirname, "gl_resource_generation_v2.js");
function once(source, before, after) {
  if (source.split(before).length !== 2) throw new Error("The original radial diagnostic boundary changed: " + before.slice(0, 100));
  return source.replace(before, after);
}
let source = fs.readFileSync(donor, "utf8");
source = once(source, '"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup007"', '"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup012"');
source = once(source, "const sources = [__filename,", `const originalDonor = ${JSON.stringify(donor)}, radialHelper = ${JSON.stringify(observerHelper)}, glHelper = ${JSON.stringify(glHelper)};\nconst sources = [__filename, originalDonor, radialHelper, glHelper,`);
source = once(source, 'path.basename(item.path)));', 'item.path === originalDonor ? "DONOR_COMPILED_DRIVER_BODY.cjs" : path.basename(item.path)));');
source = once(source, 'const prefix = "http://127.0.0.1:28232"', 'const prefix = "http://127.0.0.1:28242"');
const injection = `
viewer = 'import { installGlResourceGenerationV2 } from "./gl_resource_generation_v2.js";\\nimport { installRadialDependencyObserver012 } from "./radial_dependency_observer_012.js";\\n' + viewer;
viewer = once(viewer, "  const costSpark = new SparkRenderer", "  state.glGeneration = installGlResourceGenerationV2({ renderer });\\n  const costSpark = new SparkRenderer");
viewer = once(viewer, "  const modelLoadStarted = performance.now();", "  if (new URLSearchParams(location.search).get('hooks') === '1') state.radialObserver = installRadialDependencyObserver012({ state, scene, camera, spark: costSpark, SplatMesh });\\n  const modelLoadStarted = performance.now();");
`;
source = once(source, 'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });', injection + '\nfs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });');
const routeAnchor = 'const body = relative.endsWith("/ordering_allocation_cost_v2.js") ?';
if (source.split(routeAnchor).length !== 3) throw new Error("The original two private route boundaries changed");
source = source.replaceAll(routeAnchor, 'const body = relative.endsWith("/radial_dependency_observer_012.js") ? fs.readFileSync(radialHelper) : relative.endsWith("/gl_resource_generation_v2.js") ? fs.readFileSync(glHelper) : relative.endsWith("/ordering_allocation_cost_v2.js") ?');
source = once(source, '  const before = await page.evaluate(() => ({ at: performance.now(), snapshot: window.box2Viewer.costProbe.snapshot() }));', '  await page.evaluate(name => window.box2Viewer.radialObserver?.setLabel(name), name);\n  const before = await page.evaluate(() => ({ at: performance.now(), snapshot: window.box2Viewer.costProbe.snapshot() }));');
const looks = `
      const originView = views.find(view => view.name === range + "_capture_0").capture;
      for (let k = 1; k <= 6; ++k) {
        const angle = k * Math.PI / 12, d = originView.direction;
        const direction = [Math.cos(angle) * d[0] + Math.sin(angle) * d[2], d[1], -Math.sin(angle) * d[0] + Math.cos(angle) * d[2]];
        const position = [...originView.position];
        views.push(await step(page, range + "_pure_look_" + k, () => page.evaluate(({ position, direction }) => window.box2Viewer.costProbe.freeCamera(position, direction), { position, direction })));
      }
`;
source = once(source, '    }\n    await page.getByLabel("移动方式").selectOption("fly");', looks + '    }\n    await page.getByLabel("移动方式").selectOption("fly");');
source = once(source, '    const flyBegin = performance.now(),', '    await page.evaluate(() => window.box2Viewer.radialObserver?.setLabel("continuous_W"));\n    const flyBegin = performance.now(),');
source = once(source, 'const disposed = await page.evaluate(() => window.box2Viewer.costProbe.dispose());', 'const disposed = await page.evaluate(() => window.box2Viewer.costProbe.dispose());\n    const radial = await page.evaluate(() => window.box2Viewer.radialObserver?.dispose() ?? null);\n    const glGeneration = await page.evaluate(() => window.box2Viewer.glGeneration.dispose());\n    if (radial) save("RADIAL_DEPENDENCY_TRACE.json", radial);\n    if (!glGeneration.originalAliasesRestored || radial && !radial.originalAliasesRestored) throw new Error("The original radial observer or common GL aliases were not restored");');
source = once(source, 'result = { hooks, orbitDag,', 'result = { hooks, radial: radial ? { totals: radial.totals, eventCount: radial.eventCount, originalAliasesRestored: radial.originalAliasesRestored, active: radial.active } : null, glGeneration, orbitDag,');
source = once(source, 'root_accepted_count: 33', 'root_accepted_count: 36');
source = once(source, '"complete_fresh_orbit_dag_pair_exact_pending_acceptance0"', '"original_complete_radial_dependency_cost_only0"');
source = once(source, 'independent_mechanism: "R162_original_discrete_orbit_CPU_convergence_before_GPU_generation"', 'independent_mechanism: "original_radial_dependency_source_cost_diagnostic_only", observer_overhead_included_no_speed_claim: true');
source = once(source, 'registered_context_orders: ["native","candidate"]', 'registered_context_orders: ["original","original_with_readonly_observer"], registered_extra_pure_looks: { angles_degrees: [15,30,45,60,75,90], origin_per_range: "original_capture_0", axis: "world_Y", all_original21_plus12: true }');
source = once(source, 'const gpuBefore = gpu(), browser = await chromium.launch({ headless: true, args: ["--use-angle=d3d11"] });', 'const gpuBefore = gpu(), browser = await chromium.launch({ headless: true, args: ["--use-angle=d3d11"] });\n  const gpuSamples = [gpuBefore], gpuFailures = [];\n  const gpuMonitor = setInterval(() => { try { gpuSamples.push(gpu()); } catch (error) { gpuFailures.push(String(error)); void browser.close(); } }, 2000);');
source = once(source, 'gpuBefore, gpuAfter: gpu(), episode_files:', 'gpuBefore, gpuAfter: gpu(), gpuSamples, gpuFailures, episode_files:');
source = once(source, '} finally { await browser.close(); }', '} finally { clearInterval(gpuMonitor); await browser.close(); save("GPU_RESOURCE_SAMPLES.json", { gpuSamples, gpuFailures }); }');
source = once(source, 'const checks = { fixed_all21_view_results_model_camera_and_RGBA_exact:', `const checks = {
      complete_original21_plus12_fixed_poses_both: episodes.every(e => e.views.length === 33 && e.views.filter(v => v.name.includes("_pure_look_")).length === 12),
      all_pure_looks_keep_original_capture0_exact_origin: episodes.every(e => ["scene","box"].every(range => {
        const origin = e.views.find(v => v.name === range + "_capture_0").capture.position;
        return e.views.filter(v => v.name.startsWith(range + "_pure_look_")).every(v => JSON.stringify(v.capture.position) === JSON.stringify(origin));
      })),
      observed_auto_update_and_native_generation_present: right.radial && right.radial.totals.automaticUpdates > 0 && right.radial.totals.generates > 0,
      all_original_radial_and_common_GL_aliases_restored: episodes.every(e => e.glGeneration.originalAliasesRestored && (!e.radial || e.radial.originalAliasesRestored && e.radial.active === 0)),
      no_scratch_reuse_or_new_radial_algorithm: !viewer.includes("installOrderingScratch") && !viewer.includes("installRadialCache"),
      original_complete_model_populations: episodes.every(e => JSON.stringify(e.first.count) === "[526886,0]" && JSON.stringify(e.flight.after.count) === "[526886,60451]"),
      fixed_all33_view_results_model_camera_and_RGBA_exact:`);
const compiled = source;
source = once(source, 'fs.mkdirSync(path.join(output, "sources"));', 'fs.mkdirSync(path.join(output, "sources"));\nfs.writeFileSync(path.join(output, "sources/COMPILED_DRIVER_BODY.cjs"), ' + JSON.stringify(compiled) + ', { flag: "wx" });');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
