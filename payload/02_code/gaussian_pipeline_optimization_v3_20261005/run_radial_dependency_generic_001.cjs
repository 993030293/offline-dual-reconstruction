// Run the original complete preregistered viewer pair with one private dependency change.
const fs = require("fs"), path = require("path"), Module = require("module");
const root = path.resolve(__dirname, "../..");
const donor = path.join(root, "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup012/p0/sources/COMPILED_DRIVER_BODY.cjs");
const candidateHelper = path.join(__dirname, "radial_translation_dependency_v3.js");
const license = path.join(root, "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release/shared-build/vendor/SPARK-LICENSE.txt");
function once(source, before, after) {
  if (source.split(before).length !== 2) throw new Error("The complete radial candidate boundary changed: " + before.slice(0, 100));
  return source.replace(before, after);
}
let source = fs.readFileSync(donor, "utf8");
source = once(source, '"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup012"', '"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup012/R167"');
source = once(source, "const sources = [__filename,", `const candidateHelper = ${JSON.stringify(candidateHelper)}, candidateDonor = ${JSON.stringify(donor)}, license = ${JSON.stringify(license)};\nconst sources = [__filename, candidateHelper, candidateDonor, license,`);
source = once(source, 'item.path === originalDonor ? "DONOR_COMPILED_DRIVER_BODY.cjs" : path.basename(item.path)', 'item.path === originalDonor ? "DONOR_007_COMPILED_DRIVER_BODY.cjs" : item.path === candidateDonor ? "DONOR_012_COMPILED_DRIVER_BODY.cjs" : path.basename(item.path)');
source = once(source, 'const prefix = "http://127.0.0.1:28242"', 'const prefix = "http://127.0.0.1:28243"');
source = once(source, 'import { installRadialDependencyObserver012 } from "./radial_dependency_observer_012.js";', 'import { installRadialTranslationDependency } from "./radial_translation_dependency_v3.js";');
source = once(source,
  "if (new URLSearchParams(location.search).get('hooks') === '1') state.radialObserver = installRadialDependencyObserver012({ state, scene, camera, spark: costSpark, SplatMesh });",
  "state.radialCache = installRadialTranslationDependency({ state, scene, camera, renderer, spark: costSpark, SplatMesh, glGeneration: state.glGeneration, enabled: new URLSearchParams(location.search).get('hooks') === '1' });");
const routeAnchor = 'const body = relative.endsWith("/radial_dependency_observer_012.js") ?';
if (source.split(routeAnchor).length !== 3) throw new Error("The two radial routes changed");
source = source.replaceAll(routeAnchor, 'const body = relative.endsWith("/radial_translation_dependency_v3.js") ? fs.readFileSync(candidateHelper) : relative.endsWith("/radial_dependency_observer_012.js") ?');
source = once(source, 'const disposed = await page.evaluate(() => window.box2Viewer.costProbe.dispose());', 'const radialCache = await page.evaluate(() => window.box2Viewer.radialCache.dispose());\n    const disposed = await page.evaluate(() => window.box2Viewer.costProbe.dispose());');
source = once(source, 'result = { hooks, radial:', 'result = { hooks, radialCache, radial:');
source = once(source, 'observed_auto_update_and_native_generation_present: right.radial && right.radial.totals.automaticUpdates > 0 && right.radial.totals.generates > 0,', 'native_dependency_helper_bound_and_drained: episodes.every(e => e.radialCache.originalAliasesRestored && e.radialCache.active === 0 && e.radialCache.fullOriginalMethodInverseExact),\n      actual_candidate_direction_and_unused_producer_changes_suppressed: right.radialCache.totals.eligibleDirectionChanges > 0 && right.radialCache.totals.suppressedUnusedProducerChanges > 0,');
source = once(source, 'no_scratch_reuse_or_new_radial_algorithm: !viewer.includes("installOrderingScratch") && !viewer.includes("installRadialCache"),', 'no_scratch_reuse_or_sort_arithmetic_change: !viewer.includes("installOrderingScratch") && episodes.every(e => e.radialCache.fullOriginalMethodInverseExact),');
source = once(source, '"original_complete_radial_dependency_cost_only0"', '"R167_first_complete_pair_pending_formal_and_lifecycle0"');
source = once(source, '"original_radial_dependency_source_cost_diagnostic_only", observer_overhead_included_no_speed_claim: true', '"R167_exact_radial_and_packed_SH_translation_dependency", first_pair_only_no_formal_acceptance: true');
source = once(source, 'registered_context_orders: ["original","original_with_readonly_observer"]', 'registered_context_orders: ["baseline","candidate"]');
source = once(source, 'const traceController = process.argv[3] === "--trace-controller";', 'const mode = process.argv[3] ?? "AB"; const traceController = false;');
source = once(source, 'if (process.argv[3] && !traceController) throw new Error("The diagnostic mode argument is unsupported");', 'if (!["AB", "BA"].includes(mode)) throw new Error("The preregistered R167 pair order is unsupported");');
source = once(source, 'mode: "AB", traceController,', 'mode, traceController,');
source = once(source, 'await episode(browser, false); await episode(browser, true);', 'for (const enabled of (mode === "BA" ? [true, false] : [false, true])) await episode(browser, enabled);');
source = once(source, 'const sources = [__filename,', 'const sources = [__filename, ' + JSON.stringify(path.join(__dirname, "run_radial_translation_dependency_001.cjs")) + ',');
source = once(source, "const sources = [__filename,", "const nativeGenericHelper = \"C:/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/radial_dependency_generic_001.js\";\nconst sources = [__filename, nativeGenericHelper,");
source = once(source, "fs.writeFileSync(path.join(output, \"sources/EXECUTED_VIEWER.js\"), viewer, { flag: \"wx\" });", "viewer = 'import { runRadialDependencyGeneric } from \"./radial_dependency_generic_001.js\";\\n' + viewer;\nviewer = once(viewer, \"  const modelLoadStarted = performance.now();\", \"  state.radialNativeGeneric = () => runRadialDependencyGeneric({ state, scene, camera, spark: costSpark, cache: state.radialCache });\\n  const modelLoadStarted = performance.now();\");\nfs.writeFileSync(path.join(output, \"sources/EXECUTED_VIEWER.js\"), viewer, { flag: \"wx\" });");
if (source.split("const body = relative.endsWith(\"/radial_translation_dependency_v3.js\") ?").length !== 3) throw new Error("The full generic routes changed");
source = source.replaceAll("const body = relative.endsWith(\"/radial_translation_dependency_v3.js\") ?", "const body = relative.endsWith(\"/radial_dependency_generic_001.js\") ? fs.readFileSync(nativeGenericHelper) : relative.endsWith(\"/radial_translation_dependency_v3.js\") ?");
source = once(source, "    const controllerTrace = traceController && hooks ?", "    await page.evaluate(origin => { window.box2Viewer.radialGenericOrigin = { position: origin.position, direction: origin.direction }; }, views.find(v => v.name === \"box_capture_0\").capture);\n    const nativeComponents = await page.evaluate(() => window.box2Viewer.radialNativeGeneric());\n    const controllerTrace = traceController && hooks ?");
source = once(source, "result = { hooks, radialCache,", "result = { hooks, nativeComponents, radialCache,");
source = once(source, "const checks = {", "const checks = {\n      all16_native_dependency_fixtures_and_captures_exact: [left,right].every(e => Object.values(e.nativeComponents.checks).every(Boolean)) && left.nativeComponents.fixtures.length === right.nativeComponents.fixtures.length && left.nativeComponents.fixtures.every((f,i) => { const g=right.nativeComponents.fixtures[i]; return f.name===g.name && [\"reset\",\"first\",\"second\"].every(tag => keys.every(k => JSON.stringify(f[tag][k])===JSON.stringify(g[tag][k]))); }),");
source = once(source, "const pubFile = path.join(OUT, \"public_inputs001/RESULT.json\"), pub =", "const pubFile = \"C:/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/public_inputs001/RESULT.json\", pub =");
source = once(source, "    save(\"RESULT.json\", result);", "    save(\"CONTROLLER_GENERIC_RESULT.json\", result);");
source = once(source, "await page.evaluate(() => { window.box2Viewer.orbitDag.dispose(); window.box2Viewer.costProbe.dispose(); });", "await page.evaluate(() => { window.box2Viewer.radialCache.dispose(); window.box2Viewer.orbitDag.dispose(); window.box2Viewer.costProbe.dispose(); window.box2Viewer.glGeneration.dispose(); });");
source = once(source, "for (const enabled of (mode === \"BA\" ? [true, false] : [false, true])) await episode(browser, enabled);", "for (const enabled of (mode === \"BA\" ? [true, false] : [false, true])) await episode(browser, enabled);\n    await runGeneric(browser);");
source = once(source, "\"R167_first_complete_pair_pending_formal_and_lifecycle0\"", "\"R167_actual_native_and_public_controller_components_pending_formal0\"");
const compiled = source;
source = once(source, 'fs.mkdirSync(path.join(output, "sources"));', 'fs.mkdirSync(path.join(output, "sources"));\nfs.writeFileSync(path.join(output, "sources/COMPILED_DRIVER_BODY.cjs"), ' + JSON.stringify(compiled) + ', { flag: "wx" });');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
