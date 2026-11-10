// Reuse the frozen complete original allocation diagnostic without scratch reuse or replay oracles.
const fs = require("fs"), path = require("path"), Module = require("module");
const root = path.resolve(__dirname, "../..");
const donor = path.join(root, "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup007/p3/sources/COMPILED_DRIVER_BODY.cjs");
const pageHelper = path.join(__dirname, "sort32_bridge_page_observer.js");
const workerBlock = path.join(__dirname, "sort32_bridge_worker_observer.js");
const glHelper = path.join(__dirname, "gl_resource_generation_v2.js");
const sdkFile = path.join(root, "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release/shared-build/vendor/spark.module.js");
function once(source, before, after) {
  if (source.split(before).length !== 2) throw new Error("The original sort diagnostic boundary changed: " + before.slice(0, 100));
  return source.replace(before, after);
}
const sdk = fs.readFileSync(sdkFile, "utf8");
const literals = sdk.split(/\r?\n/).filter(line => line.startsWith("var jsContent = ") && line.endsWith(";"));
if (literals.length !== 1) throw new Error("The frozen original worker literal boundary changed");
const literal = literals[0].slice("var jsContent = ".length, -1);
if (/[\x00-\x08\x0a-\x1f]/.test(literal)) throw new Error("The original worker literal contains unsupported control characters");
const originalWorkerSource = JSON.parse(literal.replaceAll("\t", "\\t"));
const observedWorkerSource = once(originalWorkerSource, "initialize().catch(console.error);", fs.readFileSync(workerBlock, "utf8") + "\ninitialize().catch(console.error);");
const bridgeData = "export const originalWorkerSource = " + JSON.stringify(originalWorkerSource) + ";\nexport const observedWorkerSource = " + JSON.stringify(observedWorkerSource) + ";\n";
const nativeHTML = fs.readFileSync(path.join(path.dirname(sdkFile), "../../baseline/cases/packed/index.html"), "utf8");
const html = once(nativeHTML, '<script type="module" src="./viewer.js"></script>', '<script type="module">\nimport { installSort32BridgeObserver } from "./sort32_bridge_page_observer.js";\nimport { originalWorkerSource, observedWorkerSource } from "./sort32_bridge_data.js";\nwindow.sort32BridgeObserver = installSort32BridgeObserver({ originalWorkerSource, observedWorkerSource });\nawait import("./viewer.js");\n</script>');
let source = fs.readFileSync(donor, "utf8");
source = once(source, '"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup007"', '"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup008"');
const definitions = `const bridgePageHelper = ${JSON.stringify(pageHelper)}, bridgeWorkerBlock = ${JSON.stringify(workerBlock)}, originalDonor = ${JSON.stringify(donor)}, glHelper = ${JSON.stringify(glHelper)};\nconst bridgeData = ${JSON.stringify(bridgeData)}, privateHTML = ${JSON.stringify(html)}, originalWorkerSource = ${JSON.stringify(originalWorkerSource)}, observedWorkerSource = ${JSON.stringify(observedWorkerSource)};\n`;
source = once(source, "const sources = [__filename,", definitions + "const sources = [__filename, bridgePageHelper, bridgeWorkerBlock, originalDonor, glHelper,");
source = once(source, 'const prefix = "http://127.0.0.1:28232"', 'const prefix = "http://127.0.0.1:28238"');
const injection = `
viewer = 'import { installGlResourceGenerationV2 } from "./gl_resource_generation_v2.js";\\n' + viewer;
viewer = once(viewer, "  const costSpark = new SparkRenderer", "  state.glGeneration = installGlResourceGenerationV2({ renderer });\\n  const costSpark = new SparkRenderer");
fs.writeFileSync(path.join(output, "sources/ORIGINAL_WORKER.js"), originalWorkerSource, { flag: "wx" });
fs.writeFileSync(path.join(output, "sources/OBSERVED_WORKER.js"), observedWorkerSource, { flag: "wx" });
fs.writeFileSync(path.join(output, "sources/EXECUTED_INDEX.html"), privateHTML, { flag: "wx" });
fs.writeFileSync(path.join(output, "sources/EXECUTED_BRIDGE_DATA.js"), bridgeData, { flag: "wx" });
`;
source = once(source, 'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });', injection + '\nfs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });');
const routeAnchor = 'const body = relative.endsWith("/ordering_allocation_cost_v2.js") ?';
if (source.split(routeAnchor).length !== 3) throw new Error("The original two route boundaries changed");
source = source.replaceAll(routeAnchor, 'const body = relative.endsWith("/index.html") ? privateHTML : relative.endsWith("/sort32_bridge_page_observer.js") ? fs.readFileSync(bridgePageHelper) : relative.endsWith("/sort32_bridge_data.js") ? bridgeData : relative.endsWith("/gl_resource_generation_v2.js") ? fs.readFileSync(glHelper) : relative.endsWith("/ordering_allocation_cost_v2.js") ?');
source = once(source, 'const disposed = await page.evaluate(() => window.box2Viewer.costProbe.dispose());', 'const disposed = await page.evaluate(() => window.box2Viewer.costProbe.dispose());\n    const bridge = await page.evaluate(() => window.sort32BridgeObserver.dispose());\n    const glGeneration = await page.evaluate(() => window.box2Viewer.glGeneration.dispose());\n    if (!bridge.originalAliasesRestored || !glGeneration.originalAliasesRestored) throw new Error("The original bridge and GL aliases were not restored");');
source = once(source, 'result = { hooks, orbitDag,', 'result = { hooks, bridge, glGeneration, orbitDag,');
source = once(source, 'root_accepted_count: 33', 'root_accepted_count: 35');
source = once(source, '"complete_fresh_orbit_dag_pair_exact_pending_acceptance0"', '"complete_original_sort32_bridge_import_cost_diagnostic0"');
source = once(source, 'independent_mechanism: "R162_original_discrete_orbit_CPU_convergence_before_GPU_generation"', 'independent_mechanism: "original_sort32_WASM_import_copy_cost_diagnostic_only", observer_overhead_included_no_speed_claim: true');
source = once(source, 'const checks = { fixed_all21_view_results_model_camera_and_RGBA_exact:', 'const checks = { actual_sort32_import_traces_bound_to_original_RPC: right.bridge.totals.statusTraces > 0 && left.bridge.totals.statusTraces === 0 && right.bridge.traces.every(t => t.importCalls > 0 && !t.primaryError && t.resultReadbackSameView && t.resultOrderingSameView),\n      bridge_observer_globals_restored_and_native_gl_generation_common: episodes.every(e => e.bridge.originalAliasesRestored && e.glGeneration.originalAliasesRestored),\n      no_reference_replay_or_scratch_reuse: !viewer.includes("installOrderingScratch") && !helperText.includes("installOrderingScratch"),\n      original_initial_and_whole_model_populations: episodes.every(e => JSON.stringify(e.first.count) === "[526886,0]" && JSON.stringify(e.flight.after.count) === "[526886,60451]"),\n      fixed_all21_view_results_model_camera_and_RGBA_exact:');
const compiled = source;
source = once(source, 'fs.mkdirSync(path.join(output, "sources"));', 'fs.mkdirSync(path.join(output, "sources"));\nfs.writeFileSync(path.join(output, "sources/COMPILED_DRIVER_BODY.cjs"), ' + JSON.stringify(compiled) + ', { flag: "wx" });');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
