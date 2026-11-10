// Run the original complete preregistered viewer pair with one private dependency change.
const fs = require("fs"), path = require("path"), Module = require("module");
const root = path.resolve(__dirname, "../..");
const donor = path.join(root, "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup012/p0/sources/COMPILED_DRIVER_BODY.cjs");
const candidateHelper = path.join(__dirname, "radial_translation_dependency_001.js");
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
source = once(source, 'import { installRadialDependencyObserver012 } from "./radial_dependency_observer_012.js";', 'import { installRadialTranslationDependency } from "./radial_translation_dependency_001.js";');
source = once(source,
  "if (new URLSearchParams(location.search).get('hooks') === '1') state.radialObserver = installRadialDependencyObserver012({ state, scene, camera, spark: costSpark, SplatMesh });",
  "state.radialCache = installRadialTranslationDependency({ state, scene, camera, renderer, spark: costSpark, SplatMesh, glGeneration: state.glGeneration, enabled: new URLSearchParams(location.search).get('hooks') === '1' });");
const routeAnchor = 'const body = relative.endsWith("/radial_dependency_observer_012.js") ?';
if (source.split(routeAnchor).length !== 3) throw new Error("The two radial routes changed");
source = source.replaceAll(routeAnchor, 'const body = relative.endsWith("/radial_translation_dependency_001.js") ? fs.readFileSync(candidateHelper) : relative.endsWith("/radial_dependency_observer_012.js") ?');
source = once(source, 'const disposed = await page.evaluate(() => window.box2Viewer.costProbe.dispose());', 'const radialCache = await page.evaluate(() => window.box2Viewer.radialCache.dispose());\n    const disposed = await page.evaluate(() => window.box2Viewer.costProbe.dispose());');
source = once(source, 'result = { hooks, radial:', 'result = { hooks, radialCache, radial:');
source = once(source, 'observed_auto_update_and_native_generation_present: right.radial && right.radial.totals.automaticUpdates > 0 && right.radial.totals.generates > 0,', 'native_dependency_helper_bound_and_drained: episodes.every(e => e.radialCache.originalAliasesRestored && e.radialCache.active === 0 && e.radialCache.fullOriginalMethodInverseExact),\n      actual_candidate_direction_and_unused_producer_changes_suppressed: right.radialCache.totals.eligibleDirectionChanges > 0 && right.radialCache.totals.suppressedUnusedProducerChanges > 0,');
source = once(source, 'no_scratch_reuse_or_new_radial_algorithm: !viewer.includes("installOrderingScratch") && !viewer.includes("installRadialCache"),', 'no_scratch_reuse_or_sort_arithmetic_change: !viewer.includes("installOrderingScratch") && episodes.every(e => e.radialCache.fullOriginalMethodInverseExact),');
source = once(source, '"original_complete_radial_dependency_cost_only0"', '"R167_first_complete_pair_pending_formal_and_lifecycle0"');
source = once(source, '"original_radial_dependency_source_cost_diagnostic_only", observer_overhead_included_no_speed_claim: true', '"R167_exact_radial_and_packed_SH_translation_dependency", first_pair_only_no_formal_acceptance: true');
source = once(source, 'registered_context_orders: ["original","original_with_readonly_observer"]', 'registered_context_orders: ["baseline","candidate"]');
const compiled = source;
source = once(source, 'fs.mkdirSync(path.join(output, "sources"));', 'fs.mkdirSync(path.join(output, "sources"));\nfs.writeFileSync(path.join(output, "sources/COMPILED_DRIVER_BODY.cjs"), ' + JSON.stringify(compiled) + ', { flag: "wx" });');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
