// Bind the original 118 full-byte cases to the frozen GPU-resident candidate.
const fs = require("fs"), path = require("path");
const CODE = __dirname, originalPath = path.join(CODE, "origin_dependency_run_003.cjs");
let source = fs.readFileSync(originalPath, "utf8");
function replace(before, after) {
  if (source.split(before).length !== 2) throw new Error("Original primary runner anchor changed");
  source = source.replace(before, after);
}
source = source.replaceAll("origin_dependency_browser_003.mjs", "origin_dependency_browser_005.mjs");
replace('const owned = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v3_20261005/origin_dependency_001");',
  'const owned = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v3_20261005/gpu_origin_dependency_001");');
replace('return [__filename, path.join(CODE, "origin_dependency.mjs"),',
  'return [__filename, path.join(CODE, "gpu_origin_dependency_primary.cjs"), path.join(CODE, "gpu_origin_dependency_002.mjs"), path.join(CODE, "origin_dependency.mjs"),');
replace('if (pathname === "/r115/origin_dependency.mjs" || pathname === "/r115/origin_dependency_browser_005.mjs") {',
  'if (pathname === "/r115/origin_dependency.mjs" || pathname === "/r115/origin_dependency_browser_005.mjs" || pathname === "/r118/frozen_origin_dependency.mjs") {');
replace('response.end(fs.readFileSync(path.join(CODE, path.basename(pathname)))); return;',
  'response.end(fs.readFileSync(path.join(CODE, pathname === "/r118/frozen_origin_dependency.mjs" ? "origin_dependency.mjs" : pathname === "/r115/origin_dependency.mjs" ? "gpu_origin_dependency_002.mjs" : path.basename(pathname)))); return;');
replace('mechanism: "R115_static_full_SH_radial_origin_dependency"', 'mechanism: "R118_GPU_residency_static_full_SH_radial_origin_dependency"');
replace('campaignInstrumentationSeparate: ["resource_monitor", "campaign_summary_readback", "contract_freeze_and_generic"],',
  'campaignInstrumentationSeparate: [], mandatoryFirstGenericAndNodeCloseFullyBilled: true,');
replace('steps.push({ name: mode + "_select", action: "select", index: mode });',
  'steps.push({ name: mode + "_select", action: "select", index: mode });\n' +
  '      steps.push({ name: mode + "_recovery_base", action: "rotation-base" });\n' +
  '      steps.push({ name: mode + "_recovery1", action: "context-restore-rotate", index: 6 });\n' +
  '      steps.push({ name: mode + "_recovery2", action: "context-restore-rotate", index: 7 });');
if (process.argv.includes("--source-preparation-only")) {
  new Function("require", "__filename", "__dirname", source);
  console.log(JSON.stringify({ sourceAnchorsPrepared: true, cases: 124, actualGPUOracleRun: false }));
} else {
  new Function("require", "__filename", "__dirname", source)(require, originalPath, CODE);
}
