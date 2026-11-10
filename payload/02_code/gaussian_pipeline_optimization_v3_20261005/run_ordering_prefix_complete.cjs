// Preserve the original complete R162 V2 workflow and vary only the GPU upload prefix helper.
const fs = require("fs"), path = require("path"), Module = require("module");
const originalCostDriver = path.join(__dirname, "profile_ordering_upload_cost_006.cjs");
function once(source, before, after) {
  if (source.split(before).length !== 2) throw new Error("The complete prefix binding changed: " + before.slice(0, 80));
  return source.replace(before, after);
}
let source = fs.readFileSync(originalCostDriver, "utf8");
if (source.split("ordering_upload_cost.js").length !== 4 || source.split("installOrderingUploadCost").length !== 3) throw new Error("The measured upload helper source boundaries changed");
source = source.replaceAll("ordering_upload_cost.js", "ordering_active_prefix.js").replaceAll("installOrderingUploadCost", "installOrderingActivePrefix");
source = once(source, '"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup006"', '"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup006/R163"');
source = once(source, 'JSON.stringify(original) + ", " + JSON.stringify(orderingHelper)', 'JSON.stringify(' + JSON.stringify(originalCostDriver) + ') + ", " + JSON.stringify(original) + ", " + JSON.stringify(orderingHelper)');
source = once(source, "installOrderingActivePrefix({ renderer, spark });", "installOrderingActivePrefix({ renderer, spark, enabled: state.prefixEnabled });");
const setup = 'viewer = once(viewer, "  const useOrbitDag = true;", ' + JSON.stringify("  const useOrbitDag = true;\n  state.prefixEnabled = new URLSearchParams(location.search).get('hooks') === '1';") + ');';
source = once(source, '  \'viewer = once(viewer, "  function stopPlayback() {",', '  ' + JSON.stringify(setup) + ',\n  \'viewer = once(viewer, "  function stopPlayback() {",');
const mechanismEdit = 'edits.push(' + JSON.stringify('source = once(source, "R162_original_discrete_orbit_CPU_convergence_before_GPU_generation", "R163_readable_ordering_prefix_GPU_submission_unchanged_sort_storage_R162_V2_both_routes");') + ');\n';
source = once(source, 'const viewerEdits = [', mechanismEdit + 'const viewerEdits = [');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
