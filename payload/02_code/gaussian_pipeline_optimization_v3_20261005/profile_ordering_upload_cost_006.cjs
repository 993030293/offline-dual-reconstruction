// Bind the complete original diagnostic to symmetric R162 V2 and read-only GL counters.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "run_orbit_convergence_dag.cjs");
const orderingHelper = path.join(__dirname, "ordering_upload_cost.js");
function once(source, before, after) {
  if (source.split(before).length !== 2) throw new Error("The complete ordering-cost source boundary changed: " + before.slice(0, 80));
  return source.replace(before, after);
}
let source = fs.readFileSync(original, "utf8");
source = once(source, '"orbit_convergence_dag.js"', '"orbit_convergence_dag_v2.js"');
source = once(source, '"04_outputs/gaussian_pipeline_optimization_v3_20261005/ocd001"', '"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup006"');
const edits = [];
edits.push('source = once(source, "const sources = [__filename,", "const sources = [__filename, " + ' + JSON.stringify(JSON.stringify(original) + ", " + JSON.stringify(orderingHelper) + ", ") + ');');
edits.push('if (source.split(\'const body = relative.endsWith("/orbit_convergence_dag.js") ?\').length !== 3) throw new Error("The complete and generic private route boundaries changed");');
edits.push('source = source.replaceAll(\'const body = relative.endsWith("/orbit_convergence_dag.js") ?\', \'const body = relative.endsWith("/ordering_upload_cost.js") ? fs.readFileSync(\' + ' + JSON.stringify(JSON.stringify(orderingHelper)) + ' + \') : relative.endsWith("/orbit_convergence_dag.js") ?\');');
const viewerEdits = [
  'viewer = once(viewer, "  const useOrbitDag = new URLSearchParams(location.search).get(\'hooks\') === \'1\';", "  const useOrbitDag = true;");',
  'viewer = once(viewer, "  function stopPlayback() {", "  function stopPlayback() {\\n    if (useOrbitDag) orbitDag.cancelDiscrete();");',
  'helperText = \'import { installOrderingUploadCost } from "./ordering_upload_cost.js";\\n\' + helperText;',
  'helperText = once(helperText, "  const originalRender = renderer.render;", "  const orderingCost = installOrderingUploadCost({ renderer, spark });\\n  const originalRender = renderer.render;");',
  'helperText = once(helperText, "      frame: state.frames, pendingRAF: state.isRenderPending() };", "      frame: state.frames, pendingRAF: state.isRenderPending(), orderingUpload: orderingCost.snapshot() };");',
  'helperText = once(helperText, "      disposed = true;", "      const orderingUpload = orderingCost.dispose();\\n      disposed = true;");',
  'helperText = once(helperText, "      return { active, disposed, originalRender:", "      return { active, disposed, orderingUpload, originalRender:");',
].join("\n") + "\n";
edits.push('source = once(source, \'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });\', ' + JSON.stringify(viewerEdits) + ' + \'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });\');');
source = once(source, 'const compiled = source;', edits.join("\n") + '\nconst compiled = source;');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
