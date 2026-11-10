// Preserve V1 evidence while binding fresh diagnostics to the V2 intent lifecycle.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "run_orbit_convergence_dag.cjs");
function once(source, before, after) {
  if (source.split(before).length !== 2) throw new Error("The complete V2 intent source boundary changed");
  return source.replace(before, after);
}
let source = fs.readFileSync(original, "utf8");
source = once(source, '"orbit_convergence_dag.js"', '"orbit_convergence_dag_v2.js"');
const intentSetup = 'viewer = once(viewer, ' + JSON.stringify("  function stopPlayback() {") + ', ' + JSON.stringify("  function stopPlayback() {\n    if (useOrbitDag) orbitDag.cancelDiscrete();") + ');\n';
const modifications = `
source = once(source, 'const sources = [__filename,', 'const sources = [__filename, ' + ${JSON.stringify(JSON.stringify(original))} + ' ,');
source = once(source, 'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });', ${JSON.stringify(intentSetup)} + 'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });');
`;
source = once(source, 'const compiled = source;', modifications + '\nconst compiled = source;');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
