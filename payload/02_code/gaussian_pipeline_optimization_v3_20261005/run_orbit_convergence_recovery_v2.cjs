// Repeat actual original faults, page cancellation and full recovery with V2 intents.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "run_orbit_convergence_recovery.cjs");
function once(source, before, after) {
  if (source.split(before).length !== 2) throw new Error("The complete V2 recovery source boundary changed");
  return source.replace(before, after);
}
let source = fs.readFileSync(original, "utf8");
source = once(source, 'let source = fs.readFileSync(original, "utf8");', 'let source = fs.readFileSync(original, "utf8");\nsource = once(source, \'"orbit_convergence_dag.js"\', \'"orbit_convergence_dag_v2.js"\');');
const intentSetup = 'viewer = once(viewer, ' + JSON.stringify("  function stopPlayback() {") + ', ' + JSON.stringify("  function stopPlayback() {\n    if (useOrbitDag) orbitDag.cancelDiscrete();") + ');\n';
source = once(source, 'const setupFault = `', 'const setupFault = ' + JSON.stringify(intentSetup) + ' + `');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
