// Check complete immutable preflight and cancelled navigation intent with the V2 helper.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "run_orbit_convergence_dag.cjs");
process.argv[3] = "generic";
function once(source, before, after) {
  if (source.split(before).length !== 2) throw new Error("The complete V2 preflight source boundary changed");
  return source.replace(before, after);
}
let source = fs.readFileSync(original, "utf8");
source = once(source, '"orbit_convergence_dag.js"', '"orbit_convergence_dag_v2.js"');
const edges = `
  let intentEdgesExact = 0;
  for (const type of ["playing_fraction", "fraction", "fly_overwrite", "explicit_cancel", "play_pause_cancel"]) {
    const native = setup(generated[0]), candidate = setup(generated[0]), state = { navigationMode: "orbit", playing: false };
    try {
      const helper = createOrbitConvergenceDag({ ...candidate, state }); helper.markDiscrete();
      if (type === "playing_fraction") { state.playing = true; helper.markDiscrete(.25); state.playing = false; }
      if (type === "fraction") helper.markDiscrete(.25);
      if (type === "fly_overwrite") { state.navigationMode = "fly"; helper.markDiscrete(); state.navigationMode = "orbit"; }
      if (type === "explicit_cancel") helper.cancelDiscrete();
      if (type === "play_pause_cancel") { state.playing = true; helper.cancelDiscrete(); state.playing = false; }
      const pendingCleared = !helper.snapshot().pending;
      native.controls.update(); helper.update();
      const trace = helper.snapshot();
      const equal = pendingCleared && trace.shadowUpdates === 0 && trace.replayUpdates === 0 && trace.converged === 0 && complete(native.controls, native.camera) === complete(candidate.controls, candidate.camera);
      intentEdgesExact += Number(equal); if (!equal) failures.push({ intentEdge: type, trace }); helper.dispose();
    } finally { native.controls.dispose(); candidate.controls.dispose(); }
  }
`;
const transformations = [
  ["const summaries = [], failures = [], routes =", "let markerExact = 0, firstActualInputExact = 0;\n  const summaries = [], failures = [], routes ="],
  ["const helper = createOrbitConvergenceDag({ ...candidate, state }); helper.markDiscrete(); helper.update();",
    "const beforeMark = complete(candidate.controls, candidate.camera), nativeUpdate = candidate.controls.update;\n        let actualCalls = 0;\n        candidate.controls.update = function (...args) { if (this === candidate.controls && actualCalls++ === 0) firstActualInputExact += Number(complete(candidate.controls, candidate.camera) === beforeMark); return nativeUpdate.apply(this, args); };\n        const helper = createOrbitConvergenceDag({ ...candidate, state }); helper.markDiscrete();\n        markerExact += Number(complete(candidate.controls, candidate.camera) === beforeMark); helper.update();"],
  ["const restored = helper.dispose(); if (!restored.nativeUpdateUnchanged || restored.pending) throw new Error(\"The original generic controller alias changed\");",
    "const restored = helper.dispose(); if (!restored.nativeUpdateUnchanged || restored.pending) throw new Error(\"The original generic controller alias changed\");\n        candidate.controls.update = nativeUpdate;"],
  ["const helper = createOrbitConvergenceDag({ ...candidate, state }); helper.markDiscrete();\n      const before = complete(candidate.controls, candidate.camera);",
    "const before = complete(candidate.controls, candidate.camera);\n      const helper = createOrbitConvergenceDag({ ...candidate, state }); helper.markDiscrete();"],
  ["  return { status: failures.length ?", edges + "\n  return { status: failures.length ?"],
  ["checks: { all_full_pose_populations_exact:",
    "checks: { all5_cancelled_or_overwritten_intent_routes_original_single_update_exact: intentEdgesExact === 5, all1504_marker_inputs_unchanged: markerExact === 1504, all1504_original_inputs_unchanged_before_first_actual_update: firstActualInputExact === 1504, all_full_pose_populations_exact:"],
];
const injection = `
const supplementTransforms = ${JSON.stringify(transformations)};
let supplementModule = fs.readFileSync(genericHelper, "utf8");
for (const [before, after] of supplementTransforms) supplementModule = once(supplementModule, before, after);
fs.writeFileSync(path.join(output, "sources/EXECUTED_GENERIC_V2_SUPPLEMENT.js"), supplementModule, { flag: "wx" });
`;
const intentSetup = 'viewer = once(viewer, ' + JSON.stringify("  function stopPlayback() {") + ', ' + JSON.stringify("  function stopPlayback() {\n    if (useOrbitDag) orbitDag.cancelDiscrete();") + ');\n';
const outer = `
source = once(source, 'const sources = [__filename,', 'const sources = [__filename, ' + ${JSON.stringify(JSON.stringify(original))} + ' ,');
source = once(source, 'fs.writeFileSync(path.join(output, "sources/EXECUTED_PROBE.js"), helperText, { flag: "wx" });', 'fs.writeFileSync(path.join(output, "sources/EXECUTED_PROBE.js"), helperText, { flag: "wx" });\\n' + ${JSON.stringify(injection)});
source = source.replaceAll('relative.endsWith("/orbit_convergence_generic.js") ? fs.readFileSync(genericHelper)', 'relative.endsWith("/orbit_convergence_generic.js") ? supplementModule');
source = once(source, 'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });', ${JSON.stringify(intentSetup)} + 'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });');
`;
source = once(source, 'const compiled = source;', outer + '\nconst compiled = source;');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
