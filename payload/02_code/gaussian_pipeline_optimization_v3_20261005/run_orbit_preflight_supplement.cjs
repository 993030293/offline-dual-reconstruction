// Verify the original inputs before marking and before the first actual replay.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "run_orbit_convergence_dag.cjs");
process.argv[3] = "generic";
function once(source, before, after) {
  if (source.split(before).length !== 2) throw new Error("The complete preflight supplement source boundary changed");
  return source.replace(before, after);
}
let source = fs.readFileSync(original, "utf8");
const transformations = [
  ["const summaries = [], failures = [], routes =", "let markerExact = 0, firstActualInputExact = 0;\n  const summaries = [], failures = [], routes ="],
  ["const helper = createOrbitConvergenceDag({ ...candidate, state }); helper.markDiscrete(); helper.update();",
    "const beforeMark = complete(candidate.controls, candidate.camera), nativeUpdate = candidate.controls.update;\n        let actualCalls = 0;\n        candidate.controls.update = function (...args) { if (this === candidate.controls && actualCalls++ === 0) firstActualInputExact += Number(complete(candidate.controls, candidate.camera) === beforeMark); return nativeUpdate.apply(this, args); };\n        const helper = createOrbitConvergenceDag({ ...candidate, state }); helper.markDiscrete();\n        markerExact += Number(complete(candidate.controls, candidate.camera) === beforeMark); helper.update();"],
  ["const restored = helper.dispose(); if (!restored.nativeUpdateUnchanged || restored.pending) throw new Error(\"The original generic controller alias changed\");",
    "const restored = helper.dispose(); if (!restored.nativeUpdateUnchanged || restored.pending) throw new Error(\"The original generic controller alias changed\");\n        candidate.controls.update = nativeUpdate;"],
  ["const helper = createOrbitConvergenceDag({ ...candidate, state }); helper.markDiscrete();\n      const before = complete(candidate.controls, candidate.camera);",
    "const before = complete(candidate.controls, candidate.camera);\n      const helper = createOrbitConvergenceDag({ ...candidate, state }); helper.markDiscrete();"],
  ["checks: { all_full_pose_populations_exact:",
    "checks: { all1504_marker_inputs_unchanged: markerExact === 1504, all1504_original_inputs_unchanged_before_first_actual_update: firstActualInputExact === 1504, all_full_pose_populations_exact:"],
];
// Transform only the served test module; keep all formal source files unchanged.
const injection = `
const supplementTransforms = ${JSON.stringify(transformations)};
const originalGenericModule = fs.readFileSync(genericHelper, "utf8");
let supplementModule = originalGenericModule;
for (const [before, after] of supplementTransforms) supplementModule = once(supplementModule, before, after);
fs.writeFileSync(path.join(output, "sources/EXECUTED_GENERIC_SUPPLEMENT.js"), supplementModule, { flag: "wx" });
`;
const outer = `
source = once(source, 'const sources = [__filename,', 'const sources = [__filename, ' + ${JSON.stringify(JSON.stringify(original))} + ' ,');
source = once(source, 'fs.writeFileSync(path.join(output, "sources/EXECUTED_PROBE.js"), helperText, { flag: "wx" });', 'fs.writeFileSync(path.join(output, "sources/EXECUTED_PROBE.js"), helperText, { flag: "wx" });\\n' + ${JSON.stringify(injection)});
source = source.replaceAll('relative.endsWith("/orbit_convergence_generic.js") ? fs.readFileSync(genericHelper)', 'relative.endsWith("/orbit_convergence_generic.js") ? supplementModule');
`;
source = once(source, 'const compiled = source;', outer + '\nconst compiled = source;');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
