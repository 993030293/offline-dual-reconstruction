// Reuse the original complete four-step GPU diagnostic as an isolated candidate guard.
const fs = require("fs"), path = require("path"), crypto = require("crypto");
const CODE = __dirname;
const originalPath = path.join(CODE, "gpu_origin_dependency_diagnostic_run.cjs");
let source = fs.readFileSync(originalPath, "utf8");
function replace(before, after) {
  if (source.split(before).length !== 2) throw new Error("Original diagnostic runner anchor changed");
  source = source.replace(before, after);
}
replace('const sources = [__filename, path.join(CODE, "gpu_origin_dependency_diagnostic.mjs"),',
  'const sources = [__filename, path.join(CODE, "gpu_origin_dependency_smoke_002.cjs"), path.join(CODE, "gpu_origin_dependency_002.mjs"), path.join(CODE, "gpu_origin_dependency_diagnostic.mjs"),');
replace('if (/^\\/r118\\/(origin_dependency\\.mjs|origin_dependency_browser_005\\.mjs|gpu_origin_dependency_diagnostic\\.mjs)$/.test(pathname)) {',
  'if (/^\\/r118\\/(origin_dependency\\.mjs|frozen_origin_dependency\\.mjs|origin_dependency_browser_005\\.mjs|gpu_origin_dependency_diagnostic\\.mjs)$/.test(pathname)) {');
replace('response.end(fs.readFileSync(path.join(CODE, path.basename(pathname)))); return;',
  'response.end(fs.readFileSync(path.join(CODE, pathname.endsWith("/frozen_origin_dependency.mjs") ? "origin_dependency.mjs" : pathname.endsWith("/origin_dependency.mjs") ? "gpu_origin_dependency_002.mjs" : path.basename(pathname)))); return;');
replace('task: "R118_one_bounded_causal_diagnostic"', 'task: "R118_root_new_GPU_residency_algorithm_four_step_guard"');
replace('oldCandidateReadOnly: true, newAlgorithmFrozen: false', 'oldCandidateReadOnly: true, newAlgorithmFrozen: true');
replace('maximumNativeInterventions: 1', 'maximumNativeInterventions: 0');
replace('if (mode === "old_candidate" && name === "scene_real_context_restore_rotate" && trace.numericErrors.some',
  'if (false && mode === "old_candidate" && name === "scene_real_context_restore_rotate" && trace.numericErrors.some');
replace('"bounded_diagnostic_observed_failure" : "bounded_diagnostic_no_failure_reproduced"',
  '"new_GPU_dependency_required_guard_failed" : "new_GPU_dependency_four_step_guard_passed"');
replace('"Only after old-candidate nonzero restored-context GL guard, one original NativeSparkRenderer.updateInternal(autoUpdate=false), then exact full original capture comparison"',
  '"No intervention or error recovery is permitted for this new frozen algorithm guard"');
const compiled = new Function("require", "__filename", "__dirname", source);
compiled(require, originalPath, CODE);
