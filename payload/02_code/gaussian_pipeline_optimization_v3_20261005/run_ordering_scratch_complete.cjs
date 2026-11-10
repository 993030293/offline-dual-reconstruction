// Vary allocation/recycling only, with the original full preview, native dependencies and symmetric validation.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "profile_ordering_upload_cost_006.cjs");
const scratchFiles = ["ordering_scratch_native.js", "ordering_scratch_pool.js", "gl_resource_generation_v2.js"];
function once(source, before, after) { if (source.split(before).length !== 2) throw new Error("The R164 complete binding changed: " + before.slice(0, 80)); return source.replace(before, after); }
let source = fs.readFileSync(original, "utf8");
source = once(source, '"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup006"', '"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup007/R164"');
const edits = [];
edits.push('source = once(source, "const sources = [__filename,", "const sources = [__filename, " + ' + JSON.stringify(JSON.stringify(original) + ", " + scratchFiles.map(n => JSON.stringify(path.join(__dirname, n))).join(", ") + ", ") + ');');
edits.push('if (source.split(\'const body = relative.endsWith("/ordering_upload_cost.js") ?\').length !== 3) throw new Error("The complete R164 private route boundaries changed");');
edits.push('source = source.replaceAll(\'const body = relative.endsWith("/ordering_upload_cost.js") ?\', ' + JSON.stringify('const body = ' + scratchFiles.map(n => 'relative.endsWith("/' + n + '") ? fs.readFileSync(' + JSON.stringify(path.join(__dirname, n)) + ') : ').join('') + 'relative.endsWith("/ordering_upload_cost.js") ?') + ');');
const viewerEdits = [
  'helperText = \'import { installOrderingScratchNative } from "./ordering_scratch_native.js";\\nimport { installGlResourceGenerationV2 } from "./gl_resource_generation_v2.js";\\n\' + helperText;',
  'helperText = once(helperText, "  const orderingCost = installOrderingUploadCost", "  const generationTracker = installGlResourceGenerationV2({ renderer });\\n  const scratch = installOrderingScratchNative({ spark, enabled: new URLSearchParams(location.search).get(\'hooks\') === \'1\' });\\n  state.orderingScratch = scratch; state.glGeneration = generationTracker;\\n  const orderingCost = installOrderingUploadCost");',
  'helperText = once(helperText, "orderingUpload: orderingCost.snapshot()", "orderingUpload: orderingCost.snapshot(), orderingScratch: scratch.snapshot(), glGeneration: generationTracker.snapshot()");',
  'helperText = once(helperText, "      const orderingUpload = orderingCost.dispose();", "      const orderingUpload = orderingCost.dispose();\\n      const orderingScratch = scratch.dispose();\\n      const generation = generationTracker.dispose();");',
  'helperText = once(helperText, "      return { active, disposed, orderingUpload, originalRender:", "      return { active, disposed, orderingUpload, orderingScratch, generation, originalRender:");',
  'helperText = once(helperText, "originalReadback: spark.readbackDepth === originalReadback", "originalReadback: orderingScratch.originalReadbackAliasRestored");',
  'helperText = once(helperText, "originalDriveSort: spark.driveSort === originalDriveSort", "originalDriveSort: orderingScratch.originalSortAliasRestored");',
].join("\n") + "\n";
edits.push('source = once(source, \'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });\', ' + JSON.stringify(viewerEdits) + ' + \'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });\');');
edits.push('source = once(source, "R162_original_discrete_orbit_CPU_convergence_before_GPU_generation", "R164_exact_capacity_full_zero_ordering_scratch_reuse_original_image_data_protected_R162_V2_shared_generation_both");');
edits.push('source = once(source, \'fs.mkdirSync(path.join(output, "sources"));\', \'fs.mkdirSync(path.join(output, "sources"));\\nfs.copyFileSync(path.join(RELEASE, "shared-build/vendor/SPARK-LICENSE.txt"), path.join(output, "sources/SPARK-LICENSE.txt"));\');');
const anchor = 'source = once(source, \'const compiled = source;\', edits.join("\\n") + \'\\nconst compiled = source;\');';
source = once(source, anchor, edits.map(edit => 'edits.push(' + JSON.stringify(edit) + ');').join("\n") + "\n" + anchor);
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
