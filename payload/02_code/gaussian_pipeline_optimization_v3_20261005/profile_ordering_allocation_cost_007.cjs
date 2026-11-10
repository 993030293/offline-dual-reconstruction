// Add read-only allocation-request attribution to two original complete R162 V2 workflows.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "profile_ordering_upload_cost_006.cjs"), allocationHelper = path.join(__dirname, "ordering_allocation_cost.js");
function once(source, before, after) { if (source.split(before).length !== 2) throw new Error("The native allocation diagnostic boundary changed: " + before.slice(0, 80)); return source.replace(before, after); }
let source = fs.readFileSync(original, "utf8");
source = once(source, '"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup006"', '"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup007"');
const edits = [];
edits.push('source = once(source, "const sources = [__filename,", "const sources = [__filename, " + ' + JSON.stringify(JSON.stringify(original) + ", " + JSON.stringify(allocationHelper) + ", ") + ');');
edits.push('if (source.split(\'const body = relative.endsWith("/ordering_upload_cost.js") ?\').length !== 3) throw new Error("The original allocation-observer route boundaries changed");');
edits.push('source = source.replaceAll(\'const body = relative.endsWith("/ordering_upload_cost.js") ?\', \'const body = relative.endsWith("/ordering_allocation_cost.js") ? fs.readFileSync(\' + ' + JSON.stringify(JSON.stringify(allocationHelper)) + ' + \') : relative.endsWith("/ordering_upload_cost.js") ?\');');
const viewerEdits = [
  'helperText = \'import { installOrderingAllocationCost } from "./ordering_allocation_cost.js";\\n\' + helperText;',
  'helperText = once(helperText, "  const originalDriveSort = spark.driveSort;", "  const allocationCost = installOrderingAllocationCost({ spark });\\n  const originalDriveSort = spark.driveSort;");',
  'helperText = once(helperText, "orderingUpload: orderingCost.snapshot()", "orderingUpload: orderingCost.snapshot(), orderingAllocation: allocationCost.snapshot()");',
  'helperText = once(helperText, "      const orderingUpload = orderingCost.dispose();", "      const orderingUpload = orderingCost.dispose();\\n      const orderingAllocation = allocationCost.dispose();");',
  'helperText = once(helperText, "      return { active, disposed, orderingUpload, originalRender:", "      return { active, disposed, orderingUpload, orderingAllocation, originalRender:");',
].join("\n") + "\n";
edits.push('source = once(source, \'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });\', ' + JSON.stringify(viewerEdits) + ' + \'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });\');');
const anchor = 'source = once(source, \'const compiled = source;\', edits.join("\\n") + \'\\nconst compiled = source;\');';
source = once(source, anchor, edits.map(edit => 'edits.push(' + JSON.stringify(edit) + ');').join("\n") + "\n" + anchor);
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
