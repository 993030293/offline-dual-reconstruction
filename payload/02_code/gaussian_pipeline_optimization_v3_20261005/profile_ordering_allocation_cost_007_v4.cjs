// Check final restoration against the native readback method after both observer layers are removed.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "profile_ordering_allocation_cost_007.cjs");
function once(source, before, after) { if (source.split(before).length !== 2) throw new Error("The corrected final alias diagnostic boundary changed"); return source.replace(before, after); }
let source = fs.readFileSync(original, "utf8");
source = source.replaceAll("ordering_allocation_cost.js", "ordering_allocation_cost_v2.js").replaceAll("installOrderingAllocationCost", "installOrderingAllocationCostV2").replaceAll("  const originalDriveSort = spark.driveSort;", "  const originalReadback = spark.readbackDepth;");
const edit = 'source = once(source, "const sources = [__filename,", "const sources = [__filename, " + ' + JSON.stringify(JSON.stringify(original) + ", ") + ');';
source = once(source, 'const edits = [];', 'const edits = [];\nedits.push(' + JSON.stringify(edit) + ');');
const finalAlias = 'helperText = once(helperText, "originalReadback: spark.readbackDepth === originalReadback", "originalReadback: orderingAllocation.originalAliasRestored");';
source = once(source, 'const viewerEdits = [', 'const viewerEdits = [\n  ' + JSON.stringify(finalAlias) + ',');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
