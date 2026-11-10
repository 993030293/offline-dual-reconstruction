// Keep the first preallocation observer failure and bind its successor after the native read pause.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "profile_ordering_allocation_cost_007.cjs");
function once(source, before, after) { if (source.split(before).length !== 2) throw new Error("The corrected allocation diagnostic source boundary changed"); return source.replace(before, after); }
let source = fs.readFileSync(original, "utf8");
source = source.replaceAll("ordering_allocation_cost.js", "ordering_allocation_cost_v2.js").replaceAll("installOrderingAllocationCost", "installOrderingAllocationCostV2").replaceAll("  const originalDriveSort = spark.driveSort;", "  const originalReadback = spark.readbackDepth;");
source = once(source, 'const edits = [];', 'const edits = [];\nedits.push(\'source = once(source, "const sources = [__filename,", "const sources = [__filename, " + ' + JSON.stringify(JSON.stringify(original) + ", ") + ');\');');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
