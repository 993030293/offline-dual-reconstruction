// Verify the shared prerequisite in real WebGL without changing the frozen prefix components.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "run_ordering_prefix_generic.cjs");
function once(source, before, after) { if (source.split(before).length !== 2) throw new Error("The real GL generation component boundary changed: " + before.slice(0, 80)); return source.replace(before, after); }
let source = fs.readFileSync(original, "utf8");
source = source.replaceAll("ordering_active_prefix_generic.js", "gl_resource_generation_generic_v2.js").replaceAll("ordering_active_prefix.js", "gl_resource_generation_v2.js").replaceAll("runOrderingPrefixGeneric", "runGlResourceGenerationGenericV2");
source = once(source, 'const sourceFiles = [__filename,', 'const sourceFiles = [__filename, ' + JSON.stringify(original) + ',');
source = once(source, 'cases: result.cases.length', 'cases: result.rows.length');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
