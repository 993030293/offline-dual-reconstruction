// Preserve staged recovery checks and use the freshly measured shared-prerequisite sources.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "run_ordering_prefix_lifecycle_v2.cjs"), generationFile = path.join(__dirname, "gl_resource_generation_v2.js");
function once(source, before, after) { if (source.split(before).length !== 2) throw new Error("The shared lifecycle prerequisite boundary changed: " + before.slice(0, 80)); return source.replace(before, after); }
let source = fs.readFileSync(original, "utf8");
const edits = [];
edits.push('source = once(source, "const sources = [__filename,", "const sources = [__filename, " + ' + JSON.stringify(JSON.stringify(original) + ", " + JSON.stringify(generationFile) + ", ") + ');');
edits.push('source = source.replaceAll("c0/", "c4/");');
edits.push('source = once(source, \'const body = relative.endsWith("/ordering_active_prefix.js") ?\', \'const body = relative.endsWith("/gl_resource_generation_v2.js") ? fs.readFileSync(\' + ' + JSON.stringify(JSON.stringify(generationFile)) + ' + \') : relative.endsWith("/ordering_active_prefix.js") ?\');');
edits.push('source = once(source, "disposed.timing.orderingUpload.originalAliasRestored && disposed.dag.nativeUpdateUnchanged", "disposed.timing.orderingUpload.originalAliasRestored && disposed.timing.generation.originalAliasesRestored && disposed.dag.nativeUpdateUnchanged");');
source = once(source, 'const instance = new Module(__filename, module);', edits.join("\n") + '\nconst instance = new Module(__filename, module);');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
