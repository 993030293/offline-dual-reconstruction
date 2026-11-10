// Bind the validation worker before the outer timing hook captures its owned native alias.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "run_ordering_scratch_complete.cjs");
function once(source, before, after) { if (source.split(before).length !== 2) throw new Error("The R164 worker ownership binding changed"); return source.replace(before, after); }
let source = fs.readFileSync(original, "utf8");
source = source.replaceAll("ordering_scratch_native.js", "ordering_scratch_native_v2.js");
const edit = 'helperText = once(helperText, "  function attachWorker() {", "  function attachWorker() {\\n    scratch.attachWorker();");';
source = once(source, "const viewerEdits = [", "const viewerEdits = [\n  " + JSON.stringify(edit) + ",");
source = once(source, 'const edits = [];', 'const edits = [];\nedits.push(' + JSON.stringify('source = once(source, "const sources = [__filename,", "const sources = [__filename, " + ' + JSON.stringify(JSON.stringify(original) + ", ") + ');') + ');');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
