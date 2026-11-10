// Retain the first bounded event-wait failure and use explicit frame and timeout event boundaries.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "run_gl_resource_generation_generic_v2.cjs");
function once(source, before, after) { if (source.split(before).length !== 2) throw new Error("The bounded GL event component boundary changed"); return source.replace(before, after); }
let source = fs.readFileSync(original, "utf8");
const transform = 'source = source.replaceAll("gl_resource_generation_generic_v2.js", "gl_resource_generation_generic_v3.js").replaceAll("runGlResourceGenerationGenericV2", "runGlResourceGenerationGenericV3");\n' +
  'source = once(source, "const sourceFiles = [__filename,", "const sourceFiles = [__filename, " + ' + JSON.stringify(JSON.stringify(original) + ", ") + ');\n';
source = once(source, 'const instance = new Module(__filename, module);', transform + 'const instance = new Module(__filename, module);');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
