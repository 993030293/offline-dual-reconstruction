// Append shared prerequisite transformations after the frozen complete driver's own bindings.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "run_ordering_prefix_complete.cjs"), generationFile = path.join(__dirname, "gl_resource_generation_v2.js");
function once(source, before, after) { if (source.split(before).length !== 2) throw new Error("The shared complete prerequisite boundary changed: " + before.slice(0, 80)); return source.replace(before, after); }
let source = fs.readFileSync(original, "utf8");
const edits = [];
edits.push('source = once(source, "const sources = [__filename,", "const sources = [__filename, " + ' + JSON.stringify(JSON.stringify(original) + ", " + JSON.stringify(generationFile) + ", ") + ');');
edits.push('if (source.split(\'const body = relative.endsWith("/ordering_active_prefix.js") ?\').length !== 3) throw new Error("The shared generation route boundaries changed");');
edits.push('source = source.replaceAll(\'const body = relative.endsWith("/ordering_active_prefix.js") ?\', \'const body = relative.endsWith("/gl_resource_generation_v2.js") ? fs.readFileSync(\' + ' + JSON.stringify(JSON.stringify(generationFile)) + ' + \') : relative.endsWith("/ordering_active_prefix.js") ?\');');
const viewerEdits = [
  'helperText = \'import { installGlResourceGenerationV2 } from "./gl_resource_generation_v2.js";\\n\' + helperText;',
  'helperText = once(helperText, "  const orderingCost = installOrderingActivePrefix", "  const generationTracker = installGlResourceGenerationV2({ renderer });\\n  state.glGeneration = generationTracker;\\n  const orderingCost = installOrderingActivePrefix");',
  'helperText = once(helperText, "orderingUpload: orderingCost.snapshot()", "orderingUpload: orderingCost.snapshot(), glGeneration: generationTracker.snapshot()");',
  'helperText = once(helperText, "      const orderingUpload = orderingCost.dispose();", "      const orderingUpload = orderingCost.dispose();\\n      const generation = generationTracker.dispose();");',
  'helperText = once(helperText, "      return { active, disposed, orderingUpload, originalRender:", "      return { active, disposed, orderingUpload, generation, originalRender:");',
].join("\n") + "\n";
edits.push('source = once(source, \'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });\', ' + JSON.stringify(viewerEdits) + ' + \'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });\');');
const anchor = 'source = once(source, \'const compiled = source;\', edits.join("\\n") + \'\\nconst compiled = source;\');';
const registration = edits.map(edit => 'edits.push(' + JSON.stringify(edit) + ');').join("\n");
const transform = 'source = once(source, ' + JSON.stringify(anchor) + ', ' + JSON.stringify(registration + "\n" + anchor) + ');\n';
source = once(source, 'const instance = new Module(__filename, module);', transform + 'const instance = new Module(__filename, module);');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
