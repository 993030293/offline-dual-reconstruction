// Diagnose stale native deletions without suppressing calls or consuming original GL errors.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "run_ordering_prefix_lifecycle_v2.cjs"), helper = path.join(__dirname, "gl_resource_generation.js");
function once(source, before, after) { if (source.split(before).length !== 2) throw new Error("The GL trace source boundary changed: " + before.slice(0, 70)); return source.replace(before, after); }
let source = fs.readFileSync(original, "utf8");
const edits = [];
edits.push('source = once(source, "const sources = [__filename,", "const sources = [__filename, " + ' + JSON.stringify(JSON.stringify(original) + ", " + JSON.stringify(helper) + ", ") + ');');
const importCode = 'probe = \'import { installGlResourceGeneration } from "./gl_resource_generation.js";\\n\' + probe;\n' +
  'probe = once(probe, "  const orderingCost = installOrderingActivePrefix", "  const generationTracker = installGlResourceGeneration({ renderer, enabled: false });\\n  const orderingCost = installOrderingActivePrefix");\n' +
  'probe = once(probe, "  state.orderingDiagnostics = {", "  state.glGeneration = generationTracker;\\n  state.orderingDiagnostics = {");\n' +
  'probe = once(probe, "orderingUpload: orderingCost.snapshot()", "orderingUpload: orderingCost.snapshot(), glGeneration: generationTracker.snapshot()");\n';
edits.push('source = once(source, \'if (mode === "lifecycle") {\', ' + JSON.stringify(importCode) + ' + \'if (mode === "lifecycle") {\');');
edits.push('source = once(source, \'const body = relative.endsWith("/ordering_active_prefix.js") ?\', \'const body = relative.endsWith("/gl_resource_generation.js") ? fs.readFileSync(\' + ' + JSON.stringify(JSON.stringify(helper)) + ' + \') : relative.endsWith("/ordering_active_prefix.js") ?\');');
edits.push('source = once(source, "timing: s.costProbe.dispose() };", "timing: s.costProbe.dispose(), generation: s.glGeneration.dispose() };");');
source = once(source, 'const instance = new Module(__filename, module);', edits.join("\n") + '\nconst instance = new Module(__filename, module);');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
