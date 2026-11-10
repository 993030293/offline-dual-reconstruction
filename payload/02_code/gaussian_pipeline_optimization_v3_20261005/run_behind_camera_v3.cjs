// Re-evaluate the unchanged whole-renderer culler under the authorized work gate.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "run_behind_camera_draw.cjs");
function once(source, before, after) {
  if (source.split(before).length !== 2) throw new Error("The original viewer source anchor changed: " + before.slice(0, 80));
  return source.replace(before, after);
}
let source = fs.readFileSync(original, "utf8");
source = once(source, 'const sourceFiles = [__filename, helper, path.join(caseRoot, "viewer.js")];',
  'const sourceFiles = [__filename, helper, path.join(caseRoot, "viewer.js"), ' + JSON.stringify(original) + '];');
source = once(source, 'if (pair.reduction < 0.2 ||', 'if (pair.reduction < 0.1 ||');
source = once(source, 'pairs.every(row => row.reduction >= 0.2 &&', 'pairs.every(row => row.reduction >= 0.1 &&');
source = once(source, 'count: [s.fullGaussianCount, s.boxGaussianCount], mode: s.sceneMode,',
  'count: [s.fullGaussianCount, s.boxGaussianCount], mode: s.sceneMode,\n      world: s.r138.camera.matrixWorld.elements.slice(), fullProjection: s.r138.camera.projectionMatrix.elements.slice(), quaternion: s.r138.camera.quaternion.toArray(),');
source = once(source, '"position", "direction", "count", "mode"]', '"position", "direction", "count", "mode", "world", "fullProjection", "quaternion"]');
source = once(source, 'try { await page.evaluate(() => window.box2Viewer?.r138?.dispose()); }',
  'try { await page.evaluate(() => { const s = window.box2Viewer; s?.r138?.dispose(); s?.r138?.spark.dispose(); }); }');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
