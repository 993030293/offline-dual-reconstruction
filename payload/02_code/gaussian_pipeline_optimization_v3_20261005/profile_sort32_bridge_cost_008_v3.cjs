// Preserve separate donor and executed-source snapshots after the source-only naming collision.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "profile_sort32_bridge_cost_008.cjs");
let source = fs.readFileSync(original, "utf8");
source = source.replaceAll("sort32_bridge_page_observer.js", "sort32_bridge_page_observer_v2.js");
const before = '"const sources = [__filename, bridgePageHelper, bridgeWorkerBlock, originalDonor, glHelper,"';
if (source.split(before).length !== 2) throw new Error("The original diagnostic source attribution boundary changed");
source = source.replace(before, JSON.stringify("const sources = [__filename, " + JSON.stringify(original) + ", bridgePageHelper, bridgeWorkerBlock, originalDonor, glHelper,"));
const copyBefore = 'for (const item of sources) fs.copyFileSync(item.path, path.join(output, "sources", path.basename(item.path)));';
const copyAfter = 'for (const item of sources) fs.copyFileSync(item.path, path.join(output, "sources", item.path === originalDonor ? "DONOR_COMPILED_DRIVER_BODY.cjs" : path.basename(item.path)));';
const insertion = 'source = once(source, ' + JSON.stringify(copyBefore) + ', ' + JSON.stringify(copyAfter) + ');\nconst compiled = source;';
if (source.split("const compiled = source;").length !== 2) throw new Error("The generated-source snapshot boundary changed");
source = source.replace("const compiled = source;", insertion);
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
