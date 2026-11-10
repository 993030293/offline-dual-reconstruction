// Correct only the observer's exact SDK URL revocation source match.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "profile_sort32_bridge_cost_008.cjs");
let source = fs.readFileSync(original, "utf8");
source = source.replaceAll("sort32_bridge_page_observer.js", "sort32_bridge_page_observer_v2.js");
const before = '"const sources = [__filename, bridgePageHelper, bridgeWorkerBlock, originalDonor, glHelper,"';
if (source.split(before).length !== 2) throw new Error("The original diagnostic source attribution boundary changed");
source = source.replace(before, JSON.stringify("const sources = [__filename, " + JSON.stringify(original) + ", bridgePageHelper, bridgeWorkerBlock, originalDonor, glHelper,"));
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
