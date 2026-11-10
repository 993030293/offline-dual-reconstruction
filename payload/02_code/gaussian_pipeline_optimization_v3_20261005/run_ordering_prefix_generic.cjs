// Run real Three/WebGL2 component checks in a private route and retain exact source snapshots.
const fs = require("fs"), path = require("path"), crypto = require("crypto");
const { chromium } = require("C:/Users/99303/git/playwright/node_modules/playwright");
const { execFileSync } = require("child_process");
const ROOT = path.resolve(__dirname, "../.."), OUT = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup006/R163");
const output = path.resolve(process.argv[2] ?? ""), code = __dirname;
const vendor = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release/shared-build/vendor");
if (!output.startsWith(OUT + path.sep) || fs.existsSync(output)) throw new Error("A fresh bounded component output is required");
fs.mkdirSync(path.join(output, "sources"), { recursive: true });
const identity = file => { const bytes = fs.readFileSync(file); return { path: file, bytes: bytes.length, sha256: crypto.createHash("sha256").update(bytes).digest("hex") }; };
const sourceFiles = [__filename, path.join(code, "ordering_active_prefix.js"), path.join(code, "ordering_active_prefix_generic.js")];
const sources = sourceFiles.map(identity), inputs = ["three.module.js", "three.core.js", "THREE-LICENSE.txt"].map(n => identity(path.join(vendor, n)));
for (const source of sourceFiles) fs.copyFileSync(source, path.join(output, "sources", path.basename(source)));
const save = (name, value) => fs.writeFileSync(path.join(output, name), JSON.stringify(value, null, 2) + "\n", { flag: "wx" });
const memory = execFileSync("nvidia-smi", ["--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"], { encoding: "utf8" }).trim().split(",").map(Number);
if (memory.length !== 3 || memory[1] > 12288 || memory[2] < 2048) throw new Error("The original GPU resource reserve changed");
(async () => {
  const browser = await chromium.launch({ headless: true, args: ["--use-angle=d3d11"] });
  const context = await browser.newContext(), page = await context.newPage(), errors = [];
  page.on("pageerror", error => errors.push(String(error)));
  const prefix = "http://127.0.0.1:28233";
  await context.route("**/*", async route => {
    const url = new URL(route.request().url());
    if (url.origin !== prefix) throw new Error("The component requested an external origin");
    const name = url.pathname.slice(1), native = ["three.module.js", "three.core.js"].includes(name);
    if (name === "index.html") return route.fulfill({ status: 200, body: "<!doctype html><meta charset=utf-8><title>Private integer upload component</title>", contentType: "text/html" });
    if (!native && !["ordering_active_prefix.js", "ordering_active_prefix_generic.js"].includes(name)) throw new Error("The component requested an unregistered source");
    await route.fulfill({ status: 200, body: fs.readFileSync(path.join(native ? vendor : code, name)), contentType: "text/javascript" });
  });
  try {
    await page.goto(prefix + "/index.html");
    const result = await page.evaluate(async () => (await import("./ordering_active_prefix_generic.js")).runOrderingPrefixGeneric());
    result.checks.original_vendor_and_sources_unchanged = [...sources, ...inputs].every(row => JSON.stringify(row) === JSON.stringify(identity(row.path)));
    result.checks.page_errors_empty = errors.length === 0;
    result.sources = sources; result.inputs = inputs; result.errors = errors; result.new_optimization_count = 0;
    result.status = Object.values(result.checks).every(Boolean) ? "real_WebGL_component_passed0_pending_complete_original_pairs_and_recovery" : "component_failed0";
    save("RESULT.json", result);
    console.log(JSON.stringify({ status: result.status, checks: result.checks, cases: result.cases.length, GPUWriteBytes: result.extraGPUWriteBytes, GPUReadBytes: result.extraGPUReadBytes }));
    if (!Object.values(result.checks).every(Boolean)) throw new Error("A real component or original input contract failed");
  } finally { await context.close(); await browser.close(); }
})().catch(error => { save("FAILURE.json", { error: String(error), new_optimization_count: 0 }); process.exitCode = 1; });
