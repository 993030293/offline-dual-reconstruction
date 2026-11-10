// Load the frozen original preview and use an isolated instance of its actual sort32 worker.
const fs = require("fs"), path = require("path"), crypto = require("crypto");
const { chromium } = require("C:/Users/99303/git/playwright/node_modules/playwright");
const { execFileSync } = require("child_process");
const ROOT = path.resolve(__dirname, "../.."), OUT = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup007/R164");
const RELEASE = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release");
const PAGE = path.join(RELEASE, "baseline/cases/packed"), frozen = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup007/p3/sources");
const output = path.resolve(process.argv[2] ?? "");
if (!output.startsWith(OUT + path.sep) || fs.existsSync(output)) throw new Error("A fresh private R164 output is required");
fs.mkdirSync(path.join(output, "sources"), { recursive: true });
const identity = file => { const bytes = fs.readFileSync(file); return { path: file, bytes: bytes.length, sha256: crypto.createHash("sha256").update(bytes).digest("hex") }; };
const save = (name, value) => fs.writeFileSync(path.join(output, name), JSON.stringify(value, null, 2) + "\n", { flag: "wx" });
const helperNames = ["ordering_scratch_pool.js", "ordering_scratch_generic.js", "ordering_allocation_cost_v2.js", "ordering_upload_cost.js", "orbit_convergence_dag_v2.js"];
const sources = [__filename, ...helperNames.map(n => path.join(__dirname, n)), path.join(frozen, "EXECUTED_VIEWER.js"), path.join(frozen, "EXECUTED_PROBE.js")].map(identity);
const inputs = [...fs.readdirSync(PAGE).filter(n => /\.(js|json|spk|html)$/.test(n)).map(n => path.join(PAGE, n)), path.join(RELEASE, "baseline/ASSET_MANIFEST.json"),
  ...fs.readdirSync(path.join(RELEASE, "shared-build/vendor")).filter(n => /\.(js|txt)$/.test(n)).map(n => path.join(RELEASE, "shared-build/vendor", n))].map(identity);
for (let i = 0; i < sources.length; ++i) fs.copyFileSync(sources[i].path, path.join(output, "sources", i + "_" + path.basename(sources[i].path)));
const viewer = fs.readFileSync(path.join(frozen, "EXECUTED_VIEWER.js"), "utf8");
let probe = fs.readFileSync(path.join(frozen, "EXECUTED_PROBE.js"), "utf8");
const anchor = "  const originalRender = renderer.render;";
if (probe.split(anchor).length !== 2) throw new Error("The frozen native probe binding changed");
probe = probe.replace(anchor, "  state.r164NativeWorker = () => spark.sortWorker;\n" + anchor);
fs.writeFileSync(path.join(output, "sources/EXECUTED_PROBE.js"), probe, { flag: "wx" });
fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });
const prefix = "http://127.0.0.1:28239";
function gpu() {
  const a = execFileSync("nvidia-smi", ["--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"], { encoding: "utf8" }).trim().split(",").map(Number);
  if (a.length !== 3 || a.some(x => !Number.isFinite(x)) || a[1] > 12288 || a[2] < 2048) throw new Error("The device resource reserve changed");
  return a;
}
(async () => {
  const disk = fs.statfsSync(ROOT, { bigint: true });
  if (disk.bavail * disk.bsize < 20n * 1024n ** 3n) throw new Error("The disk reserve changed");
  const started = performance.now(), gpuBefore = gpu();
  save("PROTOCOL.json", { sources, inputs, realNativeWasmWorker: true, originalR162V2Preview: true, new_optimization_count: 0 });
  const browser = await chromium.launch({ headless: true, args: ["--use-angle=d3d11"] });
  const context = await browser.newContext({ viewport: { width: 960, height: 600 }, deviceScaleFactor: 1 });
  const page = await context.newPage(), errors = [];
  page.on("pageerror", e => errors.push(String(e)));
  await context.route("**/*", async route => {
    const url = new URL(route.request().url());
    if (url.origin !== prefix) throw new Error("The component requested an external origin");
    const relative = decodeURIComponent(url.pathname.slice(1)), name = path.basename(relative);
    const file = path.resolve(RELEASE, relative);
    if (!file.startsWith(RELEASE + path.sep)) throw new Error("The native route escaped the frozen release");
    const body = name === "viewer.js" ? viewer : name === "viewer_cost_probe.js" ? probe : name === "orbit_convergence_dag.js" ? fs.readFileSync(path.join(__dirname, "orbit_convergence_dag_v2.js")) : helperNames.includes(name) ? fs.readFileSync(path.join(__dirname, name)) : fs.readFileSync(file);
    await route.fulfill({ status: 200, body, contentType: name.endsWith(".js") ? "text/javascript" : name.endsWith(".json") ? "application/json" : name.endsWith(".html") ? "text/html" : "application/octet-stream" });
  });
  try {
    await page.goto(prefix + "/baseline/cases/packed/index.html?hooks=1", { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => {
      const s = window.box2Viewer, p = s?.costProbe?.snapshot();
      return s?.ready && !s.modelLoading && !s.isRenderPending() && p && !p.sorting && !p.sortDirty && !p.sortTimeout && p.active === 0 && s.r164NativeWorker();
    }, null, { timeout: 30000, polling: "raf" });
    const initial = await page.evaluate(() => window.box2Viewer.costProbe.capture());
    const result = await page.evaluate(async () => (await import("./ordering_scratch_generic.js")).runOrderingScratchGeneric(window.box2Viewer.r164NativeWorker()));
    const aliases = await page.evaluate(() => { window.box2Viewer.orbitDag.dispose(); return window.box2Viewer.costProbe.dispose(); });
    result.initial = initial; result.aliases = aliases; result.sources = sources; result.inputs = inputs; result.errors = errors;
    result.checks.native_initial_nonempty_GL0 = initial.glError === 0 && initial.nonuniformPixels > 0;
    result.checks.native_probes_drained_and_restored = aliases.active === 0 && aliases.originalRender && aliases.originalReadback && aliases.originalDriveSort;
    result.checks.original_vendor_assets_and_sources_unchanged = [...sources, ...inputs].every(row => JSON.stringify(row) === JSON.stringify(identity(row.path)));
    result.checks.actual_native_page_errors_empty = errors.length === 0;
    result.originalPreviewNewOrderingBytes = aliases.orderingAllocation.totals.verifiedNativeNewArrayBytes;
    result.totalKnownNewArrayRequestBytes += result.originalPreviewNewOrderingBytes;
    result.gpuBefore = gpuBefore; result.gpuAfter = gpu();
    result.status = Object.values(result.checks).every(Boolean) ? "real_sort32_and_native_preview_component_passed0" : "component_failed0";
    result.elapsedSecondsBeforeClose = (performance.now() - started) / 1000;
    save("RESULT.json", result);
    console.log(JSON.stringify({ status: result.status, checks: result.checks, cases: result.cases.length, pool: result.pool.totals, knownArrayBytes: result.totalKnownNewArrayRequestBytes }));
    if (!Object.values(result.checks).every(Boolean)) throw new Error("A real R164 component contract failed");
  } catch (error) {
    let partial;
    try { partial = await page.evaluate(() => ({ progress: window.r164ScratchGenericProgress,
      native: window.box2Viewer?.costProbe?.snapshot() })); }
    catch (snapshotError) { partial = { error: String(snapshotError) }; }
    save("FAILURE.json", { error: String(error), stack: error.stack, errors, partial, sources, inputs, new_optimization_count: 0 });
    throw error;
  } finally { await context.close(); await browser.close(); }
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
