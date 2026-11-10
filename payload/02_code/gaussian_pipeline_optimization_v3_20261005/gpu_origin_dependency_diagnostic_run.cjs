// Run one bounded real hardware diagnostic in a dedicated private release.
const started = process.hrtime.bigint();
const fs = require("fs"), path = require("path"), crypto = require("crypto"), http = require("http");
const { execFileSync } = require("child_process");
const { chromium } = require("C:/Users/99303/git/playwright/node_modules/playwright");
const ROOT = "C:/lidar-camera-fusion-mask-robust-downsampling";
const CODE = path.join(ROOT, "02_code/gaussian_pipeline_optimization_v3_20261005");
const OWN = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v3_20261005/gpu_origin_dependency_001");
const RELEASE = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release");
const PYTHON = path.join(ROOT, "05_envs/sfm_py311_v1/Scripts/python.exe");
const args = process.argv.slice(2), OUTPUT = path.resolve(args[args.indexOf("--output") + 1]);
if (!OUTPUT.startsWith(OWN + path.sep)) throw new Error("Only an isolated R118 child directory is writable");
const now = () => Number(process.hrtime.bigint() - started) / 1e9;
const save = (name, value) => fs.writeFileSync(path.join(OUTPUT, name), JSON.stringify(value, null, 2));
const identity = name => { const bytes = fs.readFileSync(name); return { path: name, bytes: bytes.length,
  sha256: crypto.createHash("sha256").update(bytes).digest("hex") }; };
const processLedger = new Map(), processSamples = [];
function resources(reason) {
  const gpu = execFileSync("nvidia-smi", ["--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
    { encoding: "utf8", windowsHide: true }).trim().split(",").map(Number);
  const script = "import psutil,sys,json; p=psutil.Process(int(sys.argv[1])); rows=[]\nfor x in [p]+p.children(recursive=True):\n try: rows.append({'pid':x.pid,'creation_time':x.create_time(),'name':x.name(),'rss':x.memory_info().rss})\n except (psutil.NoSuchProcess,psutil.AccessDenied): pass\nprint(json.dumps(rows))";
  const rows = JSON.parse(execFileSync(PYTHON, ["-B", "-c", script, String(process.pid)], { encoding: "utf8", windowsHide: true }));
  for (const row of rows) processLedger.set(row.pid + ":" + row.creation_time, row);
  const rss = rows.reduce((sum, row) => sum + row.rss, 0), disk = fs.statfsSync(OUTPUT);
  const row = { reason, atSeconds: now(), gpuTotalMiB: gpu[0], gpuUsedMiB: gpu[1], gpuFreeMiB: gpu[2],
    processTreeRSS: rss, processes: rows, diskFreeBytes: Number(disk.bavail) * Number(disk.bsize) };
  processSamples.push(row); save("RESOURCE_PID_LEDGER.json", { samples: processSamples, recorded: [...processLedger.values()],
    coverage: "actual owned process tree sampled at every registered boundary; unsampled transient processes are not claimed", counted: false });
  if (gpu.length !== 3 || gpu.some(n => !Number.isFinite(n)) || gpu[1] > 12288 || gpu[2] < 2048 || rss > 8 * 1024 ** 3 || row.diskFreeBytes < 20 * 1024 ** 3) {
    throw new Error("The original R118 GPU/process-tree/disk resource gate failed");
  }
  if (now() > 1800) throw new Error("The original 1800 second diagnostic node gate failed");
  return row;
}
const packed = path.join(RELEASE, "combined/cases/packed");
const sources = [__filename, path.join(CODE, "gpu_origin_dependency_diagnostic.mjs"),
  path.join(CODE, "origin_dependency.mjs"), path.join(CODE, "origin_dependency_browser_005.mjs"),
  ...["spark.module.js", "three.module.js", "three.core.js", "OrbitControls.js"].map(n => path.join(RELEASE, "shared-build/vendor", n)),
  ...["viewer.js", "lazy_models.js", "packed_splat_asset.js", "persistent_asset_cache.js", "render_schedule.js",
    "motion_resolution.js", "render_filter_contract.js", "calibrated_camera.js"].map(n => path.join(packed, n)),
  path.join(CODE, "run_with_resource_lease.py"), "C:/Users/99303/git/playwright/node_modules/playwright/package.json"];
const inputs = ["scene.spk", "box.spk", "scene.ply", "box.ply", "index.html", "display_manifest.json", "camera_path.json", "render_filter.json"]
  .map(n => path.join(packed, n)).concat(path.join(RELEASE, "combined/ASSET_MANIFEST.json"));
function verify(contract) {
  if (JSON.stringify(contract.identities) !== JSON.stringify([...sources, ...inputs].map(identity))) throw new Error("Frozen diagnostic source/input changed");
}
function viewer(mode) {
  let text = fs.readFileSync(path.join(packed, "viewer.js"), "utf8");
  const substitutions = [
    ['import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";',
      'import { SparkRenderer as NativeSparkRenderer, SplatMesh, PackedSplats, SplatGenerator, SplatTransformer } from "@sparkjsdev/spark";\n' +
      'import { createOriginDependency } from "/r118/origin_dependency.mjs";\n' +
      'import { attachGpuDiagnostic } from "/r118/gpu_origin_dependency_diagnostic.mjs";\n' +
      'const originDependency = ' + (mode === "old_candidate" ? 'createOriginDependency({ THREE, NativeSparkRenderer, NativeSplatMesh: SplatMesh, NativePackedSplats: PackedSplats, NativeSplatGenerator: SplatGenerator, NativeSplatTransformer: SplatTransformer })' : 'null') + ';\n' +
      'const SparkRenderer = originDependency?.SparkRenderer ?? NativeSparkRenderer;'],
    ['scene.add(new SparkRenderer({ renderer, onDirty: requestRender, ...filterOptions }));',
      'const spark = new SparkRenderer({ renderer, onDirty: requestRender, ...filterOptions }); scene.add(spark);'],
    ['load: async mode => new SplatMesh({ splats: await loadPackedAsset("./" + mode + ".spk", assets[mode]), lod: false })',
      'load: async mode => { const mesh = new SplatMesh({ splats: await loadPackedAsset("./" + mode + ".spk", assets[mode]), lod: false }); await mesh.initialized; originDependency?.install(mesh, spark); return mesh; }'],
    ['state.renderer = renderer;', 'state.renderer = renderer;\nstate.motionResolutionEnabled = false; ui("motion-priority").checked = false;\n' +
      'attachGpuDiagnostic({ state, renderer, scene, camera, spark, controls, modelRegistry, requestRender, dependency: originDependency });']
  ];
  for (const [before, after] of substitutions) {
    if (text.split(before).length !== 2) throw new Error("Original viewer anchor changed: " + before);
    text = text.replace(before, after);
  }
  return text;
}
async function server() {
  const viewers = Object.fromEntries(["native", "old_candidate"].map(mode => [mode, viewer(mode)]));
  for (const [mode, text] of Object.entries(viewers)) fs.writeFileSync(path.join(OUTPUT, mode + "_owned_viewer.js"), text);
  const server = http.createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, "http://127.0.0.1").pathname);
      if (request.method === "POST" && pathname.startsWith("/r115/publish/")) {
        const parts = pathname.slice(14).split("/");
        if (parts.length !== 2 || !/^[a-zA-Z0-9_]+$/.test(parts[0]) || !/^[a-z_0-9]+\.bin$/.test(parts[1])) throw new Error("Invalid isolated publication path");
        const folder = path.join(OUTPUT, "actual", parts[0]); fs.mkdirSync(folder, { recursive: true });
        const chunks = []; let count = 0;
        for await (const chunk of request) { count += chunk.length; if (count > 64 * 1024 ** 2) throw new Error("Original binary publication budget exceeded"); chunks.push(chunk); }
        fs.writeFileSync(path.join(folder, parts[1]), Buffer.concat(chunks), { flag: "wx" });
        response.writeHead(201); response.end(); return;
      }
      if (request.method !== "GET" && request.method !== "HEAD") { response.writeHead(405); response.end(); return; }
      if (/^\/r118\/(origin_dependency\.mjs|origin_dependency_browser_005\.mjs|gpu_origin_dependency_diagnostic\.mjs)$/.test(pathname)) {
        response.writeHead(200, { "Content-Type": "text/javascript", "Cache-Control": "no-store" }); response.end(fs.readFileSync(path.join(CODE, path.basename(pathname)))); return;
      }
      const match = /^\/(native|old_candidate)\/(.+)$/.exec(pathname);
      if (!match) { response.writeHead(404); response.end(); return; }
      const file = path.resolve(RELEASE, match[2]);
      if (!file.startsWith(RELEASE + path.sep) || !fs.statSync(file).isFile()) throw new Error("A private request left the original release");
      const bytes = match[2] === "combined/cases/packed/viewer.js" ? Buffer.from(viewers[match[1]]) : fs.readFileSync(file);
      const type = /\.js$/.test(file) ? "text/javascript" : /\.html$/.test(file) ? "text/html" : /\.json$/.test(file) ? "application/json" : "application/octet-stream";
      response.writeHead(200, { "Content-Type": type, "Content-Length": bytes.length, "Cache-Control": "no-store" });
      response.end(request.method === "HEAD" ? undefined : bytes);
    } catch (error) { response.writeHead(500); response.end(String(error)); }
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  return { server, url: "http://127.0.0.1:" + server.address().port };
}
async function open(browser, url, mode) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  const page = await context.newPage(), errors = [], consoleMessages = [];
  page.on("pageerror", error => errors.push(String(error)));
  page.on("console", message => consoleMessages.push({ type: message.type(), text: message.text(),
    location: message.location(), atSeconds: now() }));
  await page.goto(url + "/" + mode + "/combined/cases/packed/index.html?mode=scene", { waitUntil: "load", timeout: 60000 });
  await page.waitForFunction(() => window.box2Viewer?.ready && window.box2Viewer?.gpuDiagnostic, { timeout: 60000 });
  await page.evaluate(() => { const element = document.getElementById("navigation-mode"); element.value = "fly"; element.dispatchEvent(new Event("change", { bubbles: true })); window.box2Viewer.setView(508); });
  await page.evaluate(() => window.box2Viewer.originHarness.settle());
  const gpu = await page.evaluate(() => { const gl = window.box2Viewer.renderer.getContext(), ext = gl.getExtension("WEBGL_debug_renderer_info"); return { WebGL2: gl instanceof WebGL2RenderingContext, renderer: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER) }; });
  if (!gpu.WebGL2 || !/NVIDIA|RTX 5060/i.test(gpu.renderer) || /SwiftShader|Software/i.test(gpu.renderer)) throw new Error("Actual hardware WebGL2 was not established");
  resources("opened_" + mode);
  return { context, page, errors, consoleMessages, gpu };
}
async function exportTrace(side, mode, suffix) {
  const trace = await side.page.evaluate(() => window.box2Viewer.gpuDiagnostic.export());
  trace.originalBrowserConsole = side.consoleMessages;
  save(mode + "_TRACE_" + suffix + ".json", trace);
  return trace;
}
async function capture(side, label) {
  const row = await side.page.evaluate(name => window.box2Viewer.originHarness.capture(name), label);
  fs.writeFileSync(path.join(OUTPUT, "actual", label, "STATE.json"), JSON.stringify(row, null, 2), { flag: "wx" });
  if (side.errors.length || row.errors.length || !row.sourceSortingClosed || !row.originalInstance || row.numSh !== 3 || row.maxSh !== 3 || !row.nativeIntegrityChecks) throw new Error("Original full source/SH/capture guard failed");
  return row;
}
function compare(a, b) {
  const parts = ["pixels.bin", "ordering_gpu.bin", "ordering_worker.bin", "packed_gpu.bin", "camera.bin", "source_packed.bin", "source_sh1.bin", "source_sh2.bin", "source_sh3.bin"];
  const result = {};
  for (const key of ["width", "height", "mode", "activeSplats", "pointCount", "maxSh", "numSh", "meshCount", "projection", "originalInstance", "sourceSortingClosed", "errors", "nativeIntegrityChecks"]) {
    if (JSON.stringify(a[key]) !== JSON.stringify(b[key])) throw new Error("Complete diagnostic metadata differs: " + key);
  }
  for (const part of parts) {
    const files = [a, b].map(row => path.join(OUTPUT, "actual", row.label, part));
    if (fs.existsSync(files[0]) !== fs.existsSync(files[1])) throw new Error("Initial full source publication differs: " + part);
    if (!fs.existsSync(files[0])) continue;
    const left = fs.readFileSync(files[0]), right = fs.readFileSync(files[1]);
    if (!left.equals(right)) throw new Error("Actual complete diagnostic bytes differ: " + part);
    result[part] = { bytes: left.length, completeBytesEqual: true };
  }
  return result;
}
async function main() {
  fs.mkdirSync(OWN, { recursive: true }); fs.mkdirSync(OUTPUT, { recursive: false });
  const contract = { task: "R118_one_bounded_causal_diagnostic", identities: [...sources, ...inputs].map(identity),
    oldCandidateReadOnly: true, newAlgorithmFrozen: false, fullSH: 3, viewport: [1440, 900, 1],
    maximumNativeInterventions: 1, noPerformanceClaims: true, counted: false };
  fs.mkdirSync(path.join(OUTPUT, "source_snapshot"));
  sources.forEach((name, index) => fs.copyFileSync(name, path.join(OUTPUT, "source_snapshot", index.toString().padStart(2, "0") + "_" + path.basename(name))));
  save("CONTRACT.json", contract);
  save("PREREGISTRATION.json", { frozenUTC: new Date().toISOString(), steps: ["initial", "scene_rotation_base", "scene_rotation_before_context", "scene_real_context_restore_rotate"],
    diagnosticOnlyNativeIntervention: "Only after old-candidate nonzero restored-context GL guard, one original NativeSparkRenderer.updateInternal(autoUpdate=false), then exact full original capture comparison",
    firstDiagnosticFeeFullyChargedIfCostRuns: true, resourceGatesUnchanged: true, counted: false });
  const manifest = JSON.parse(fs.readFileSync(path.join(RELEASE, "combined/ASSET_MANIFEST.json"), "utf8"));
  for (const mode of ["scene", "box"]) {
    for (const [extension, key] of [[".ply", "source_ply_sha256"], [".spk", "asset_sha256"]]) {
      if (contract.identities.find(row => row.path === path.join(packed, mode + extension)).sha256 !== manifest[mode][key]) throw new Error("Original full PLY/SPK provenance changed");
    }
  }
  let service, browser, sides = {}, diagnosticFailure = null, intervention = null;
  const completed = [], cleanup = [];
  try {
    resources("before_browser"); service = await server();
    browser = await chromium.launch({ headless: true, args: ["--use-angle=d3d11"] }); resources("browser_launched");
    for (const mode of ["native", "old_candidate"]) sides[mode] = await open(browser, service.url, mode);
    save("BROWSER_GPU_IDENTITY.json", Object.fromEntries(Object.entries(sides).map(([mode, side]) => [mode, side.gpu])));
    const steps = [["initial", "none"], ["scene_rotation_base", "rotation-base"], ["scene_rotation_before_context", "rotate", 5], ["scene_real_context_restore_rotate", "context-restore-rotate", 6]];
    for (const [name, action, index] of steps) {
      const rows = {};
      for (const mode of ["native", "old_candidate"]) {
        try {
          await sides[mode].page.evaluate(([action, index]) => window.box2Viewer.originHarness.action(action, index), [action, index]);
          rows[mode] = await capture(sides[mode], mode + "_" + name);
          await exportTrace(sides[mode], mode, name);
        } catch (error) {
          const trace = await exportTrace(sides[mode], mode, "FAILURE");
          diagnosticFailure = { mode, step: name, errorType: error.name, error: error.message, numericErrors: trace.numericErrors, nativeCompleted: Boolean(rows.native),
            kind: mode === "old_candidate" && trace.numericErrors.length ? "actual_old_candidate_runtime_guard_failure" : "diagnostic_or_setup_failure", counted: false };
          save("DIAGNOSTIC_FAILURE.json", diagnosticFailure);
          if (mode === "old_candidate" && name === "scene_real_context_restore_rotate" && trace.numericErrors.some(row => row.code !== 0 && !row.nativeState.contextLost) && rows.native) {
            try {
              await sides[mode].page.evaluate(() => window.box2Viewer.gpuDiagnostic.nativeIntervention());
              const row = await capture(sides[mode], "old_candidate_diagnostic_native_full_update");
              intervention = { status: "diagnostic_native_full_update_completed", row, exactReferenceComparison: compare(rows.native, row), candidateAccepted: false };
            } catch (probeError) {
              intervention = { status: "diagnostic_native_full_update_failed", errorType: probeError.name, error: probeError.message, candidateAccepted: false };
            }
            save("NATIVE_INTERVENTION.json", intervention); await exportTrace(sides[mode], mode, "AFTER_NATIVE_INTERVENTION");
          }
          throw error;
        }
      }
      completed.push({ name, rows, exact: compare(rows.native, rows.old_candidate), resources: resources("completed_" + name) });
      save("DIAGNOSTIC_STATE.json", { completed, counted: false });
    }
  } catch (error) {
    if (!diagnosticFailure) diagnosticFailure = { kind: "diagnostic_or_setup_failure", errorType: error.name, error: error.message, counted: false };
    save("DIAGNOSTIC_FAILURE.json", diagnosticFailure);
  } finally {
    if (browser) {
      resources("before_owned_browser_close");
      for (const context of browser.contexts()) {
        for (const page of context.pages()) {
          try { cleanup.push(await page.evaluate(() => window.box2Viewer?.originHarness?.restore())); }
          catch (error) { cleanup.push({ errorType: error.name, error: error.message, ownedUnbindUnconfirmed: true }); }
        }
        await context.close();
      }
      await browser.close();
    }
    if (service) await new Promise(resolve => service.server.close(resolve));
    resources("after_owned_browser_server_close");
    save("RESOURCE_CLOSE.json", { browserCloseAwaited: Boolean(browser), serverCloseAwaited: Boolean(service),
      privateURL: service?.url, cleanup, processLedger: [...processLedger.values()], elapsedSeconds: now(), counted: false });
    verify(contract);
  }
  save("RESULT.json", { status: diagnosticFailure ? "bounded_diagnostic_observed_failure" : "bounded_diagnostic_no_failure_reproduced", diagnosticFailure,
    completedPrefixPairs: completed.length, completePrefixEvidence: completed, nativeIntervention: intervention,
    fullAlgorithmAcceptance: false, performancePairs: 0, diagnosticSeconds: now(), counted: false });
  console.log(JSON.stringify({ status: diagnosticFailure ? diagnosticFailure.kind : "no_failure_reproduced", completedPrefixPairs: completed.length,
    numericGLCodes: diagnosticFailure?.numericErrors?.map(row => row.code) ?? [], interventionStatus: intervention?.status, diagnosticSeconds: now(), counted: false }));
}
main().catch(error => { console.error(error.stack); process.exitCode = 1; });
