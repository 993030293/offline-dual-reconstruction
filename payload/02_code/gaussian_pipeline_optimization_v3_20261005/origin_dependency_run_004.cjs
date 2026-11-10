// Execute source-bound private browser oracles and complete fresh-process pairs.
const started = process.hrtime.bigint();
const fs = require("fs"), path = require("path"), crypto = require("crypto"), http = require("http");
const { spawnSync, execFileSync } = require("child_process");
const { chromium } = require("C:/Users/99303/git/playwright/node_modules/playwright");
const ROOT = "C:/lidar-camera-fusion-mask-robust-downsampling";
const CODE = path.join(ROOT, "02_code/gaussian_pipeline_optimization_v3_20261005");
const RELEASE = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release");
const PYTHON = path.join(ROOT, "05_envs/sfm_py311_v1/Scripts/python.exe");
const argumentsList = process.argv.slice(2);
const option = name => argumentsList[argumentsList.indexOf(name) + 1];
const OUTPUT = path.resolve(option("--output"));
const now = () => Number(process.hrtime.bigint() - started) / 1e9;
const sha = bytes => crypto.createHash("sha256").update(bytes).digest("hex");
const save = (name, value) => fs.writeFileSync(path.join(OUTPUT, name), JSON.stringify(value, null, 2));
const identity = name => { const bytes = fs.readFileSync(name); return { path: name, bytes: bytes.length, sha256: sha(bytes) }; };
const owned = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v3_20261005/origin_dependency_001");
if (OUTPUT !== owned && !OUTPUT.startsWith(owned + path.sep)) throw new Error("Only the dedicated R115 output is writable");

function sourceFiles() {
  const packed = path.join(RELEASE, "combined/cases/packed");
  return [__filename, path.join(CODE, "origin_dependency.mjs"), path.join(CODE, "origin_dependency_browser_004.mjs"),
    ...["spark.module.js", "three.module.js", "three.core.js", "OrbitControls.js"].map(n => path.join(RELEASE, "shared-build/vendor", n)),
    ...["viewer.js", "lazy_models.js", "packed_splat_asset.js", "persistent_asset_cache.js", "render_schedule.js",
      "motion_resolution.js", "render_filter_contract.js", "calibrated_camera.js"].map(n => path.join(packed, n)),
    path.join(CODE, "run_with_resource_lease.py"),
    "C:/Users/99303/git/playwright/node_modules/playwright/package.json"];
}
function inputFiles() {
  const packed = path.join(RELEASE, "combined/cases/packed");
  return ["scene.spk", "box.spk", "scene.ply", "box.ply", "index.html", "display_manifest.json",
    "camera_path.json", "render_filter.json"].map(n => path.join(packed, n))
    .concat(path.join(RELEASE, "combined/ASSET_MANIFEST.json"));
}
function validateIdentity(contract) {
  const actual = contract.sourcesAndInputs.map(row => identity(row.path));
  if (JSON.stringify(actual) !== JSON.stringify(contract.sourcesAndInputs)) throw new Error("Frozen source or complete input changed");
}
function resources() {
  const fields = execFileSync("nvidia-smi", ["--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
    { encoding: "utf8", windowsHide: true }).trim().split(",").map(Number);
  if (fields.length !== 3 || fields.some(n => !Number.isFinite(n)) || fields[1] > 12288 || fields[2] < 2048) {
    throw new Error("The original whole-device 12 GiB/2 GiB GPU resource gate failed");
  }
  const rss = Number(execFileSync(PYTHON, ["-B", "-c", "import psutil,sys; p=psutil.Process(int(sys.argv[1])); print(sum(x.memory_info().rss for x in [p]+p.children(recursive=True) if x.is_running()))", String(process.pid)],
    { encoding: "utf8", windowsHide: true }));
  if (!Number.isFinite(rss) || rss > 8 * 1024 ** 3) throw new Error("The original 8 GiB Windows process-tree resource gate failed");
  const disk = fs.statfsSync(OUTPUT);
  if (Number(disk.bavail) * Number(disk.bsize) < 20 * 1024 ** 3) throw new Error("The original disk headroom gate failed");
  return { totalMiB: fields[0], usedMiB: fields[1], freeMiB: fields[2], rss, GPUbasis: "whole_device_including_other_owners" };
}
function prepareViewer(mode) {
  let text = fs.readFileSync(path.join(RELEASE, "combined/cases/packed/viewer.js"), "utf8");
  const replacements = [
    ['import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";',
      'import { SparkRenderer as NativeSparkRenderer, SplatMesh, PackedSplats, SplatGenerator, SplatTransformer } from "@sparkjsdev/spark";\n' +
      'import { createOriginDependency } from "/r115/origin_dependency.mjs";\n' +
      'import { attachOriginHarness } from "/r115/origin_dependency_browser_004.mjs";\n' +
      'const originDependency = ' + (mode === "candidate" ? 'createOriginDependency({ THREE, NativeSparkRenderer, NativeSplatMesh: SplatMesh, NativePackedSplats: PackedSplats, NativeSplatGenerator: SplatGenerator, NativeSplatTransformer: SplatTransformer })' : 'null') + ';\n' +
      'const SparkRenderer = originDependency?.SparkRenderer ?? NativeSparkRenderer;'],
    ['scene.add(new SparkRenderer({ renderer, onDirty: requestRender, ...filterOptions }));',
      'const spark = new SparkRenderer({ renderer, onDirty: requestRender, ...filterOptions }); scene.add(spark);'],
    ['load: async mode => new SplatMesh({ splats: await loadPackedAsset("./" + mode + ".spk", assets[mode]), lod: false })',
      'load: async mode => { const mesh = new SplatMesh({ splats: await loadPackedAsset("./" + mode + ".spk", assets[mode]), lod: false }); await mesh.initialized; originDependency?.install(mesh, spark); return mesh; }'],
    ['state.renderer = renderer;', 'state.renderer = renderer;\nstate.motionResolutionEnabled = false; ui("motion-priority").checked = false;\n' +
      'attachOriginHarness({ state, renderer, scene, camera, spark, controls, modelRegistry, requestRender, dependency: originDependency });']
  ];
  for (const [before, after] of replacements) {
    if (text.split(before).length !== 2) throw new Error("The original viewer source anchor differs: " + before);
    text = text.replace(before, after);
  }
  return text;
}
async function makeServer() {
  const viewers = Object.fromEntries(["native", "candidate"].map(mode => [mode, prepareViewer(mode)]));
  for (const [mode, text] of Object.entries(viewers)) fs.writeFileSync(path.join(OUTPUT, mode + "_owned_viewer.js"), text);
  const server = http.createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, "http://127.0.0.1").pathname);
      if (request.method === "POST" && pathname.startsWith("/r115/publish/")) {
        const bits = pathname.slice(14).split("/");
        if (bits.length !== 2 || !/^[a-zA-Z0-9_]+$/.test(bits[0]) || !/^[a-z_0-9]+\.bin$/.test(bits[1])) throw new Error("Invalid owned publication path");
        const folder = path.join(OUTPUT, "actual", bits[0]); fs.mkdirSync(folder, { recursive: true });
        const target = path.join(folder, bits[1]);
        if (fs.existsSync(target)) throw new Error("An original capture would be overwritten");
        const chunks = []; let bytes = 0;
        for await (const chunk of request) { bytes += chunk.length; if (bytes > 64 * 1024 ** 2) throw new Error("A complete publication exceeds the fixed byte budget"); chunks.push(chunk); }
        fs.writeFileSync(target, Buffer.concat(chunks), { flag: "wx" });
        response.writeHead(201); response.end(); return;
      }
      if (request.method !== "GET" && request.method !== "HEAD") { response.writeHead(405); response.end(); return; }
      if (pathname === "/r115/origin_dependency.mjs" || pathname === "/r115/origin_dependency_browser_004.mjs") {
        response.writeHead(200, { "Content-Type": "text/javascript", "Cache-Control": "no-store" });
        response.end(fs.readFileSync(path.join(CODE, path.basename(pathname)))); return;
      }
      const match = /^\/(native|candidate)\/(.+)$/.exec(pathname);
      if (!match) { response.writeHead(404); response.end(); return; }
      const target = path.resolve(RELEASE, match[2]);
      if (!target.startsWith(RELEASE + path.sep) || !fs.statSync(target).isFile()) throw new Error("A private request left the original release");
      const type = target.endsWith(".js") ? "text/javascript" : target.endsWith(".html") ? "text/html" : target.endsWith(".json") ? "application/json" : "application/octet-stream";
      const bytes = match[2] === "combined/cases/packed/viewer.js" ? Buffer.from(viewers[match[1]]) : fs.readFileSync(target);
      response.writeHead(200, { "Content-Type": type, "Content-Length": bytes.length, "Cache-Control": "no-store" });
      response.end(request.method === "HEAD" ? undefined : bytes);
    } catch (error) { response.writeHead(500); response.end(String(error)); }
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  return { server, url: "http://127.0.0.1:" + server.address().port };
}
async function openSide(browser, url, mode) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(String(error)));
  await page.goto(url + "/" + mode + "/combined/cases/packed/index.html?mode=scene", { waitUntil: "load", timeout: 60000 });
  await page.waitForFunction(() => window.box2Viewer?.ready && window.box2Viewer?.originHarness, { timeout: 60000 });
  await page.evaluate(() => {
    const mode = document.getElementById("navigation-mode");
    mode.value = "fly";
    mode.dispatchEvent(new Event("change", { bubbles: true }));
    window.box2Viewer.setView(508);
  });
  await page.evaluate(() => window.box2Viewer.originHarness.settle());
  const gpu = await page.evaluate(() => {
    const gl = window.box2Viewer.renderer.getContext(), extension = gl.getExtension("WEBGL_debug_renderer_info");
    return { renderer: extension ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER), WebGL2: gl instanceof WebGL2RenderingContext };
  });
  if (!gpu.WebGL2 || !/NVIDIA|RTX 5060/i.test(gpu.renderer) || /SwiftShader|Software/i.test(gpu.renderer)) throw new Error("Actual browser hardware GPU was not established: " + JSON.stringify(gpu));
  return { context, page, errors, gpu };
}
async function act(side, action, index) {
  return await side.page.evaluate(([name, value]) => window.box2Viewer.originHarness.action(name, value), [action, index]);
}
async function capture(side, label) {
  const row = await side.page.evaluate(label => window.box2Viewer.originHarness.capture(label), label);
  fs.writeFileSync(path.join(OUTPUT, "actual", label, "STATE.json"), JSON.stringify(row, null, 2), { flag: "wx" });
  if (side.errors.length || row.errors.length || !row.originalInstance || !row.sourceSortingClosed || row.maxSh !== 3 || row.numSh !== 3 || !row.nativeIntegrityChecks) {
    throw new Error("An original full SH/source/lifecycle capture guard failed");
  }
  return row;
}
function compare(a, b) {
  const protectedNames = ["width", "height", "mode", "activeSplats", "pointCount", "maxSh", "numSh", "meshCount", "projection", "originalInstance", "sourceSortingClosed", "errors", "nativeIntegrityChecks"];
  for (const key of protectedNames) if (JSON.stringify(a[key]) !== JSON.stringify(b[key])) {
    const error = new Error("Required actual browser semantic field differs: " + key); error.kind = "candidate_semantics"; throw error;
  }
  const result = {};
  for (const file of ["pixels.bin", "ordering_gpu.bin", "ordering_worker.bin", "packed_gpu.bin", "camera.bin"]) {
    const original = fs.readFileSync(path.join(OUTPUT, "actual", a.label, file));
    const candidate = fs.readFileSync(path.join(OUTPUT, "actual", b.label, file));
    if (!original.equals(candidate)) {
      let index = 0; while (index < Math.min(original.length, candidate.length) && original[index] === candidate[index]) index++;
      const error = new Error("Required exact original/candidate bytes differ in " + file + " at byte " + index);
      error.kind = "candidate_semantics"; error.actualDifference = { file, index, originalBytes: original.length, candidateBytes: candidate.length,
        originalByte: original[index], candidateByte: candidate[index] }; throw error;
    }
    result[file] = { bytes: original.length, completeBytesEqual: true };
  }
  for (const file of ["source_packed.bin", "source_sh1.bin", "source_sh2.bin", "source_sh3.bin"]) {
    const first = path.join(OUTPUT, "actual", a.label, file), second = path.join(OUTPUT, "actual", b.label, file);
    if (fs.existsSync(first) !== fs.existsSync(second) || fs.existsSync(first) && !fs.readFileSync(first).equals(fs.readFileSync(second))) {
      const error = new Error("Required initial complete source buffer differs: " + file); error.kind = "candidate_semantics"; throw error;
    }
    if (fs.existsSync(first)) result[file] = { bytes: fs.statSync(first).size, completeBytesEqual: true };
  }
  return result;
}
async function generic(contract) {
  let server, browser;
  const completed = [], restored = [];
  let current = null;
  try {
    validateIdentity(contract);
    server = await makeServer();
    browser = await chromium.launch({ headless: true, args: ["--use-angle=d3d11"] });
    const sides = {};
    for (const mode of ["native", "candidate"]) sides[mode] = await openSide(browser, server.url, mode);
    save("BROWSER_GPU_IDENTITY.json", { native: sides.native.gpu, candidate: sides.candidate.gpu, source: "actual WebGL2 UNMASKED_RENDERER_WEBGL", counted: false });
    const steps = [
      { name: "initial", action: "none" },
      { name: "scene_rotation_base", action: "rotation-base" },
      { name: "scene_rotation_before_context", action: "rotate", index: 5 },
      { name: "scene_real_context_restore_rotate", action: "context-restore-rotate", index: 6 },
      { name: "box_select", action: "select", index: "box" },
      { name: "box_rotation_base", action: "rotation-base" },
      { name: "box_rotation_before_context", action: "rotate", index: 5 },
      { name: "box_real_context_restore_rotate", action: "context-restore-rotate", index: 6 }
    ];
    save("GENERIC_PLAN.json", { steps, actualHardwareRequired: true, exactBinaryParts: ["pixels", "GPUordering", "workerOrdering", "GPUpackedPoints", "camera"],
      additionalRequired: ["synthetic_irregular_full_SH", "modifier_LoD_cov_ext_paged_original_routes", "instance_method_rejection", "native_source_rejection", "real_original_controls_resize_context_recovery"], counted: false });
    for (const step of steps) {
      if (now() > 1800) throw new Error("The original 1800 second node budget was exceeded");
      current = step;
      const rows = {};
      for (const mode of ["native", "candidate"]) {
        await act(sides[mode], step.action, step.index);
        rows[mode] = await capture(sides[mode], mode + "_" + step.name);
      }
      const check = compare(rows.native, rows.candidate);
      completed.push({ step, rows, check, resources: resources() });
      save("GENERIC_STATE.json", { current, completed, counted: false });
    }
    save("GENERIC_PRIMARY_RESULT.json", { status: "context_restoration_cases_passed_other_supplemental_required_pending",
      completed, fullSemanticAcceptance: false, counted: false });
  } catch (error) {
    save("GENERIC_FAILURE.json", { status: error.kind === "candidate_semantics" ? "failed_verification" : "validator_setup_failed",
      errorType: error.name, error: error.message, kind: error.kind ?? "validator_setup", actualDifference: error.actualDifference,
      current, completed, remaining: "All steps not in completed plus all supplemental formats/controls", counted: false });
    throw error;
  } finally {
    if (browser) {
      for (const context of browser.contexts()) {
        for (const page of context.pages()) {
          try { restored.push(await page.evaluate(() => window.box2Viewer?.originHarness?.restore())); }
          catch (error) { restored.push({ errorType: error.name, error: error.message, completeRestoration: false }); }
        }
        await context.close();
      }
      await browser.close();
    }
    if (server) await new Promise(resolve => server.server.close(resolve));
    save("GENERIC_RESOURCE_CLOSE.json", { browserClosed: Boolean(browser), serverClosed: Boolean(server), restored,
      elapsedSeconds: now(), existingTabsOrServicesTouched: false, counted: false });
  }
}
async function main() {
  fs.mkdirSync(OUTPUT, { recursive: false });
  const sources = sourceFiles(), inputs = inputFiles();
  const contract = { mechanism: "R115_static_full_SH_radial_origin_dependency", sourcesAndInputs: [...sources, ...inputs].map(identity),
    captureIndices: Array.from({ length: 24 }, (_, i) => Math.round(i * 1015 / 23)), fixedOriginRotations: 24,
    resolution: { width: 1440, height: 900, pixelRatio: 1 }, SH: 3, models: ["scene", "box"],
    minimumPhaseReduction: .1, maximumEachWholeRatio: 1.1, wholeMedianNonregression: true, freshProcessPairs: 3,
    comparisonFeeBothSides: true, firstPreparationDivisor: 6, fullSourceInputHashesOnEachLeaf: true,
    browserCloseServerCloseProcessCloseRequired: true, supplementalGPUGuardsRequired: true, counted: false };
  const snapshot = path.join(OUTPUT, "source_snapshot"); fs.mkdirSync(snapshot);
  sources.forEach((name, i) => fs.copyFileSync(name, path.join(snapshot, String(i).padStart(2, "0") + "_" + path.basename(name))));
  save("CONTRACT.json", contract);
  save("PREREGISTRATION.json", { status: "frozen_before_first_actual_GPU_or_cost", frozenUTC: new Date().toISOString(),
    firstRequiredFailureStopsContingent: true, noTrainingChanges: true, noOxfordTrainedGaussianAvailable: true,
    completePhaseScope: "first_source_prepare+full_source_input_hash+native_model_load_and_asset_verify+dependency_checks+original_generate_sort_wait_GPU_finish+allpixel_order_point_SH_camera_publication_validation+restore_close+pair_compare+actual_process_close",
    campaignInstrumentationSeparate: ["resource_monitor", "campaign_summary_readback", "contract_freeze_and_generic"],
    candidateStatus: "unverified", counted: false });
  const firstPreparation = now();
  const assetManifest = JSON.parse(fs.readFileSync(path.join(RELEASE, "combined/ASSET_MANIFEST.json"), "utf8"));
  for (const mode of ["scene", "box"]) {
    const fileRow = name => contract.sourcesAndInputs.find(row => row.path === path.join(RELEASE, "combined/cases/packed", mode + name));
    if (fileRow(".ply").sha256 !== assetManifest[mode].source_ply_sha256 || fileRow(".spk").sha256 !== assetManifest[mode].asset_sha256) {
      throw new Error("Original complete PLY/SPK provenance failed before browser execution");
    }
  }
  try {
    resources();
    await generic(contract);
    save("RESULT.json", { status: "context_restoration_generic_passed_other_supplemental_and_cost_pending", firstPreparation,
      fullSemanticAcceptance: false, counted: false });
  } catch (error) {
    save("RESULT.json", { status: error.kind === "candidate_semantics" ? "negative_first_required_semantics_failure" : "validator_setup_failed",
      failureType: error.name, failure: error.message, firstPreparation, fullCampaignSeconds: now(),
      pairs: { status: error.kind === "candidate_semantics" ? "not_run_after_required_semantics_failure" : "not_run_validator_setup_failed", complete: 0 },
      Oxford: "not_run_no_real_trained_Gaussian_input", training: "not_run_no_training_change",
      originalNegativePreserved: true, counted: false });
    process.exitCode = 1;
  }
  validateIdentity(contract);
}
main().catch(error => { console.error(error.stack); process.exitCode = 1; });
