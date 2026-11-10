// Capture actual rendered transition frames without changing the candidate runtime.
const fs = require("fs"), path = require("path"), Module = require("module");
const driver = path.join(__dirname, "run_frustum_sort.cjs");
const { once } = require("./frustum_sort_build.cjs");
const record = `
    if (state.r145Transient && state.r145Transient.samples.length < 3) {
      const gl = renderer.getContext(), width = renderer.domElement.width, height = renderer.domElement.height;
      const pixels = new Uint8Array(width * height * 4);
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      const sample = { index: state.r145Transient.samples.length, width, height,
        world: camera.matrixWorld.elements.slice(), projection: camera.projectionMatrix.elements.slice(),
        quaternion: camera.quaternion.toArray(), position: camera.position.toArray(),
        direction: camera.getWorldDirection(camera.position.clone()).toArray(),
        active: spark.r145DrawActive ?? spark.activeSplats, originalActive: spark.activeSplats,
        maskSelected: spark.r145MaskSelected ?? false, sorting: spark.sorting, sortDirty: spark.sortDirty };
      state.r145Transient.samples.push(sample);
      const pending = crypto.subtle.digest("SHA-256", pixels).then(digest => {
        sample.rgba_sha256 = [...new Uint8Array(digest)].map(x => x.toString(16).padStart(2, "0")).join("");
      });
      state.r145Transient.pending.push(pending);
      if (state.r145Transient.samples.length < 3) requestRender();
    }
`;
const main = `
async function transientEpisode(browser, enabled) {
  const entry = await contextPage(browser, enabled), { page, context, errors } = entry;
  const episodes = [];
  try {
    await page.goto(entry.url, { waitUntil: "domcontentloaded" }); await settle(page);
    await page.getByLabel("移动方式").selectOption("fly");
    await page.getByLabel("移动时优先流畅").uncheck(); await settle(page);
    const reference = await page.evaluate(() => ({ position: window.box2Viewer.getCameraPosition(), direction: window.box2Viewer.getCameraDirection() }));
    for (const sign of [-1, 1, -1]) {
      await page.evaluate(({ reference, sign }) => {
        const s = window.box2Viewer;
        s.r145Transient = { samples: [], pending: [] };
        s.r138.freeCamera(reference.position, reference.direction.map(x => x * sign));
      }, { reference, sign });
      await page.waitForFunction(() => window.box2Viewer.r145Transient.samples.length === 3, null, { timeout: 15000 });
      const samples = await page.evaluate(async () => {
        const s = window.box2Viewer;
        await Promise.all(s.r145Transient.pending);
        return s.r145Transient.samples;
      });
      episodes.push({ sign, samples });
      await settle(page);
    }
    check(errors.length === 0, "actual_transition_original_runtime_errors");
    return { enabled, reference, episodes, errors };
  } finally {
    try { await page.evaluate(() => window.box2Viewer?.r138.dispose()); }
    finally { await context.close(); }
  }
}
(async () => {
  const started = performance.now(), gpuBefore = gpuState();
  const browser = await chromium.launch({ headless: true, args: ["--use-angle=d3d11"] });
  let result;
  try {
    const native = await transientEpisode(browser, false), candidate = await transientEpisode(browser, true);
    const rows = native.episodes.flatMap((left, i) => left.samples.map((a, j) => {
      const b = candidate.episodes[i].samples[j];
      const cameraExact = ["index", "width", "height", "world", "projection", "quaternion", "position", "direction"].every(key => JSON.stringify(a[key]) === JSON.stringify(b[key]));
      return { episode: i, frame: j, sign: left.sign, camera_exact: cameraExact,
        rgba_bytes_exact: a.rgba_sha256 === b.rgba_sha256, native_active: a.active, candidate_active: b.active,
        native_sorting: a.sorting, candidate_sorting: b.sorting };
    }));
    check([...sources, ...inputIdentities].every(row => identity(row.path).sha256 === row.sha256), "all_original_transient_source_and_asset_identities_unchanged");
    result = { status: rows.every(row => row.camera_exact && row.rgba_bytes_exact) ? "actual_transition_frames_exact" : "actual_transition_quality_changed_no_acceptance",
      sources, inputIdentities, derivation, native, candidate, rows, checks, gpuBefore, gpuAfter: gpuState(),
      browserVersion: browser.version(), newOptimizationCount: 0, noUserTabsOrListeningService: true };
  } catch (error) {
    result = { status: "failed_verification", error: String(error), stack: error.stack, sources, inputIdentities, derivation,
      checks, gpuBefore, newOptimizationCount: 0 };
    throw error;
  } finally {
    await browser.close(); result.elapsedSeconds = (performance.now() - started) / 1000;
    publish(result);
    console.log(JSON.stringify({ status: result.status, rows: result.rows, elapsedSeconds: result.elapsedSeconds }));
  }
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
`;
const injection = `
source = once(source, 'const sources = sourceFiles.map(identity), inputIdentities = inputs.map(identity);',
  'sourceFiles.push(' + ${JSON.stringify(JSON.stringify(driver))} + ');\\nconst sources = sourceFiles.map(identity), inputIdentities = inputs.map(identity);');
source = once(source, '  return code;\\n}\\nasync function contextPage',
  '  code = replaceOnce(code, "renderer.render(scene, camera); state.frames++;", "renderer.render(scene, camera); state.frames++;" + ' + ${JSON.stringify(JSON.stringify(record))} + ');\\n  return code;\\n}\\nasync function contextPage');
const transientMainStart = source.indexOf("(async () => {\\n  const started = performance.now(), gpuBefore = gpuState();");
if (transientMainStart < 0) throw new Error("The current qualified browser entry changed");
source = source.slice(0, transientMainStart) + ${JSON.stringify(main)};
derived._compile(source, __filename);
`;
let program = once(fs.readFileSync(driver, "utf8"), "derived._compile(source, __filename);", injection);
const moduleInstance = new Module(__filename, module);
moduleInstance.filename = __filename;
moduleInstance.paths = Module._nodeModulePaths(__dirname);
moduleInstance._compile(program, __filename);
