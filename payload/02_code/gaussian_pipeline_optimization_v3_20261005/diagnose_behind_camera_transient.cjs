// Compare actual first frames while the unchanged whole-model culler is active.
const fs = require("fs"), path = require("path"), Module = require("module");
const driver = path.join(__dirname, "run_behind_camera_v3.cjs");
function once(source, before, after) {
  if (source.split(before).length !== 2) throw new Error("The original transient source anchor changed");
  return source.replace(before, after);
}
const record = `
    if (state.r138Transient && state.r138Transient.samples.length < 3) {
      const gl = renderer.getContext(), width = renderer.domElement.width, height = renderer.domElement.height;
      const pixels = new Uint8Array(width * height * 4);
      const packBuffer = gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING);
      if (renderer.getRenderTarget() !== null || gl.getParameter(gl.READ_FRAMEBUFFER_BINDING) !== null) throw new Error("The frame capture is not the displayed canvas");
      if (gl.getError() !== gl.NO_ERROR) throw new Error("A GL error preceded the frame capture");
      let originalTypedReadError = null;
      if (packBuffer !== null) {
        const invalid = new Uint8Array(pixels.length);
        gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, invalid);
        originalTypedReadError = gl.getError();
        if (originalTypedReadError !== gl.INVALID_OPERATION) throw new Error("The original PBO-bound typed read was not reproduced");
      }
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      try {
        gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        if (gl.getError() !== gl.NO_ERROR) throw new Error("The displayed canvas read failed");
      } finally { gl.bindBuffer(gl.PIXEL_PACK_BUFFER, packBuffer); }
      const sample = { index: state.r138Transient.samples.length, width, height,
        originalPackBufferBound: packBuffer !== null, originalTypedReadError, displayedFramebufferValidated: true,
        world: camera.matrixWorld.elements.slice(), projection: camera.projectionMatrix.elements.slice(),
        quaternion: camera.quaternion.toArray(), position: camera.position.toArray(),
        direction: camera.getWorldDirection(camera.position.clone()).toArray(),
        population: [state.fullGaussianCount, state.boxGaussianCount], visible: spark.visible,
        active: spark.visible ? spark.activeSplats : 0, sorting: spark.sorting,
        rejectedFrames: state.r138.culler.trace.rejectedFrames, draw: structuredClone(state.r138.drawCounter.trace) };
      state.r138Transient.samples.push(sample);
      state.r138Transient.pending.push(crypto.subtle.digest("SHA-256", pixels).then(digest => {
        sample.rgba_sha256 = [...new Uint8Array(digest)].map(x => x.toString(16).padStart(2, "0")).join("");
      }));
      if (state.r138Transient.samples.length < 3) requestRender();
    }
`;
const main = `
async function transient(browser, enabled) {
  const entry = await contextPage(browser, enabled), { page, context, errors } = entry;
  const steps = [];
  try {
    await page.goto(entry.url, { waitUntil: "domcontentloaded" }); await settle(page);
    await page.getByLabel("移动方式").selectOption("fly"); await page.getByLabel("移动时优先流畅").uncheck(); await settle(page);
    for (const range of ["scene", "box"]) {
      await page.getByLabel("展示范围").selectOption(range);
      await page.waitForFunction(r => window.box2Viewer.sceneMode === r && !window.box2Viewer.modelLoading, range); await settle(page);
      const view = await page.evaluate(() => {
        const s = window.box2Viewer, mesh = s.r138.models().find(model => model.visible);
        const bounds = mesh.getBoundingBox(true), center = bounds.getCenter(bounds.min.clone()), radius = bounds.min.distanceTo(bounds.max) / 2;
        const z = 1 - 1 / 12, xy = Math.sqrt(1 - z * z), direction = [xy, 0, z];
        const distance = 4 * radius + 2 * s.r138.camera.near;
        return { position: center.toArray().map((x, i) => x + direction[i] * distance), direction };
      });
      await step(page, "initial_front_" + range, () => page.evaluate(view => window.box2Viewer.r138.freeCamera(view.position, view.direction.map(x => -x)), view));
      for (const sign of [1, -1, 1]) {
        await page.evaluate(({ view, sign }) => {
          const s = window.box2Viewer; s.r138Transient = { samples: [], pending: [] };
          s.r138.freeCamera(view.position, view.direction.map(x => x * sign));
        }, { view, sign });
        await page.waitForFunction(() => window.box2Viewer.r138Transient.samples.length === 3, null, { timeout: 15000 });
        const samples = await page.evaluate(async () => { const s = window.box2Viewer; await Promise.all(s.r138Transient.pending); return s.r138Transient.samples; });
        steps.push({ range, sign, view, samples }); await settle(page);
      }
    }
    check(errors.length === 0, "actual_whole_renderer_transient_browser_errors");
    return { enabled, steps, errors };
  } finally {
    try { await page.evaluate(() => { const s = window.box2Viewer; s.r138.dispose(); s.r138.spark.dispose(); }); }
    finally { await context.close(); }
  }
}
(async () => {
  const started = performance.now(), gpuBefore = gpuState(), browser = await chromium.launch({ headless: true, args: ["--use-angle=d3d11"] });
  let result;
  try {
    const native = await transient(browser, false), candidate = await transient(browser, true);
    const rows = native.steps.flatMap((step, i) => step.samples.map((a, j) => {
      const b = candidate.steps[i].samples[j];
      return { range: step.range, sign: step.sign, frame: j,
        camera_population_exact: ["index", "width", "height", "world", "projection", "quaternion", "position", "direction", "population"].every(k => JSON.stringify(a[k]) === JSON.stringify(b[k])),
        rgba_bytes_exact: a.rgba_sha256 === b.rgba_sha256, candidate_visible: b.visible,
        native_active: a.active, candidate_active: b.active, actual_candidate_reject: !b.visible && b.rejectedFrames > 0 };
    }));
    check(rows.length === 18 && rows.filter(r => r.sign === 1).every(r => r.actual_candidate_reject), "actual_both_model_whole_back_reject_exercised");
    check([...native.steps, ...candidate.steps].flatMap(r => r.samples).every(r => r.displayedFramebufferValidated), "all36_displayed_canvas_reads_validated_and_original_GL_state_restored");
    check(native.steps.flatMap(r => r.samples).some(r => r.originalPackBufferBound && r.originalTypedReadError === 1282), "actual_original_PBO_bound_typed_read_failure_reproduced");
    check([...sources, ...inputIdentities].every(row => identity(row.path).sha256 === row.sha256), "whole_renderer_transient_original_inputs_unchanged");
    result = { status: rows.every(r => r.camera_population_exact && r.rgba_bytes_exact) ? "actual18_transition_frames_exact" : "actual_transition_quality_changed_no_acceptance",
      native, candidate, rows, checks, sources, inputIdentities, gpuBefore, gpuAfter: gpuState(), newOptimizationCount: 0,
      browserVersion: browser.version(), noUserTabsOrListeningService: true };
  } catch (error) {
    result = { status: "failed_verification", error: String(error), stack: error.stack, checks, sources, inputIdentities, newOptimizationCount: 0 }; throw error;
  } finally { await browser.close(); result.elapsedSeconds = (performance.now() - started) / 1000; publish(result);
    console.log(JSON.stringify({ status: result.status, rows: result.rows, elapsedSeconds: result.elapsedSeconds })); }
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
`;
const injection = `
source = once(source, 'const sources = sourceFiles.map(identity), inputIdentities = inputs.map(identity);',
  'sourceFiles.push(' + ${JSON.stringify(JSON.stringify(driver))} + ');\\nconst sources = sourceFiles.map(identity), inputIdentities = inputs.map(identity);');
source = once(source, '  return code;\\n}\\nasync function contextPage',
  '  code = replaceOnce(code, "renderer.render(scene, camera); state.frames++;", "renderer.render(scene, camera); state.frames++;" + ' + ${JSON.stringify(JSON.stringify(record))} + ');\\n  return code;\\n}\\nasync function contextPage');
const transientStart = source.indexOf("(async () => {\\n  const started = performance.now(), gpuBefore = gpuState();");
if (transientStart < 0) throw new Error("The original whole renderer entry changed");
source = source.slice(0, transientStart) + ${JSON.stringify(main)};
instance._compile(source, __filename);
`;
const program = once(fs.readFileSync(driver, "utf8"), "instance._compile(source, __filename);", injection);
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(program, __filename);
