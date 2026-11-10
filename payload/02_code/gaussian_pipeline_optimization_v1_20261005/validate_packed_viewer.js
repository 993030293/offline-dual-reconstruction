async (page) => {
  const root = "C:/lidar-camera-fusion-mask-robust-downsampling/output/playwright/gaussian_optimization_p10/002";
  const browser = page.context().browser();
  const checks = [], loading = [], pixelComparisons = [], captures = [], runtimeErrors = [];
  const controlPixels = new Map();
  const sourceManifest = await (await page.request.get("http://127.0.0.1:8199/ASSET_MANIFEST.json")).json();
  const requireCheck = (value, name) => { checks.push({ name, passed: Boolean(value) }); if (!value) throw new Error(name); };
  const pixelIdentity = p => p.evaluate(async () => {
    const renderer = window.box2Viewer.renderer, canvas = renderer.domElement, gl = renderer.getContext();
    const pixels = new Uint8Array(canvas.width * canvas.height * 4);
    gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    const digest = await crypto.subtle.digest("SHA-256", pixels);
    let nonzeroRgb = 0;
    const bins = new Uint8Array(4096);
    for (let i = 0; i < pixels.length; i += 4) {
      if (pixels[i] || pixels[i + 1] || pixels[i + 2]) nonzeroRgb++;
      bins[((pixels[i] >> 4) << 8) | ((pixels[i + 1] >> 4) << 4) | (pixels[i + 2] >> 4)] = 1;
    }
    return { width: canvas.width, height: canvas.height, nonzeroRgb, distinct_rgb_bins: bins.reduce((a, b) => a + b, 0),
      sha256: [...new Uint8Array(digest)].map(x => x.toString(16).padStart(2, "0")).join("") };
  });
  const state = p => p.evaluate(() => ({ frames: window.box2Viewer.frames,
    position: window.box2Viewer.getCameraPosition(), direction: window.box2Viewer.getCameraDirection(),
    mode: window.box2Viewer.projectionMode, playing: window.box2Viewer.playing,
    sceneMode: window.box2Viewer.sceneMode, currentView: window.box2Viewer.currentView,
    viewport: window.box2Viewer.viewport, errors: window.box2Viewer.errors }));
  const distance = (a, b) => Math.hypot(...a.map((x, i) => x - b[i]));
  async function changed(p, name, action) {
    const before = await state(p);
    await action();
    await p.waitForFunction(previous => window.box2Viewer.frames > previous, before.frames, { timeout: 5000 });
    checks.push({ name: name + "_renders", passed: true });
    return [before, await state(p)];
  }
  await page.goto("about:blank");
  for (let trial = 0; trial < 3; trial++) {
    const order = trial === 1 ? ["packed", "ply"] : ["ply", "packed"];
    for (const label of order) {
      const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
      const modelResources = [], pendingSizes = [];
      context.on("requestfinished", request => {
        if (!/\.(ply|spk)$/.test(request.url())) return;
        pendingSizes.push((async () => {
          const sizes = await request.sizes();
          const response = await request.response();
          modelResources.push({ name: request.url(),
            transfer_bytes: sizes.responseBodySize + sizes.responseHeadersSize,
            decoded_bytes: sizes.responseBodySize, status: response.status() });
        })());
      });
      const p = await context.newPage();
      p.on("pageerror", error => runtimeErrors.push(String(error)));
      try {
        const navigationStarted = Date.now();
        await p.goto("http://127.0.0.1:8199/cases/" + label + "/index.html");
        await p.bringToFront();
        await p.waitForFunction(() => window.box2Viewer?.ready, undefined, { timeout: 60000 });
        await Promise.all(pendingSizes);
        const measured = await p.evaluate(() => ({ initialization_ms: window.box2Viewer.modelInitializationMs,
          full_splats: window.box2Viewer.fullGaussianCount, box_splats: window.box2Viewer.boxGaussianCount }));
        measured.model_resources = modelResources;
        requireCheck(measured.full_splats === sourceManifest.scene.numSplats && measured.box_splats === sourceManifest.box.numSplats,
          label + "_exact_source_population_" + trial);
        loading.push({ trial, label, ...measured, navigation_to_ready_poll_ms: Date.now()-navigationStarted });
        requireCheck(measured.model_resources.length === 2 && measured.model_resources.every(row => row.status === 200 && row.transfer_bytes > 0 && row.decoded_bytes > 0), label + "_fresh_model_transfers_" + trial);
        await p.getByLabel("移动方式").selectOption("fly");
        await p.getByRole("button", { name: "重置采集视角", exact: true }).click();
        await p.waitForTimeout(2000);
        if (trial === 0) {
          for (const mode of ["scene", "box"]) {
            await p.getByLabel("展示范围").selectOption(mode);
            for (const view of ["capture", "top", "drone"]) {
              await p.getByRole("button", { name: view === "capture" ? "重置采集视角" : view === "top" ? "无人机俯视" : "无人机斜俯视", exact: true }).click();
              await p.waitForTimeout(1000);
              const identity = await pixelIdentity(p), key = mode + "_" + view;
              requireCheck(identity.nonzeroRgb > 5000 && identity.distinct_rgb_bins >= 16, label + "_nonempty_" + key);
              if (label === "ply") controlPixels.set(key, identity);
              else {
                const reference = controlPixels.get(key);
                const equal = reference.width === identity.width && reference.height === identity.height && reference.sha256 === identity.sha256;
                pixelComparisons.push({ key, reference, candidate: identity, equal });
                requireCheck(equal, "identical_pixels_" + key);
              }
              const file = root + "/" + label + "_" + key + ".png";
              await p.locator("canvas").screenshot({ path: file });
              captures.push(file);
            }
          }
        }
        if (label === "packed" && trial === 0) {
          let [a, b] = await changed(p, "wake_and_flight_forward", async () => {
            await p.locator("canvas").focus();
            await p.keyboard.down("w"); await p.waitForTimeout(450); await p.keyboard.up("w");
          });
          requireCheck(distance(a.position, b.position) > .05, "flight_forward_distance");
          [a, b] = await changed(p, "world_Z_up", async () => {
            await p.keyboard.down("e"); await p.waitForTimeout(350); await p.keyboard.up("e");
          });
          requireCheck(b.position[2] > a.position[2] + .04 && Math.hypot(b.position[0]-a.position[0], b.position[1]-a.position[1]) < 1e-5, "world_Z_up_distance");
          await p.waitForTimeout(400);
          const stopped = await state(p);
          await p.waitForTimeout(600);
          requireCheck(distance(stopped.position, (await state(p)).position) < 1e-8, "key_release_stops_motion");
          await changed(p, "background_toggle", () => p.getByRole("button", { name: "浅色背景", exact: true }).click());
          await changed(p, "grid_toggle", () => p.getByRole("button", { name: "显示参考网格", exact: true }).click());
          await changed(p, "scene_toggle", () => p.getByLabel("展示范围").selectOption("scene"));
          await p.getByLabel("移动方式").selectOption("orbit");
          await changed(p, "capture_reset", () => p.getByRole("button", { name: "重置采集视角", exact: true }).click());
          [a, b] = await changed(p, "orbit_rotate", async () => {
            await p.mouse.move(500, 400); await p.mouse.down(); await p.mouse.move(620, 460, { steps: 6 }); await p.mouse.up();
            await p.waitForTimeout(400);
          });
          requireCheck(distance(a.position, b.position) > .01 && b.mode === "free_camera", "orbit_rotate_moves_camera");
          [a, b] = await changed(p, "orbit_pan", async () => {
            await p.mouse.move(500, 400); await p.mouse.down({ button: "right" }); await p.mouse.move(580, 440, { steps: 6 }); await p.mouse.up({ button: "right" });
            await p.waitForTimeout(400);
          });
          requireCheck(distance(a.position, b.position) > .01, "orbit_pan_moves_camera");
          [a, b] = await changed(p, "orbit_zoom", async () => { await p.mouse.wheel(0, -220); await p.waitForTimeout(400); });
          requireCheck(distance(a.position, b.position) > .01, "orbit_zoom_moves_camera");
          await changed(p, "path_play", () => p.getByRole("button", { name: "播放采集路径", exact: true }).click());
          const playStart = await state(p);
          await p.waitForTimeout(600);
          requireCheck((await state(p)).frames > playStart.frames + 5 && (await state(p)).playing, "path_continues_without_input");
          await changed(p, "path_pause", () => p.getByRole("button", { name: "暂停播放", exact: true }).click());
          await changed(p, "path_slider", async () => {
            const slider = p.getByRole("slider", { name: "采集视角", exact: true }); await slider.focus(); await slider.press("End");
          });
          requireCheck((await state(p)).currentView === 211, "all_212_views_available");
          await changed(p, "resize", async () => {
            await p.setViewportSize({ width: 390, height: 844 });
            await p.waitForFunction(() => innerWidth === 390 && window.box2Viewer.viewport.width <= 390,
              undefined, { timeout: 5000 });
          });
          requireCheck((await state(p)).viewport.width <= 390, "resize_preserves_calibrated_viewport");
        }
        requireCheck(!(await state(p)).errors.length, label + "_no_viewer_errors_" + trial);
      } finally {
        await context.close();
      }
    }
  }
  const reductions = [0, 1, 2].map(trial => {
    const reference = loading.find(row => row.trial === trial && row.label === "ply");
    const candidate = loading.find(row => row.trial === trial && row.label === "packed");
    requireCheck(candidate.full_splats === reference.full_splats && candidate.box_splats === reference.box_splats, "no_model_population_loss_" + trial);
    const sumBytes = row => row.model_resources.reduce((sum, file) => sum + file.decoded_bytes, 0);
    requireCheck(1-sumBytes(candidate)/sumBytes(reference) >= .4, "model_transfer_reduction_" + trial);
    const reduction = 1 - candidate.initialization_ms / reference.initialization_ms;
    return reduction;
  });
  const median = values => [...values].sort((a, b) => a - b)[1];
  const medianReduction = 1 - median(loading.filter(row => row.label === "packed").map(row => row.initialization_ms)) /
    median(loading.filter(row => row.label === "ply").map(row => row.initialization_ms));
  requireCheck(medianReduction >= .1, "initialization_stage_median_at_least10percent_faster");
  requireCheck(!runtimeErrors.length, "no_browser_runtime_errors");
  return { status: "passed_packed_loading_and_navigation_pixel_checks", checks, loading, reductions, medianReduction,
    pixelComparisons, captures, runtimeErrors, full_machine_GPU_power_measured: false,
    visual_self_review_pending: true, counted: false };
}
