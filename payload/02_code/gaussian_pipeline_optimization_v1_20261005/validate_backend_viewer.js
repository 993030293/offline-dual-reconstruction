async (page) => {
  const initialUrl = new URL(page.url());
  const base = initialUrl.origin;
  const run = initialUrl.searchParams.get("validationRun") || "p6b";
  const candidate = initialUrl.searchParams.get("validationCandidate") || "gsplat";
  if (!/^[A-Za-z0-9_]+$/.test(run) || !["gsplat", "target_density"].includes(candidate)) throw new Error("Unsupported validation configuration");
  const root = "C:/lidar-camera-fusion-mask-robust-downsampling/output/playwright/gaussian_optimization_"+run;
  const checks = [], captures = [], errors = [], projectionErrors = [];
  page.on("pageerror", error => errors.push(String(error)));
  const check = (condition, name) => { checks.push({ name, passed: Boolean(condition) }); if (!condition) throw new Error(name); };
  const state = () => page.evaluate(() => ({
    count: window.box2Viewer.fullGaussianCount, boxCount: window.box2Viewer.boxGaussianCount,
    position: window.box2Viewer.getCameraPosition(), direction: window.box2Viewer.getCameraDirection(),
    mode: window.box2Viewer.projectionMode, calibration: window.box2Viewer.calibration,
    matrix: window.box2Viewer.projectionMatrix, viewport: window.box2Viewer.viewport,
    playing: window.box2Viewer.playing, currentView: window.box2Viewer.currentView,
    frames: window.box2Viewer.frames, errors: window.box2Viewer.errors
  }));
  const distance = (a, b) => Math.hypot(...a.map((x, i) => x - b[i]));
  function projectionError(s) {
    const k = s.calibration, p = s.matrix;
    let maximum = 0;
    for (const u of [24, k.cx, k.width - 24]) for (const v of [24, k.cy, k.height - 24]) {
      const point = [(u-k.cx)*2/k.fx, -(v-k.cy)*2/k.fy, -2, 1];
      const clip = [0, 1, 2, 3].map(row => point.reduce((sum, x, col) => sum+p[col*4+row]*x, 0));
      maximum = Math.max(maximum, Math.hypot((clip[0]/clip[3]+1)/2*s.viewport.width-u*s.viewport.width/k.width,
        (1-clip[1]/clip[3])/2*s.viewport.height-v*s.viewport.height/k.height));
    }
    return maximum;
  }
  for (const label of ["reference", candidate]) {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto(base+"/cases/"+label+"/index.html");
    await page.waitForFunction(() => window.box2Viewer?.ready, undefined, { timeout: 60000 });
    const metadata = await (await page.request.get(base+"/cases/"+label+"/display_manifest.json")).json();
    const initial = await state();
    check(initial.count === metadata.full_gaussian_count && initial.boxCount === metadata.box_gaussian_count,
      label+"_complete_native_model_population");
    check(initial.mode === "calibrated_capture", label+"_initial_calibrated_capture");
    const error = projectionError(initial);
    projectionErrors.push({ label, mode: "desktop", error });
    check(error < 1e-3, label+"_full_K_projection");
    await page.getByLabel("移动方式").selectOption("fly");
    await page.locator("canvas").focus();
    let before = await state();
    await page.keyboard.down("w"); await page.waitForTimeout(450); await page.keyboard.up("w");
    let after = await state();
    check(distance(before.position, after.position) > .05, label+"_flight_forward");
    before = after;
    await page.keyboard.down("e"); await page.waitForTimeout(350); await page.keyboard.up("e");
    after = await state();
    check(after.position[2] > before.position[2]+.04 && Math.hypot(after.position[0]-before.position[0], after.position[1]-before.position[1]) < 1e-5,
      label+"_world_Z_up");
    await page.waitForTimeout(400);
    check(distance(after.position, (await state()).position) < 1e-5, label+"_flight_release");
    for (const mode of ["scene", "box"]) {
      await page.getByLabel("展示范围").selectOption(mode);
      for (const view of ["capture", "top", "drone"]) {
        const title = view === "capture" ? "重置采集视角" : view === "top" ? "无人机俯视" : "无人机斜俯视";
        await page.getByRole("button", { name: title, exact: true }).click();
        await page.waitForTimeout(1200);
        const bins = await page.evaluate(() => {
          const renderer = window.box2Viewer.renderer, canvas = renderer.domElement;
          const gl = renderer.getContext(), pixels = new Uint8Array(canvas.width*canvas.height*4), used = new Uint8Array(4096);
          gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
          for (let i = 0; i < pixels.length; i += 4) used[((pixels[i]>>4)<<8)|((pixels[i+1]>>4)<<4)|(pixels[i+2]>>4)] = 1;
          return used.reduce((a, b) => a+b, 0);
        });
        check(bins >= 16, label+"_nonempty_"+mode+"_"+view);
        const current = await state();
        if (view === "top") check(current.direction[2] < -.99, label+"_"+mode+"_top_orientation");
        if (view === "drone") check(current.direction[2] < -.5, label+"_"+mode+"_oblique_orientation");
        const file = root+"/"+label+"_"+mode+"_"+view+".png";
        await page.locator("canvas").screenshot({ path: file });
        captures.push(file);
      }
    }
    await page.getByLabel("移动方式").selectOption("orbit");
    await page.getByRole("button", { name: "重置采集视角", exact: true }).click();
    before = await state();
    await page.mouse.move(500, 400); await page.mouse.down(); await page.mouse.move(620, 450, { steps: 6 }); await page.mouse.up();
    await page.waitForTimeout(350);
    check(distance(before.position, (await state()).position) > .01 && (await state()).mode === "free_camera", label+"_orbit_rotation");
    before = await state();
    await page.mouse.move(500, 400); await page.mouse.down({ button: "right" }); await page.mouse.move(580, 440, { steps: 6 }); await page.mouse.up({ button: "right" });
    await page.waitForTimeout(350);
    check(distance(before.position, (await state()).position) > .01, label+"_orbit_pan");
    before = await state(); await page.mouse.wheel(0, -220); await page.waitForTimeout(350);
    check(distance(before.position, (await state()).position) > .01, label+"_orbit_zoom");
    await page.getByRole("button", { name: "播放采集路径", exact: true }).click();
    before = await state(); await page.waitForTimeout(500);
    check((await state()).playing && (await state()).frames > before.frames+5, label+"_path_play");
    await page.getByRole("button", { name: "暂停播放", exact: true }).click();
    await page.getByRole("slider", { name: "采集视角", exact: true }).press("End");
    check((await state()).currentView === 211, label+"_all212_views");
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForFunction(() => innerWidth === 390 && window.box2Viewer.viewport.width <= 390);
    const mobileError = projectionError(await state());
    projectionErrors.push({ label, mode: "mobile", error: mobileError });
    check(mobileError < 1e-3, label+"_resize_full_K");
    check(!(await state()).errors.length, label+"_no_viewer_errors");
  }
  check(!errors.length, "no_browser_runtime_errors");
  return { status: "passed_native_PLY_and_free_navigation", checks, projectionErrors, captures, errors,
    visual_self_review_pending: true, seed: 0, independent_freeview_quality_reference: false, counted: false };
}
