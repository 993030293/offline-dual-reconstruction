async (page) => {
  const base = "http://127.0.0.1:8203";
  const root = "C:/lidar-camera-fusion-mask-robust-downsampling/output/playwright/gaussian_new_route_001";
  const browser = page.context().browser(), checks = [], captures = [], pixels = [], errors = [];
  const check = (value, name) => { checks.push({ name, passed: Boolean(value) }); if (!value) throw new Error(name); };
  const distance = (a, b) => Math.hypot(...a.map((x, i) => x - b[i]));
  const state = p => p.evaluate(() => ({ position: window.box2Viewer.getCameraPosition(),
    direction: window.box2Viewer.getCameraDirection(), mode: window.box2Viewer.sceneMode,
    playing: window.box2Viewer.playing, frames: window.box2Viewer.frames,
    view: window.box2Viewer.currentView, errors: window.box2Viewer.errors }));
  const settle = async (p, mode) => {
    await p.getByLabel("展示范围").selectOption(mode);
    await p.waitForFunction(m => window.box2Viewer.sceneMode === m && !window.box2Viewer.modelLoading, mode);
    await p.waitForTimeout(600);
  };
  try {
    await page.goto("about:blank");
    for (const label of ["reference", "combined"]) {
      const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
      const p = await context.newPage();
      p.on("pageerror", error => errors.push(String(error)));
      try {
        await p.goto(base + "/" + label + "/cases/packed/index.html?mode=box");
        await p.bringToFront();
        await p.waitForFunction(() => window.box2Viewer?.ready, undefined, { timeout: 60000 });
        const first = await p.evaluate(() => ({ loaded: window.box2Viewer.loadedModels,
          attempts: window.box2Viewer.modelLoadAttempts, filter: window.box2Viewer.renderFilter }));
        check(first.loaded.join() === "box" && first.attempts.scene === 0, label + "_only_requested_box_loaded");
        check(first.filter.preBlurAmount === .3 && first.filter.blurAmount === 0, label + "_classic_filter_contract");
        const pathChecks = await p.evaluate(async () => {
          const views = await (await fetch("./camera_path.json")).json();
          let projection = 0, position = 0, direction = 0;
          for (let i = 0; i < views.length; i++) {
            window.box2Viewer.setView(i);
            const s = window.box2Viewer, k = s.calibration, m = s.projectionMatrix, v = views[i];
            position = Math.max(position, Math.hypot(...s.getCameraPosition().map((x, j) => x - v.position[j])));
            direction = Math.max(direction, Math.hypot(...s.getCameraDirection().map((x, j) => x - v.forward[j])));
            for (const u of [24, k.cx, k.width - 24]) for (const y of [24, k.cy, k.height - 24]) {
              const point = [(u - k.cx) * 2 / k.fx, -(y - k.cy) * 2 / k.fy, -2, 1];
              const clip = [0, 1, 2, 3].map(row => point.reduce((sum, value, col) => sum + m[col * 4 + row] * value, 0));
              projection = Math.max(projection, Math.hypot((clip[0] / clip[3] + 1) / 2 * s.viewport.width - u * s.viewport.width / k.width,
                (1 - clip[1] / clip[3]) / 2 * s.viewport.height - y * s.viewport.height / k.height));
            }
          }
          return { count: views.length, projection, position, direction };
        });
        check(pathChecks.count === 1016 && pathChecks.position < 1e-9 && pathChecks.direction < 1e-9
          && pathChecks.projection < 1e-3, label + "_all1016_calibrated_camera_transforms");
        await p.getByLabel("移动方式").selectOption("fly");
        for (const mode of ["box", "scene"]) {
          await settle(p, mode);
          for (const [view, title] of [["capture", "重置采集视角"], ["top", "无人机俯视"], ["drone", "无人机斜俯视"]]) {
            await p.getByRole("button", { name: title, exact: true }).click();
            await p.waitForTimeout(900);
            const current = await state(p);
            if (view === "top") check(current.direction[2] < -.99, label + "_" + mode + "_top_direction");
            if (view === "drone") check(current.direction[2] < -.5, label + "_" + mode + "_drone_direction");
            const image = await p.evaluate(() => {
              const r = window.box2Viewer.renderer, c = r.domElement, gl = r.getContext();
              const data = new Uint8Array(c.width * c.height * 4), bins = new Uint8Array(4096);
              gl.readPixels(0, 0, c.width, c.height, gl.RGBA, gl.UNSIGNED_BYTE, data);
              for (let i = 0; i < data.length; i += 4) bins[((data[i] >> 4) << 8) | ((data[i + 1] >> 4) << 4) | (data[i + 2] >> 4)] = 1;
              return { width: c.width, height: c.height, bins: bins.reduce((a, b) => a + b, 0) };
            });
            check(image.bins >= 16, label + "_" + mode + "_" + view + "_nonempty");
            pixels.push({ label, mode, view, ...image });
            const file = root + "/" + label + "_" + mode + "_" + view + ".png";
            await p.locator("canvas").screenshot({ path: file }); captures.push(file);
          }
        }
        const counts = await p.evaluate(() => ({ full: window.box2Viewer.fullGaussianCount, box: window.box2Viewer.boxGaussianCount }));
        const assets = await (await p.request.get(base + "/" + label + "/ASSET_MANIFEST.json")).json();
        check(counts.full === assets.scene.numSplats && counts.box === assets.box.numSplats, label + "_complete_model_population");
        await p.getByRole("button", { name: "重置采集视角", exact: true }).click();
        await p.locator("canvas").focus();
        let a = await state(p); await p.keyboard.down("w"); await p.waitForTimeout(350); await p.keyboard.up("w");
        let b = await state(p); check(distance(a.position, b.position) > .04, label + "_fly_forward");
        a = b; await p.keyboard.down("e"); await p.waitForTimeout(350); await p.keyboard.up("e"); b = await state(p);
        check(b.position[2] > a.position[2] + .04 && Math.hypot(b.position[0] - a.position[0], b.position[1] - a.position[1]) < 1e-5, label + "_world_Z_up");
        for (const key of ["s", "a", "d", "q"]) {
          a = await state(p); await p.keyboard.down(key); await p.waitForTimeout(350); await p.keyboard.up(key); b = await state(p);
          check(distance(a.position, b.position) > .04, label + "_fly_key_" + key);
          if (key === "q") check(b.position[2] < a.position[2] - .04, label + "_world_Z_down");
        }
        await p.waitForTimeout(500); a = await state(p); await p.waitForTimeout(500); b = await state(p);
        check(distance(a.position, b.position) < 1e-8, label + "_released_key_stops_flight");
        check(b.frames === a.frames, label + "_idle_demand_rendering");
        await p.getByLabel("移动方式").selectOption("orbit");
        await p.getByRole("button", { name: "重置采集视角", exact: true }).click();
        for (const [name, button, dx] of [["rotate", "left", 100], ["pan", "right", 80]]) {
          a = await state(p); await p.mouse.move(500, 400); await p.mouse.down({ button });
          await p.mouse.move(500 + dx, 460, { steps: 6 }); await p.mouse.up({ button }); await p.waitForTimeout(400);
          check(distance(a.position, (await state(p)).position) > .01, label + "_orbit_" + name);
        }
        a = await state(p); await p.mouse.wheel(0, -200); await p.waitForTimeout(400);
        check(distance(a.position, (await state(p)).position) > .01, label + "_orbit_zoom");
        await p.getByRole("button", { name: "播放采集路径", exact: true }).click(); a = await state(p); await p.waitForTimeout(600); b = await state(p);
        check(b.playing && b.frames > a.frames + 5, label + "_path_playback");
        await p.getByRole("button", { name: "暂停播放", exact: true }).click();
        const slider = p.getByRole("slider", { name: "采集视角", exact: true }); await slider.focus(); await slider.press("End");
        await p.waitForTimeout(100); check((await state(p)).view === 1015, label + "_last_of1016_path_views");
        await p.setViewportSize({ width: 390, height: 844 });
        await p.waitForFunction(() => innerWidth === 390 && window.box2Viewer.viewport.width <= 390);
        const mobileProjectionError = await p.evaluate(() => {
          const s = window.box2Viewer, k = s.calibration, m = s.projectionMatrix;
          let error = 0;
          for (const u of [24, k.cx, k.width - 24]) for (const v of [24, k.cy, k.height - 24]) {
            const point = [(u - k.cx) * 2 / k.fx, -(v - k.cy) * 2 / k.fy, -2, 1];
            const clip = [0, 1, 2, 3].map(row => point.reduce((sum, value, col) => sum + m[col * 4 + row] * value, 0));
            error = Math.max(error, Math.hypot((clip[0] / clip[3] + 1) / 2 * s.viewport.width - u * s.viewport.width / k.width,
              (1 - clip[1] / clip[3]) / 2 * s.viewport.height - v * s.viewport.height / k.height));
          }
          return error;
        });
        check(mobileProjectionError < 1e-3, label + "_mobile_calibrated_resize");
        check(!(await state(p)).errors.length, label + "_no_runtime_errors");
        checks.push({ name: label + "_path_projection_evidence", passed: true, ...pathChecks });
      } finally { await context.close(); }
    }
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
    let sceneRequests = 0;
    await context.route("**/scene.spk", async route => { sceneRequests++; return sceneRequests === 1 ? route.abort("failed") : route.continue(); });
    const p = await context.newPage();
    try {
      await p.goto(base + "/combined/cases/packed/index.html?mode=box");
      await p.waitForFunction(() => window.box2Viewer?.ready);
      await p.getByLabel("展示范围").selectOption("scene");
      await p.waitForFunction(() => window.box2Viewer.loadFailures.length === 1);
      check((await state(p)).mode === "box", "failed_scene_retains_box");
      await p.getByRole("button", { name: "重试加载", exact: true }).click();
      await p.waitForFunction(() => window.box2Viewer.sceneMode === "scene" && !window.box2Viewer.modelLoading);
      check(sceneRequests === 2, "actual_model_retry_success");
      await p.getByLabel("对照分支").selectOption("reference");
      await p.waitForURL("**/reference/cases/packed/index.html");
      await p.waitForFunction(() => window.box2Viewer?.ready);
      check(true, "actual_control_branch_switch");
    } finally { await context.close(); }
    check(errors.length === 0, "no_uncaught_errors");
    return { status: "browser_checks_passed_visual_review_pending", checks, captures, pixels, errors,
      scope: "new_route_display_compatibility_and_navigation_not_independent_geometry_accuracy", counted: false };
  } catch (error) {
    return { status: "failed_browser_validation", error: String(error.stack ?? error), checks, captures, pixels, errors, counted: false };
  }
}
