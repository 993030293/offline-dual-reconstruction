async (page) => {
  const checks = [], measurements = {}, errors = [];
  page.on("pageerror", error => errors.push(String(error)));
  const requireCheck = (condition, label) => { if (!condition) throw new Error(label); checks.push(label); };
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.waitForFunction(() => window.box2Viewer?.ready, undefined, { timeout: 60000 });
  const state = () => page.evaluate(() => {
    const s = window.box2Viewer;
    return { ready: s.ready, mode: s.projectionMode, matrix: s.projectionMatrix,
      calibration: s.calibration, viewport: s.viewport, position: s.getCameraPosition(),
      direction: s.getCameraDirection(), errors: s.errors, keys: s.getActiveKeys(),
      sceneMode: s.sceneMode, navigationMode: s.navigationMode, currentView: s.currentView,
      playing: s.playing, count: s.fullGaussianCount };
  });
  function projectionError(s) {
    const k = s.calibration, p = s.matrix;
    let maximum = 0;
    for (const u of [24, k.cx, k.width - 24]) for (const v of [24, k.cy, k.height - 24]) {
      const point = [(u - k.cx) * 2 / k.fx, -(v - k.cy) * 2 / k.fy, -2, 1];
      const clip = [0, 1, 2, 3].map(row => point.reduce((sum, x, col) => sum + p[col * 4 + row] * x, 0));
      const x = (clip[0] / clip[3] + 1) / 2 * s.viewport.width;
      const y = (1 - clip[1] / clip[3]) / 2 * s.viewport.height;
      maximum = Math.max(maximum, Math.hypot(x - u * s.viewport.width / k.width, y - v * s.viewport.height / k.height));
    }
    return maximum;
  }
  const first = await state();
  requireCheck(first.count === 465294 && first.mode === "calibrated_capture", "full_model_calibrated_capture");
  measurements.capture_projection_error_px = projectionError(first);
  requireCheck(measurements.capture_projection_error_px < 1e-3, "full_K_browser_matrix_oracle");
  requireCheck(first.viewport.width <= 1440 && first.viewport.height <= 900 && first.viewport.left > 0, "capture_letterbox");
  const distance = (a, b) => Math.hypot(...a.map((x, i) => x - b[i]));
  await page.getByLabel("移动方式").selectOption("fly");
  let before = await state();
  requireCheck(before.mode === "free_camera" && before.viewport.width === 1440, "free_camera_restores_full_viewport");
  await page.keyboard.down("w"); await page.waitForTimeout(450); await page.keyboard.up("w");
  let after = await state();
  measurements.forward_m = distance(before.position, after.position);
  requireCheck(measurements.forward_m > .05, "free_flight_forward");
  before = after;
  await page.keyboard.down("e"); await page.waitForTimeout(350); await page.keyboard.up("e");
  after = await state();
  requireCheck(after.position[2] > before.position[2] + .04 && Math.hypot(after.position[0] - before.position[0], after.position[1] - before.position[1]) < 1e-5, "world_Z_up");
  await page.waitForTimeout(200);
  requireCheck(distance(after.position, (await state()).position) < 1e-5, "key_release_stops");
  await page.getByRole("button", { name: "无人机俯视", exact: true }).click();
  requireCheck((await state()).direction[2] < -.99, "drone_top");
  await page.getByRole("button", { name: "无人机斜俯视", exact: true }).click();
  requireCheck((await state()).direction[2] < -.5, "drone_oblique");
  await page.getByLabel("展示范围").selectOption("box");
  requireCheck((await state()).sceneMode === "box", "target_switch");
  await page.getByLabel("展示范围").selectOption("scene");
  await page.getByLabel("移动方式").selectOption("orbit");
  await page.getByRole("button", { name: "重置采集视角", exact: true }).click();
  requireCheck((await state()).mode === "calibrated_capture", "capture_reset_reapplies_full_K");
  before = await state();
  await page.mouse.move(500, 400); await page.mouse.down(); await page.mouse.move(620, 460, { steps: 6 }); await page.mouse.up();
  await page.waitForTimeout(250);
  requireCheck((await state()).mode === "free_camera" && distance(before.position, (await state()).position) > .01, "orbit_rotation_and_projection_transition");
  before = await state();
  await page.mouse.move(500, 400); await page.mouse.down({ button: "right" }); await page.mouse.move(580, 440, { steps: 6 }); await page.mouse.up({ button: "right" });
  await page.waitForTimeout(250);
  requireCheck(distance(before.position, (await state()).position) > .01, "orbit_pan");
  before = await state(); await page.mouse.wheel(0, -220); await page.waitForTimeout(250);
  requireCheck(distance(before.position, (await state()).position) > .01, "orbit_zoom");
  await page.getByRole("button", { name: "播放采集路径", exact: true }).click();
  await page.waitForTimeout(400);
  requireCheck((await state()).playing && (await state()).mode === "calibrated_capture", "path_play_uses_calibration");
  await page.getByRole("button", { name: "暂停播放", exact: true }).click();
  const slider = page.getByRole("slider", { name: "采集视角", exact: true });
  await slider.focus(); await slider.press("End");
  requireCheck((await state()).currentView === 211, "all_212_views_available");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForFunction(() => window.box2Viewer.viewport.width <= 390);
  measurements.mobile_projection_error_px = projectionError(await state());
  requireCheck(measurements.mobile_projection_error_px < 1e-3, "resize_preserves_full_K");
  await page.getByRole("button", { name: "收起控制面板", exact: true }).click();
  requireCheck(await page.getByRole("button", { name: "展开控制面板", exact: true }).isVisible(), "mobile_panel");
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.waitForFunction(() => window.box2Viewer.viewport.height === 900);
  await page.getByRole("button", { name: "展开控制面板", exact: true }).click();
  await page.getByLabel("对照分支").selectOption("legacy");
  await page.waitForURL("**/cases/legacy/index.html");
  await page.waitForFunction(() => window.box2Viewer?.ready, undefined, { timeout: 60000 });
  requireCheck((await state()).count === 536728 && projectionError(await state()) < 1e-3, "legacy_model_same_full_K_viewer");
  await page.getByLabel("对照分支").selectOption("full");
  await page.waitForURL("**/cases/full/index.html");
  await page.waitForFunction(() => window.box2Viewer?.ready, undefined, { timeout: 60000 });
  requireCheck(!errors.length && !(await state()).errors.length, "no_runtime_errors");
  return { status: "passed_projection_and_navigation", checks, measurements, errors,
    scope: "matrix and actual controls; rendered free-view visual quality reviewed separately",
    existing_8195_service_modified: false };
}
