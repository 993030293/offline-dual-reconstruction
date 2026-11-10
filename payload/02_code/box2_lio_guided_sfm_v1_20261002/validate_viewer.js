async (page) => {
  const artifactRoot = "__ARTIFACT_ROOT__";
  const failures = [];
  const badResponses = [];
  page.on("pageerror", error => failures.push(String(error)));
  page.on("response", response => { if (response.status() >= 400) badResponses.push({ url: response.url(), status: response.status() }); });
  const state = () => page.evaluate(() => ({
    ready: window.box2Viewer.ready, errors: window.box2Viewer.errors,
    currentView: window.box2Viewer.currentView, playing: window.box2Viewer.playing,
    sceneMode: window.box2Viewer.sceneMode, frames: window.box2Viewer.frames,
    full: window.box2Viewer.fullGaussianCount, box: window.box2Viewer.boxGaussianCount,
    position: window.box2Viewer.getCameraPosition()
  }));
  const distance = (a, b) => Math.hypot(...a.map((value, index) => value - b[index]));
  const requireCheck = (condition, message) => { if (!condition) throw new Error(message); };
  await page.waitForFunction(() => window.box2Viewer?.ready, undefined, { timeout: 45000 });
  const initial = await state();
  requireCheck(initial.full === 529371 && initial.box > 1000, "Weak-prior model counts do not match the trained model");
  await page.mouse.move(600, 380);
  await page.mouse.down({ button: "left" });
  await page.mouse.move(830, 360, { steps: 8 });
  await page.mouse.up({ button: "left" });
  await page.waitForTimeout(400);
  const rotated = await state();
  requireCheck(distance(initial.position, rotated.position) > .01, "Left-button rotation did not move the camera");
  await page.mouse.move(600, 380);
  await page.mouse.down({ button: "right" });
  await page.mouse.move(680, 420, { steps: 8 });
  await page.mouse.up({ button: "right" });
  await page.waitForTimeout(400);
  const panned = await state();
  requireCheck(distance(rotated.position, panned.position) > .01, "Right-button panning did not move the camera");
  await page.mouse.wheel(0, -300);
  await page.waitForTimeout(500);
  const zoomed = await state();
  requireCheck(distance(panned.position, zoomed.position) > .01, "Wheel zoom did not move the camera");
  await page.getByRole("button", { name: "重置视角", exact: true }).click();
  const reset = await state();
  requireCheck(reset.currentView === 90, "Reset did not restore the weak-prior default view");
  await page.getByRole("button", { name: "播放采集路径", exact: true }).click();
  await page.waitForTimeout(700);
  const playing = await state();
  requireCheck(playing.playing && playing.currentView !== reset.currentView, "Path playback did not advance");
  await page.getByRole("button", { name: "暂停播放", exact: true }).click();
  requireCheck(!(await state()).playing, "Playback did not pause");
  const slider = page.getByRole("slider", { name: "采集视角", exact: true });
  await slider.focus();
  await slider.press("End");
  requireCheck((await state()).currentView === 179, "Slider did not select the last view");
  await slider.press("Home");
  requireCheck((await state()).currentView === 0, "Slider did not select the first view");
  await page.getByLabel("展示范围").selectOption("scene");
  requireCheck((await state()).sceneMode === "scene", "Full-scene switch did not apply");
  await page.getByLabel("展示范围").selectOption("box");
  requireCheck((await state()).sceneMode === "box", "ROI switch did not apply");
  const cases = [{ case: "B_w_sigma015", ...initial }];
  for (const [caseId, expectedCount, expectedLast] of [["C_augmented_212", 543436, 211], ["A1_visual", 530280, 179]]) {
    await page.getByLabel("对照分支").selectOption(caseId);
    await page.waitForURL(`**/cases/${caseId}/index.html`, { timeout: 15000 });
    const visibleText = await page.locator("body").innerText();
    requireCheck(visibleText.includes(caseId.startsWith("C") ? "212" : "180"), "Case label and count are absent");
    await page.waitForFunction(() => window.box2Viewer?.ready, undefined, { timeout: 45000 });
    const loaded = await state();
    requireCheck(loaded.full === expectedCount && loaded.errors.length === 0, "Case loading or model count failed");
    requireCheck(Number(await page.getByRole("slider", { name: "采集视角" }).getAttribute("max")) === expectedLast, "Case path length is wrong");
    await page.getByLabel("展示范围").selectOption("box");
    await page.waitForTimeout(200);
    await page.screenshot({ path: `${artifactRoot}/${caseId}.png`, scale: "css" });
    cases.push({ case: caseId, ...loaded });
  }
  requireCheck(failures.length === 0 && badResponses.length === 0, "New browser runtime or HTTP errors occurred");
  return { status: "passed", cases, checks: ["three_models_loaded", "case_dropdown_navigation", "left_rotation", "right_pan", "wheel_zoom",
    "reset_default_index", "path_play_and_pause", "slider_first_and_last", "full_scene_and_ROI_switch"],
    measured_movement_m: { rotation: distance(initial.position, rotated.position), pan: distance(rotated.position, panned.position),
      zoom: distance(panned.position, zoomed.position) },
    page_errors_after_asset_repair: failures, HTTP_errors_after_asset_repair: badResponses,
    boundary: "browser_render_and_interaction_self_check_not_geometry_accuracy_or_independent_review" };
}
