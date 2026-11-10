// Verify visible controls and actual camera movement in an isolated browser.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require(process.argv[2]);
const url = process.argv[3], output = process.argv[4];
fs.mkdirSync(output, { recursive: false });
const rows = [], errors = [];

(async () => {
  let browser;
  try {
    browser = await chromium.launch({ channel: 'msedge', headless: true,
      args: ['--disable-background-timer-throttling', '--disable-renderer-backgrounding'] });
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForFunction(() => window.box2Viewer?.ready &&
      window.box2Viewer.firstReadyRenderMs != null && window.box2Viewer.renderer.info.render.calls > 0,
      null, { timeout: 90000 });
    const snapshot = async () => page.evaluate(() => {
      const state = window.box2Viewer;
      return { ready: state.ready, errors: state.errors, position: state.getCameraPosition(),
        direction: state.getCameraDirection(), frames: state.frames, navigationMode: state.navigationMode,
        sceneMode: state.sceneMode, projectionMode: state.projectionMode, currentView: state.currentView,
        playing: state.playing, loadedModels: state.loadedModels, fullGaussianCount: state.fullGaussianCount,
        drawingCalls: state.renderer.info.render.calls, firstReadyRenderMs: state.firstReadyRenderMs };
    });
    const changed = async (before, field) => {
      await page.waitForFunction(({ before, field }) => {
        const state = window.box2Viewer;
        const values = field === 'position' ? state.getCameraPosition() : state.getCameraDirection();
        return state.frames > before.frames && values.some((value, i) => Math.abs(value-before[field][i]) > 1e-6);
      }, { before, field }, { timeout: 5000 });
      const after = await snapshot();
      assert(after.ready && after.errors.length === 0 && after.drawingCalls > 0);
      rows.push({ action: field + '_changed', before, after });
      return after;
    };
    const canvas = page.locator('canvas');
    const rect = await canvas.boundingBox();
    assert(rect && rect.width > 0 && rect.height > 0);
    const x = rect.x+rect.width*.42, y = rect.y+rect.height*.52;
    rows.push({ action: 'initial_complete_scene_rendered', state: await snapshot() });
    await page.screenshot({ path: path.join(output, '01_capture_scene.png') });
    let before = await snapshot();
    await page.mouse.move(x, y); await page.mouse.down();
    await page.mouse.move(x+95, y+35, { steps: 12 }); await page.mouse.up();
    await changed(before, 'direction');
    before = await snapshot();
    await page.mouse.move(x, y); await page.mouse.down({ button: 'right' });
    await page.mouse.move(x+65, y-25, { steps: 10 }); await page.mouse.up({ button: 'right' });
    await changed(before, 'position');
    before = await snapshot();
    await page.mouse.move(x, y); await page.mouse.wheel(0, -220);
    await changed(before, 'position');
    await page.locator('#reset').click();
    await page.locator('#scene-mode').selectOption('box');
    await page.waitForFunction(() => window.box2Viewer.ready && window.box2Viewer.sceneMode === 'box' &&
      window.box2Viewer.loadedModels.includes('box'), null, { timeout: 45000 });
    rows.push({ action: 'complete_focus_asset_loaded', state: await snapshot() });
    await page.waitForTimeout(500);
    await page.screenshot({ path: path.join(output, '02_capture_focus.png') });
    await page.locator('#scene-mode').selectOption('scene');
    await page.waitForFunction(() => window.box2Viewer.ready && window.box2Viewer.sceneMode === 'scene');
    await page.locator('#navigation-mode').selectOption('fly');
    await canvas.click({ position: { x: rect.width*.42, y: rect.height*.52 } });
    before = await snapshot();
    await page.keyboard.down('w'); await page.waitForTimeout(900); await page.keyboard.up('w');
    let after = await changed(before, 'position');
    assert.equal(after.navigationMode, 'fly');
    assert(after.position.reduce((sum, value, i) => sum+(value-before.position[i])*before.direction[i], 0) > 0);
    before = await snapshot();
    await page.keyboard.down('e'); await page.waitForTimeout(600); await page.keyboard.up('e');
    after = await changed(before, 'position');
    assert(after.position[2] > before.position[2]);
    await page.locator('[data-preset="drone"]').click();
    after = await snapshot();
    assert.equal(after.navigationMode, 'fly');
    assert.equal(after.projectionMode, 'free_camera');
    rows.push({ action: 'drone_free_camera', state: after });
    await page.waitForTimeout(600);
    await page.screenshot({ path: path.join(output, '03_drone_scene.png') });
    await page.locator('#reset').click();
    before = await snapshot();
    await page.locator('#play').click();
    await page.waitForFunction(index => window.box2Viewer.playing &&
      window.box2Viewer.currentView !== index, before.currentView, { timeout: 5000 });
    rows.push({ action: 'capture_path_playing', state: await snapshot() });
    await page.locator('#play').click();
    assert.equal((await snapshot()).playing, false);
    await page.locator('#navigation-mode').selectOption('orbit');
    await page.locator('#reset').click();
    after = await snapshot();
    assert(after.position.every(Number.isFinite) && after.direction.every(Number.isFinite));
    assert.equal(after.errors.length, 0);
    assert.equal(errors.length, 0);
    fs.writeFileSync(path.join(output, 'RESULT.json'), JSON.stringify({ status: 'visible_controls_and_rendering_verified',
      url, rows, final: after, pageErrors: errors, strict_novel_view_quality_claim: false,
      visual_review_pending: true }, null, 2)+'\n');
    console.log(JSON.stringify({ status: 'visible_controls_and_rendering_verified', checks: rows.length }));
  } catch (error) {
    fs.writeFileSync(path.join(output, 'FAILURE.json'), JSON.stringify({ status: 'failed_visible_control_verification',
      error: String(error.stack??error), rows, pageErrors: errors }, null, 2)+'\n');
    process.exitCode = 1;
  } finally {
    if (browser) await browser.close();
  }
})();
