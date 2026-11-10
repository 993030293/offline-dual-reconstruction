// Check visible WebGL pixels without forcing a render or altering the model.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require(process.argv[2]);
const url = process.argv[3], output = process.argv[4];
fs.mkdirSync(output, { recursive: false });

(async () => {
  let browser;
  const rows = [], errors = [];
  try {
    browser = await chromium.launch({ channel: 'msedge', headless: true,
      args: ['--disable-background-timer-throttling', '--disable-renderer-backgrounding'] });
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForFunction(() => window.box2Viewer?.ready, null, { timeout: 45000 });
    await page.evaluate(() => {
      window.previewPixelSamples = [];
      window.inspectPreviewPixels = () => {
        const state = window.box2Viewer;
        const gl = state.renderer.getContext();
        const width = gl.drawingBufferWidth, height = gl.drawingBufferHeight;
        const pixels = new Uint8Array(width * height * 4);
        gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        const histogram = new Map();
        let total = 0;
        for (let y = 4; y < height; y += 8) {
          for (let x = 4; x < width; x += 8) {
            const i = 4 * (y * width + x);
            const key = ((pixels[i] >> 2) << 12) | ((pixels[i + 1] >> 2) << 6) | (pixels[i + 2] >> 2);
            histogram.set(key, (histogram.get(key) ?? 0) + 1);
            total++;
          }
        }
        const dominant = Math.max(...histogram.values());
        const observation = { elapsedSinceNavigationMs: performance.now(), frames: state.frames,
          sceneMode: state.sceneMode, currentView: state.currentView, projectionMode: state.projectionMode,
          loadedModels: state.loadedModels, drawingCalls: state.renderer.info.render.calls,
          width, height, sampledPixels: total, quantizedColors: histogram.size,
          nonDominantFraction: 1 - dominant / total,
          nonUniformImage: histogram.size >= 10 && 1 - dominant / total > 0.01 };
        window.previewPixelSamples.push(observation);
        return observation;
      };
    });
    const settled = async (label, filename) => {
      const first = await page.evaluate(() => window.inspectPreviewPixels());
      await page.waitForTimeout(400);
      await page.waitForFunction(() => window.inspectPreviewPixels().nonUniformImage,
        null, { timeout: 30000, polling: 250 });
      const final = await page.evaluate(() => window.inspectPreviewPixels());
      assert(final.nonUniformImage);
      await page.screenshot({ path: path.join(output, filename) });
      rows.push({ label, first, final });
    };
    await settled('default_full_scene_without_interaction', '01_settled_scene.png');
    await page.locator('#scene-mode').selectOption('box');
    await page.waitForFunction(() => window.box2Viewer.sceneMode === 'box');
    await settled('capture_focus', '02_settled_focus.png');
    await page.locator('#navigation-mode').selectOption('fly');
    await page.locator('[data-preset="drone"]').click();
    await page.waitForFunction(() => window.box2Viewer.projectionMode === 'free_camera');
    await settled('drone_focus', '03_settled_drone_focus.png');
    await page.locator('#scene-mode').selectOption('scene');
    await page.waitForFunction(() => window.box2Viewer.sceneMode === 'scene');
    await settled('drone_full_scene', '04_settled_drone_scene.png');
    assert.deepEqual(errors, []);
    const state = await page.evaluate(() => ({ errors: window.box2Viewer.errors,
      position: window.box2Viewer.getCameraPosition(), direction: window.box2Viewer.getCameraDirection(),
      fullGaussianCount: window.box2Viewer.fullGaussianCount, boxGaussianCount: window.box2Viewer.boxGaussianCount,
      pixelSamples: window.previewPixelSamples }));
    assert.deepEqual(state.errors, []);
    fs.writeFileSync(path.join(output, 'RESULT.json'), JSON.stringify({
      status: 'settled_nonuniform_WebGL_pixels_verified_visual_review_pending', url, rows, state,
      models_or_viewer_modified: false, formal_primary_clock_changed: false,
      firstReadyRenderMs_is_not_first_visible_pixel_time: true,
      strict_novel_view_quality_claim: false }, null, 2) + '\n');
    console.log(JSON.stringify({ status: 'settled_pixels_verified', rows }));
  } catch (error) {
    fs.writeFileSync(path.join(output, 'FAILURE.json'), JSON.stringify({ error: String(error.stack ?? error), rows, errors }, null, 2) + '\n');
    process.exitCode = 1;
  } finally {
    if (browser) await browser.close();
  }
})();
