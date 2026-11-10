// Prepare the unchanged native display assets in an isolated hidden browser.
const fs = require('node:fs');
const path = require('node:path');
const { performance } = require('node:perf_hooks');
const { chromium } = require(process.argv[2]);
const url = process.argv[3], output = process.argv[4];
const started = performance.now();
const budgetMs = Number(process.argv[5] ?? 360000);
if (!Number.isFinite(budgetMs) || budgetMs <= 0 || budgetMs > 360000)
  throw new Error('Native packing must retain the original total time budget');
function remaining(cap) {
  const value = Math.min(cap, budgetMs - (performance.now() - started));
  if (value <= 0) throw new Error('The original native packing budget expired');
  return value;
}
(async () => {
  let browser, page, native, viewer, result, failure;
  let phase = 'launch', domLoaded = false, browserClosed = false;
  const errors = [], consoleErrors = [], failedRequests = [], badResponses = [];
  const safeURL = value => {
    const item = new URL(value);
    return item.protocol + '//' + item.host + item.pathname;
  };
  try {
    browser = await chromium.launch({ channel: 'msedge', headless: true,
      args: ['--disable-background-timer-throttling', '--disable-renderer-backgrounding'] });
    const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
    page = await context.newPage();
    page.on('pageerror', error => errors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') consoleErrors.push(message.text()); });
    page.on('requestfailed', request => failedRequests.push({url: safeURL(request.url()),
      error: request.failure()?.errorText ?? null}));
    page.on('response', response => { if (response.status() >= 400)
      badResponses.push({url: safeURL(response.url()), status: response.status()}); });
    phase = 'packing';
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: remaining(30000) });
    const button = page.getByRole('button', { name: '生成两份显示资产', exact: true });
    if (await button.count() !== 1 || !(await button.isVisible())) throw new Error('The native packing control is unavailable');
    fs.writeFileSync(path.join(output, 'PACKING_DOM.txt'), await page.locator('body').innerText());
    await button.click();
    await page.waitForFunction(() => {
      const state = document.getElementById('status');
      return state?.dataset.complete === 'true' || state?.dataset.failed === 'true';
    }, null, { timeout: remaining(180000) });
    const status = await page.locator('#status').innerText();
    if (await page.locator('#status').getAttribute('data-complete') !== 'true') throw new Error(status);
    native = JSON.parse(status);
    if (native.status !== 'both_native_assets_published' || native.records.length !== 2 ||
        native.records.some(row => !row.complete_native_arrays_equal)) throw new Error('Complete native asset validation failed');
    const packingSeconds = (performance.now()-started)/1000;
    phase = 'full_scene_initialization';
    await page.goto(url.replace('export.html', 'index.html'), { waitUntil: 'domcontentloaded', timeout: remaining(30000) });
    domLoaded = true;
    await page.waitForFunction(() => window.box2Viewer?.ready || window.box2Viewer?.errors?.length,
      null, { timeout: remaining(90000) });
    viewer = await page.evaluate(() => ({ ready: window.box2Viewer.ready,
      errors: window.box2Viewer.errors, fullGaussianCount: window.box2Viewer.fullGaussianCount,
      loadedModels: window.box2Viewer.loadedModels }));
    if (!viewer.ready || viewer.errors.length || viewer.fullGaussianCount !== native.records[0].numSplats || errors.length)
      throw new Error('Native viewer failed complete full-scene loading: '+JSON.stringify({viewer,errors}));
    result = { status: 'native_assets_and_viewer_ready',
      native, viewer, pageErrors: errors, browser_inclusive_packing_seconds: packingSeconds,
      elapsed_seconds: (performance.now()-started)/1000, isolated_browser: true,
      browser_dependency: process.argv[2], failedRequests, badResponses, consoleErrors };
  } catch (error) {
    let snapshot = null;
    if (page) {
      try { snapshot = await page.evaluate(() => ({documentReadyState: document.readyState,
        viewer: window.box2Viewer ? {ready: window.box2Viewer.ready, errors: window.box2Viewer.errors} : null})); }
      catch (snapshotError) { snapshot = {snapshotError: String(snapshotError)}; }
    }
    failure = { status: 'failed_native_packing_or_viewer', phase, domLoaded, snapshot,
      error: String(error.stack??error), pageErrors: errors, consoleErrors, failedRequests, badResponses,
      native_assets_validated: native?.status === 'both_native_assets_published' && native.records?.length === 2 &&
        native.records.every(row => row.complete_native_arrays_equal === true),
      native, elapsed_seconds: (performance.now()-started)/1000, root_cause_established: false };
    process.exitCode = 1;
  } finally {
    if (browser) {
      try { await browser.close(); browserClosed = true; }
      catch (closeError) {
        failure = {status: 'failed_native_packing_browser_close', phase,
          error: String(closeError), original_failure: failure, elapsed_seconds: (performance.now()-started)/1000};
        process.exitCode = 1;
      }
    }
    if (failure) {
      failure.browserClosed = browserClosed;
      fs.writeFileSync(path.join(output, 'FAILURE.json'), JSON.stringify(failure, null, 2)+'\n');
    } else {
      result.browserClosed = browserClosed;
      fs.writeFileSync(path.join(output, 'RESULT.json'), JSON.stringify(result, null, 2)+'\n');
      console.log(JSON.stringify({status: result.status, elapsed_seconds: result.elapsed_seconds}));
    }
  }
})();
