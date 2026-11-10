// Keep the first recovery audit and compare identical preloaded navigation states explicitly.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "run_ordering_prefix_lifecycle.cjs");
function once(source, before, after) { if (source.split(before).length !== 2) throw new Error("The staged restoration audit boundary changed: " + before.slice(0, 70)); return source.replace(before, after); }
let source = fs.readFileSync(original, "utf8");
source = once(source, 'const sources = [__filename,', 'const sources = [__filename, ' + JSON.stringify(original) + ',');
source = once(source, 'const context = await browser.newContext({ viewport: { width: 960, height: 600 }, deviceScaleFactor: 1 }), page = await context.newPage(), errors = [];', 'const context = await browser.newContext({ viewport: { width: 960, height: 600 }, deviceScaleFactor: 1 }), page = await context.newPage(), errors = [], warnings = [];\n  page.on("console", message => { if (["warning", "error"].includes(message.type())) warnings.push(message.text()); });');
source = once(source, 'return { context, page, errors };', 'return { context, page, errors, warnings };');
source = once(source, 'async function contextCase(browser, enabled) {', 'async function contextCase(browser, enabled, control = null, loseContext = true) {\n  const caseTag = loseContext ? Number(enabled) : "NO_LOSS_REFERENCE";');
source = once(source, 'const { context, page, errors } = await setup(browser, enabled); let result;', 'const { context, page, errors, warnings } = await setup(browser, enabled); let result;');
source = once(source, '    await step(page, "preloss_switch_box",', '    const initialCapture = await capture(page);\n    await step(page, "preloss_switch_box",');
source = once(source, '    await step(page, "preloss_box203",', '    const preLossPose0 = await step(page, "preloss_box0", () => page.evaluate(() => window.box2Viewer.setView(0)));\n    await step(page, "preloss_box203",');
source = once(source, 'const before = await page.evaluate(async () => {', 'const before = await page.evaluate(async loseContext => {');
source = once(source, 'const snapshot = s.costProbe.snapshot(); s.renderer.forceContextLoss(); return snapshot;\n    });', 'const snapshot = s.costProbe.snapshot(); if (loseContext) s.renderer.forceContextLoss(); return snapshot;\n    }, loseContext);');
source = once(source, '    await page.waitForFunction(() => window.box2Viewer.actualContextLost', '    if (loseContext) {\n    await page.waitForFunction(() => window.box2Viewer.actualContextLost');
source = once(source, '    const restoredFirst = await step(page,', '    }\n    const restoredFirst = await step(page,');
source = once(source, 'const views = await allFixed(page), reference = prior.episodes.find(row => !row.hooks);', 'const views = await allFixed(page);\n    const priorReference = prior.episodes.find(row => !row.hooks);\n    const populations = JSON.parse(fs.readFileSync(path.join(RELEASE, "baseline/ASSET_MANIFEST.json"), "utf8"));\n    const expectedViews = control ? control.views : priorReference.views.map(row => ({ ...row, capture: { ...row.capture, count: [populations.scene.numSplats, populations.box.numSplats] } }));');
source = once(source, 'row.name === reference.views[i].name && exact(row.capture, reference.views[i].capture)', 'row.name === expectedViews[i].name && exact(row.capture, expectedViews[i].capture)');
source = once(source, 'const firstExpected = reference.views.find(row => row.name === "box_capture_0").capture;', 'const firstExpected = control ? control.restoredFirst.capture : preLossPose0.capture;');
source = once(source, 'result = { enabled, before, restoredFirst, restoration, views, disposed, errors, checks: {', 'result = { enabled, loseContext, initialCapture, preLossPose0, before, restoredFirst, restoration, views, disposed, errors, warnings, checks: {\n      original_initial_scene_only_full_camera_RGBA_exact: exact(initialCapture, priorReference.first),');
source = once(source, 'actual_loss_and_restore_events: restoration.actualContextLost && restoration.actualContextRestored,', 'actual_loss_and_restore_events: !loseContext || restoration.actualContextLost && restoration.actualContextRestored,');
source = once(source, 'actual_restoration_initial_upload_full_image_alias: restoration.snapshot', 'actual_restoration_initial_upload_full_image_alias: !loseContext || restoration.snapshot');
source = once(source, 'save("CONTEXT_" + Number(enabled)', 'save("CONTEXT_" + caseTag');
source = once(source, 'save("FAILED_CONTEXT_" + Number(enabled)', 'save("FAILED_CONTEXT_" + caseTag');
source = once(source, 'if (mode === "context") for (const enabled of [false, true]) cases.push(await contextCase(browser, enabled));', 'if (mode === "context") { const control = await contextCase(browser, false, null, false);\n        for (const enabled of [false, true]) cases.push(await contextCase(browser, enabled, control, true)); }');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
