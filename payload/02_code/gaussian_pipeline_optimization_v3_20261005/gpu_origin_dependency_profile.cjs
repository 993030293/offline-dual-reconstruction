// Profile the frozen R118 candidate with the original complete consumer workload.
const fs = require("fs"), path = require("path");
const CODE = __dirname;
const originalPath = path.join(CODE, "origin_dependency_run_003.cjs");
let source = fs.readFileSync(originalPath, "utf8");
function replace(before, after) {
  if (source.split(before).length !== 2) throw new Error("Original profile source anchor changed");
  source = source.replace(before, after);
}
source = source.replaceAll("origin_dependency_browser_003.mjs", "origin_dependency_browser_005.mjs");
replace('const owned = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v3_20261005/origin_dependency_001");',
  'const owned = path.join(ROOT, "04_outputs/gaussian_pipeline_optimization_v3_20261005/gpu_origin_dependency_001");');
replace('return [__filename, path.join(CODE, "origin_dependency.mjs"),',
  'return [__filename, path.join(CODE, "gpu_origin_dependency_profile.cjs"), path.join(CODE, "gpu_origin_dependency_002.mjs"), path.join(CODE, "origin_dependency.mjs"),');
replace('if (pathname === "/r115/origin_dependency.mjs" || pathname === "/r115/origin_dependency_browser_005.mjs") {',
  'if (pathname === "/r115/origin_dependency.mjs" || pathname === "/r115/origin_dependency_browser_005.mjs" || pathname === "/r118/frozen_origin_dependency.mjs") {');
replace('response.end(fs.readFileSync(path.join(CODE, path.basename(pathname)))); return;',
  'response.end(fs.readFileSync(path.join(CODE, pathname === "/r118/frozen_origin_dependency.mjs" ? "origin_dependency.mjs" : pathname === "/r115/origin_dependency.mjs" ? "gpu_origin_dependency_002.mjs" : path.basename(pathname)))); return;');
replace('mechanism: "R115_static_full_SH_radial_origin_dependency"',
  'mechanism: "R118_GPU_residency_static_full_SH_radial_origin_dependency_complete_screen"');
replace('minimumPhaseReduction: .1,', 'minimumPhaseReduction: .05,');
replace('campaignInstrumentationSeparate: ["resource_monitor", "campaign_summary_readback", "contract_freeze_and_generic"],',
  'campaignInstrumentationSeparate: [], diagnosticNotFormalAcceptance: true, originalSemanticsAndRecoveryRequired: true,');
replace('freshProcessPairs: 3,', 'freshProcessPairs: 3, diagnosticPairs: 1, formalPairsExecuted: 0,');
replace('status: "primary_generic_passed_supplemental_and_cost_pending",',
  'status: "complete_profile_finished_not_formal_acceptance",');
const first = source.indexOf("async function generic(contract) {");
const last = source.indexOf("async function main() {");
if (first < 0 || last <= first || source.lastIndexOf("async function generic(contract) {") !== first ||
    source.lastIndexOf("async function main() {") !== last) throw new Error("Original complete runner boundaries changed");
const originalBody = source.slice(first, last);
const stepsFirst = originalBody.indexOf('    const steps = [{ name: "initial", action: "none" }];');
const stepsLast = originalBody.indexOf('    save("GENERIC_PLAN.json",');
if (stepsFirst < 0 || stepsLast <= stepsFirst) throw new Error("Original complete workload boundaries changed");
const originalSteps = originalBody.slice(stepsFirst, stepsLast);
const profilingBody = `async function generic(contract) {
  let server, browser, activeSide = null;
  const sides = {}, completed = [], cleanup = [], processLedger = new Map();
  let current = null, currentMode = null, currentStage = "prepare";
  const sharedStart = now();
  function ownedBytes(folder) {
    let total = 0;
    for (const name of fs.readdirSync(folder)) {
      const target = path.join(folder, name), stat = fs.lstatSync(target);
      if (stat.isSymbolicLink()) throw new Error("An owned profile output link is unsupported");
      total += stat.isDirectory() ? ownedBytes(target) : stat.size;
    }
    return total;
  }
  function processes() {
    const rows = JSON.parse(execFileSync(PYTHON, ["-B", "-c",
      "import psutil,sys,json; p=psutil.Process(int(sys.argv[1])); rows=[]; \\nfor x in [p]+p.children(recursive=True):\\n try: rows.append({'pid':x.pid,'creation_time':x.create_time()})\\n except psutil.NoSuchProcess: pass\\nprint(json.dumps(rows))", String(process.pid)],
      { encoding: "utf8", windowsHide: true }));
    for (const row of rows) processLedger.set(row.pid + ":" + row.creation_time, row);
  }
  async function closeBrowser(mode, side) {
    let restored = null, restoreError = null;
    processes();
    try { if (side) restored = await side.page.evaluate(() => window.box2Viewer.originHarness.restore()); }
    catch (error) { restoreError = { name: error.name, message: error.message }; }
    for (const context of browser.contexts()) await context.close();
    await browser.close(); browser = null; activeSide = null;
    const row = { mode, browserCloseAwaited: true, contextCloseAwaited: true, restored, restoreError };
    cleanup.push(row);
    save("PROFILE_RESOURCE_CLOSE.json", { cleanup, processLedger: [...processLedger.values()],
      serverCloseAwaited: false, noExistingTabsOrServicesTouched: true, counted: false });
    if (restoreError || !restored || restored.contextBindings !== 0 || !restored.workerObserversRestored ||
        restored.GPUdrainFailure !== null || !restored.nativeMeshPrototypePreserved) {
      throw new Error("The actual owned profile cleanup guard failed");
    }
  }
${originalSteps}
  save("PROFILE_PLAN.json", { steps, originalStepsCompleteAndUnchanged: true,
    exactBinaryParts: ["pixels", "GPUordering", "workerOrdering", "GPUpackedPoints", "camera", "initialFullSHAndPacked"],
    freshBrowserOnEachSide: true, modesInOrder: ["native", "candidate"],
    screeningOnly: true, formalPairsExecuted: 0, counted: false });
  const setupBeforeServer = now() - sharedStart;
  let serverSetup = 0, comparisonSeconds = 0, serverCloseSeconds = 0;
  try {
    const beforeBudget = ownedBytes(owned);
    if (beforeBudget + 5 * 1024 ** 3 > 30 * 1024 ** 3) {
      throw new Error("The complete fixed profile would exceed the original owned output budget");
    }
    save("PROFILE_OUTPUT_BUDGET.json", { beforeBudgetBytes: beforeBudget, reservedBytes: 5 * 1024 ** 3,
      maximumOwnedBytes: 30 * 1024 ** 3, originalStepsUnchanged: true, counted: false });
    const beforeServer = now(); server = await makeServer(); serverSetup = now() - beforeServer;
    for (const mode of ["native", "candidate"]) {
      currentMode = mode; currentStage = "source_and_input_validation";
      const beforeLeaf = now(); validateIdentity(contract); resources();
      currentStage = "browser_and_original_asset_load";
      browser = await chromium.launch({ headless: true, args: ["--use-angle=d3d11"] });
      const side = await openSide(browser, server.url, mode);
      activeSide = side;
      processes();
      const rows = [], warmReadySeconds = now() - beforeLeaf;
      let actionSeconds = 0, captureSeconds = 0, resourceAndPublicationSeconds = 0;
      for (const step of steps) {
        if (now() > 1800) throw new Error("The original 1800 second node budget was exceeded");
        current = step; currentStage = "actual_action";
        let before = now(); await act(side, step.action, step.index); const actionElapsed = now() - before;
        currentStage = "actual_full_GPU_capture";
        before = now(); const row = await capture(side, mode + "_" + step.name); const captureElapsed = now() - before;
        actionSeconds += actionElapsed; captureSeconds += captureElapsed;
        before = now(); rows.push({ step, row, actionElapsed, captureElapsed, resources: resources() });
        save("PROFILE_" + mode.toUpperCase() + "_STATE.json", { current, rows, counted: false });
        resourceAndPublicationSeconds += now() - before;
      }
      currentStage = "actual_GPU_and_browser_close";
      const beforeClose = now(); await closeBrowser(mode, side); const closeSeconds = now() - beforeClose;
      sides[mode] = { rows, warmReadySeconds, actionSeconds, captureSeconds,
        resourceAndPublicationSeconds, closeSeconds, wholeLeafSeconds: now() - beforeLeaf, hardware: side.gpu };
      save("PROFILE_" + mode.toUpperCase() + "_LEAF.json", sides[mode]);
    }
    currentStage = "complete_paired_bytes_comparison";
    const beforeComparison = now();
    for (let i = 0; i < steps.length; i++) {
      const a = sides.native.rows[i], b = sides.candidate.rows[i];
      if (JSON.stringify(a.step) !== JSON.stringify(b.step)) throw new Error("An original complete step differs");
      completed.push({ step: steps[i], check: compare(a.row, b.row) });
    }
    comparisonSeconds = now() - beforeComparison;
    currentStage = "server_close";
    const beforeServerClose = now(); await new Promise(resolve => server.server.close(resolve));
    serverCloseSeconds = now() - beforeServerClose; server = null;
    const feeEach = (setupBeforeServer + serverSetup + comparisonSeconds + serverCloseSeconds) / 2;
    const native = sides.native.wholeLeafSeconds + feeEach;
    const candidate = sides.candidate.wholeLeafSeconds + feeEach;
    save("PROFILE_RESULT.json", { status: "complete_fresh_pair_profile_bytes_and_cleanup_passed_not_formal_acceptance",
      originalCompleteSteps: steps.length, completed, sides, commonMeasuredSeconds: {
        setupBeforeServer, serverSetup, comparisonSeconds, serverCloseSeconds },
      commonMeasuredFeeEach: feeEach, rawCompleteConsumerSeconds: { native, candidate },
      rawCompleteConsumerReduction: 1 - candidate / native,
      rawActionSeconds: { native: sides.native.actionSeconds, candidate: sides.candidate.actionSeconds },
      oldAndNewFirstFeesNotYetAllocated: true, remainingProtectionsAndThreeFormalPairsPending: true,
      noFormalCostAcceptance: true, formalPairsExecuted: 0, counted: false });
  } catch (error) {
    save("PROFILE_FAILURE.json", { status: "actual_profile_failed", currentMode, currentStage, current,
      errorType: error.name, error: error.message, stack: error.stack, completed,
      actualDifference: error.actualDifference, kind: error.kind ?? "not_reclassified",
      contingent: "not_run_after_first_required_profile_failure", counted: false });
    throw error;
  } finally {
    if (browser) {
      try { await closeBrowser(currentMode, activeSide); }
      catch (error) { cleanup.push({ mode: currentMode, incompleteRequiredCleanup: String(error) }); }
    }
    if (server) {
      const before = now(); await new Promise(resolve => server.server.close(resolve));
      serverCloseSeconds += now() - before;
    }
    save("PROFILE_RESOURCE_CLOSE.json", { cleanup, processLedger: [...processLedger.values()],
      serverCloseAwaited: true, noExistingTabsOrServicesTouched: true, elapsedSeconds: now(),
      ownedOutputBytes: ownedBytes(owned), originalMaximumOwnedBytes: 30 * 1024 ** 3, counted: false });
  }
}
`;
replace(originalBody, profilingBody);
if (process.argv.includes("--source-preparation-only")) {
  new Function("require", "__filename", "__dirname", source);
  console.log(JSON.stringify({ sourceAnchorsPrepared: true, completeOriginalWorkloadPreserved: true,
    actualGPUOracleRun: false, formalPairsExecuted: 0, counted: false }));
} else {
  new Function("require", "__filename", "__dirname", source)(require, originalPath, CODE);
}
