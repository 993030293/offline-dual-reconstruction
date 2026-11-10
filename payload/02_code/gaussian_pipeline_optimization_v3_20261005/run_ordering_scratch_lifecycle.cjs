// Preserve full original recovery and error checks with the R164 executed sources and native uploads.
const fs = require("fs"), path = require("path"), Module = require("module");
const original = path.join(__dirname, "run_ordering_prefix_lifecycle_v2.cjs");
function once(source, before, after) { if (source.split(before).length !== 2) throw new Error("The R164 lifecycle source boundary changed: " + before.slice(0, 80)); return source.replace(before, after); }
let source = fs.readFileSync(original, "utf8");
const edits = [];
edits.push('source = once(source, \'"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup006/R163"\', \'"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup007/R164"\');');
edits.push('source = source.replaceAll("c0/", "c1/");');
edits.push('source = source.replaceAll("ordering_active_prefix.js", "ordering_scratch_native_v2.js").replaceAll("installOrderingActivePrefix", "installOrderingUploadCost");');
edits.push('source = once(source, "const sources = [__filename,", "const sources = [__filename, " + ' + JSON.stringify(JSON.stringify(original) + ", " + ["run_ordering_prefix_lifecycle.cjs", "ordering_scratch_pool.js", "ordering_upload_cost.js", "gl_resource_generation_v2.js"].map(n => JSON.stringify(path.join(__dirname, n))).join(", ") + ", ") + ');');
edits.push('source = once(source, \'const body = relative.endsWith("/ordering_scratch_native_v2.js") ?\', ' + JSON.stringify('const body = ' + ["ordering_scratch_pool.js", "ordering_upload_cost.js", "gl_resource_generation_v2.js"].map(n => 'relative.endsWith("/' + n + '") ? fs.readFileSync(' + JSON.stringify(path.join(__dirname, n)) + ') : ').join("") + 'relative.endsWith("/ordering_scratch_native_v2.js") ?') + ');');
edits.push('source = once(source, "row.initialImageAlias && row.uploadedRows === row.requestedRows", "row.initialImageAlias && row.rows === row.textureHeight && row.submittedBytes === row.rows * 65536");');
edits.push('source = once(source, "disposed.timing.orderingUpload.originalAliasRestored && disposed.dag.nativeUpdateUnchanged", "disposed.timing.orderingUpload.originalAliasRestored && disposed.timing.generation.originalAliasesRestored && disposed.timing.orderingScratch.originalWorkerAliasRestored && disposed.dag.nativeUpdateUnchanged");');
edits.push('source = once(source, \'await page.evaluate(() => window.box2Viewer.renderer.forceContextRestore());\', \'await page.evaluate(() => new Promise(resolve => requestAnimationFrame(resolve)));\\n    await page.evaluate(() => window.box2Viewer.renderer.forceContextRestore());\');');
edits.push('source = once(source, \'save("FAILED_CONTEXT_" + caseTag + ".json", { error: String(error), result, errors });\', \'let actualState; try { actualState = await page.evaluate(() => window.box2Viewer.costProbe.snapshot()); } catch (e) { actualState = { error: String(e) }; }\\n    save("FAILED_CONTEXT_" + caseTag + ".json", { error: String(error), result, errors, actualState });\');');
edits.push('source = once(source, \'for (const kind of ["fault", "cancel"])\', \'for (const kind of ["fault", "workerFault", "cancel"])\');');
const workerFault = `    } else if (kind === "workerFault") {
      await page.evaluate(() => {
        const s = window.box2Viewer, worker = s.orderingDiagnostics.spark.sortWorker, originalCall = worker.call;
        s.workerFault = { armed: true, calls: 0 };
        const wrapped = function (...args) {
          if (s.workerFault.armed && args[0] === "sortSplats32") {
            s.workerFault.calls++; return originalCall.apply(this, ["R164_unknown_native_RPC", args[1]]);
          }
          return originalCall.apply(this, args);
        };
        worker.call = wrapped;
        s.restoreWorkerFault = () => { if (worker.call !== wrapped) throw new Error("The actual worker fault alias changed"); worker.call = originalCall; return true; };
      });
      await page.evaluate(() => window.box2Viewer.setView(406));
      await page.waitForFunction(() => window.box2Viewer.errors.some(e => e.includes("Unknown worker RPC: R164_unknown_native_RPC")), undefined, { timeout: 10000, polling: "raf" });
      const observed = await page.evaluate(() => ({ fault: { ...window.box2Viewer.workerFault }, uiErrors: window.box2Viewer.errors.slice(), snapshot: window.box2Viewer.costProbe.snapshot() }));
      await page.waitForFunction(() => window.box2Viewer.costProbe.snapshot().active === 0, undefined, { timeout: 10000 });
      const faultAliasRestored = await page.evaluate(() => window.box2Viewer.restoreWorkerFault()), restored = await closeAliases(page);
      if (observed.fault.calls !== 1 || !faultAliasRestored || !errors.some(e => e.includes("Unknown worker RPC: R164_unknown_native_RPC")) || !restored.timing.orderingScratch.originalWorkerAliasRestored || restored.timing.orderingScratch.activeLease !== null) throw new Error("The actual native worker callback error was swallowed or retained its scratch lease");
      result = { enabled, kind, completedViews, before, observed, errors, restored, faultAliasRestored,
        failed_original_context_not_repaired_or_claimed_reusable: true };
    } else {
      await page.evaluate(() => window.box2Viewer.setView(406));`;
edits.push('source = once(source, ' + JSON.stringify('    } else {\n      await page.evaluate(() => window.box2Viewer.setView(406));') + ', ' + JSON.stringify(workerFault) + ');');
edits.push('source = once(source, \'actual_two_GL_callback_errors_propagated_and_restored:\', \'actual_two_native_worker_callback_errors_propagated_and_restored: mode !== "lifecycle" || cases.filter(row => row.kind === "workerFault").length === 2 && cases.filter(row => row.kind === "workerFault").every(row => row.faultAliasRestored && row.observed.fault.calls === 1),\\n        actual_two_GL_callback_errors_propagated_and_restored:\');');
edits.push('source = once(source, \'fs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });\', \'fs.copyFileSync(path.join(RELEASE, "shared-build/vendor/SPARK-LICENSE.txt"), path.join(output, "sources/SPARK-LICENSE.txt"));\\nfs.writeFileSync(path.join(output, "sources/EXECUTED_VIEWER.js"), viewer, { flag: "wx" });\');');
source = once(source, 'const instance = new Module(__filename, module);', edits.join("\n") + '\nconst instance = new Module(__filename, module);');
const instance = new Module(__filename, module); instance.filename = __filename;
instance.paths = Module._nodeModulePaths(__dirname); instance._compile(source, __filename);
