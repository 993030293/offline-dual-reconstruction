// Own one installed browser process tree for bounded native renderer validation.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const crypto = require("crypto");
const installed = "C:/Users/99303/git/playwright/node_modules/playwright";
const { chromium } = require(installed);
const script = path.join(__dirname, "validate_viewer_execution.js");
const output = path.resolve("04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_002");
const statePath = path.join(output, "BROWSER_STATE.json");
const identity = filename => ({ path: filename, bytes: fs.statSync(filename).size,
  sha256: crypto.createHash("sha256").update(fs.readFileSync(filename)).digest("hex") });
const sources = [identity(__filename), identity(script), identity(path.join(installed,"package.json"))];
const publish = value => {
  fs.writeFileSync(statePath+".tmp", JSON.stringify(value,null,2)); fs.renameSync(statePath+".tmp",statePath);
};

(async()=>{
  if (!fs.existsSync(output) || fs.existsSync(statePath)) throw new Error("A unique packaged browser campaign is required");
  const started = Date.now();
  const browser = await chromium.launch({headless:true,args:["--use-angle=d3d11"]});
  const context = await browser.newContext({viewport:{width:1440,height:900},deviceScaleFactor:1});
  const page = await context.newPage();
  publish({status:"running",bound_sources:sources,browser_version:browser.version(),installed_runtime:installed,
    existing_user_tabs_untouched:true,maximum_seconds:900,maximum_RSS_GiB:8,counted:false});
  globalThis.saveViewerProgress = value => publish({...value,status:"running",bound_sources:sources,elapsed_seconds:(Date.now()-started)/1000});
  try {
    const operation = vm.runInThisContext("("+fs.readFileSync(script,"utf8")+")",{filename:script});
    const result = await operation(page);
    if (sources[1].sha256 !== identity(script).sha256 || sources[0].sha256 !== identity(__filename).sha256) throw new Error("The actual browser campaign source changed");
    Object.assign(result,{bound_sources:sources,browser_version:browser.version(),elapsed_seconds:(Date.now()-started)/1000,
      existing_user_tabs_untouched:true,visual_self_review_pending:true});
    fs.writeFileSync(path.join(output,"BROWSER_RESULT.json"),JSON.stringify(result,null,2)); publish(result);
    console.log(JSON.stringify({status:result.status,acceptable_mechanisms:result.acceptable_mechanisms,
      failures:result.failures,elapsed_seconds:result.elapsed_seconds}));
  } catch(error) {
    publish({status:"failed",sources,error:String(error),stack:error.stack,elapsed_seconds:(Date.now()-started)/1000,counted:false});
    throw error;
  } finally {
    delete globalThis.saveViewerProgress;
    await context.close(); await browser.close();
  }
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
