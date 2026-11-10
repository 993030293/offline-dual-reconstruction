// Use the installed browser for native packing and the final composed display acceptance.
const fs = require("fs"), path = require("path"), vm = require("vm"), crypto = require("crypto");
const installed = "C:/Users/99303/git/playwright/node_modules/playwright";
const { chromium } = require(installed);
const root = path.resolve("04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release");
const mode = process.argv[2];
if (!["pack", "validate"].includes(mode)) throw new Error("A declared native browser operation is required");
const script = mode === "pack" ? path.resolve("02_code/gaussian_pipeline_optimization_v1_20261005/export_new_route_packed.js") : path.join(__dirname,"validate_effective_viewer.js");
const receipts = mode === "pack" ? root : path.resolve(process.argv[3]);
if (mode === "validate" && (!process.argv[3] || !receipts.startsWith(path.resolve("04_outputs/gaussian_pipeline_optimization_v2_20261005")+path.sep))) throw new Error("An isolated current-route browser receipt directory is required");
const statePath = path.join(receipts, mode === "pack" ? "PACKING_STATE.json" : "BROWSER_STATE.json");
const outputPath = path.join(receipts, mode === "pack" ? "PACKING_VALIDATION.json" : "BROWSER_VALIDATION.json");
const identity = filename => ({path:filename,bytes:fs.statSync(filename).size,sha256:crypto.createHash("sha256").update(fs.readFileSync(filename)).digest("hex")});
const sources = [identity(__filename),identity(script),identity(path.join(installed,"package.json"))];
const publish = value => {fs.writeFileSync(statePath+".tmp",JSON.stringify(value,null,2));fs.renameSync(statePath+".tmp",statePath);};
const replaceOnce = (source,before,after) => {if(source.split(before).length!==2)throw new Error("Pinned native browser packing anchor changed");return source.replace(before,after);};
(async()=>{
  if (!fs.existsSync(root)||fs.existsSync(statePath)||fs.existsSync(outputPath))throw new Error("A unique owned final browser receipt is required");
  if(mode==="validate")fs.mkdirSync(receipts,{recursive:false});
  let code=fs.readFileSync(script,"utf8");
  if(mode==="pack") {
    code=replaceOnce(code,'initial.port !== "8203"','initial.port !== "8210"');
    code=replaceOnce(code,'const root = "C:/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v1_20261005/new_route_viewer_001";', 'const root = '+JSON.stringify(root.replaceAll("\\","/"))+';');
    code=replaceOnce(code,'["reference", "combined"]','["baseline", "combined"]');
    code=replaceOnce(code,'initial.origin + "/" + model','initial.origin + "/release/" + model');
  }
  const started=Date.now(),browser=await chromium.launch({headless:true,args:["--use-angle=d3d11"]});
  const context=await browser.newContext({viewport:{width:1440,height:900},deviceScaleFactor:1}),page=await context.newPage();
  publish({status:"running",bound_sources:sources,mode,browser_version:browser.version(),existing_user_tabs_untouched:true,maximum_seconds:900});
  globalThis.saveViewerProgress=value=>publish({...value,status:"running",bound_sources:sources,elapsed_seconds:(Date.now()-started)/1000});
  try {
    await page.goto("http://127.0.0.1:8210/release/baseline/cases/packed/export.html");
    globalThis.effectiveViewerOutput=receipts.replaceAll("\\","/");
    const operation=vm.runInThisContext("("+code+")",{filename:script});
    const result=await operation(page);
    if(sources[0].sha256!==identity(__filename).sha256||sources[1].sha256!==identity(script).sha256)throw new Error("Final browser operation source changed");
    Object.assign(result,{bound_sources:sources,browser_version:browser.version(),elapsed_seconds:(Date.now()-started)/1000,existing_user_tabs_untouched:true,new_optimization_count:0});
    fs.writeFileSync(outputPath,JSON.stringify(result,null,2));publish(result);
    console.log(JSON.stringify({status:result.status,checks:result.checks?.length,elapsed_seconds:result.elapsed_seconds}));
  } catch(error) {publish({status:"failed",bound_sources:sources,error:String(error),stack:error.stack,elapsed_seconds:(Date.now()-started)/1000});throw error;}
  finally {delete globalThis.saveViewerProgress;delete globalThis.effectiveViewerOutput;await context.close();await browser.close();}
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
