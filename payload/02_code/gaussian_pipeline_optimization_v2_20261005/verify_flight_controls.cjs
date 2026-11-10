// Verify all six free-flight keys through the final native UI without rerunning pixel benchmarks.
const fs=require("fs"),path=require("path"),crypto=require("crypto");
const {chromium}=require("C:/Users/99303/git/playwright/node_modules/playwright");
const output=path.resolve("04_outputs/gaussian_pipeline_optimization_v2_20261005/final_flight_controls_001");
const source={path:__filename,sha256:crypto.createHash("sha256").update(fs.readFileSync(__filename)).digest("hex")};
(async()=>{
  fs.mkdirSync(output,{recursive:false});
  const browser=await chromium.launch({headless:true,args:["--use-angle=d3d11"]});
  const result={status:"running",checks:[],rows:[],source,new_optimization_count:0};
  const check=(value,name)=>{result.checks.push({name,passed:Boolean(value)});if(!value)throw new Error(name);};
  const state=p=>p.evaluate(()=>({position:window.box2Viewer.getCameraPosition(),direction:window.box2Viewer.getCameraDirection(),keys:window.box2Viewer.getActiveKeys()}));
  const norm=a=>Math.hypot(...a),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),delta=(a,b)=>a.map((v,i)=>v-b[i]);
  try {
    for(const label of ["baseline","combined"]) {
      const context=await browser.newContext({viewport:{width:1440,height:900},deviceScaleFactor:1});
      const p=await context.newPage();
      try {
        await p.goto("http://127.0.0.1:8210/release/"+label+"/cases/packed/index.html?mode=box");
        await p.waitForFunction(()=>window.box2Viewer?.ready,undefined,{timeout:60000});
        await p.getByLabel("移动方式").selectOption("fly");
        const reset=async()=>{await p.getByRole("button",{name:"重置采集视角",exact:true}).click();await p.locator("canvas").focus();await p.waitForTimeout(120);};
        for(const key of ["w","s","a","d","q","e"]) {
          await reset();const before=await state(p);await p.keyboard.down(key);await p.waitForTimeout(250);
          const moving=await state(p);await p.keyboard.up(key);await p.waitForTimeout(220);const after=await state(p);
          const movement=delta(after.position,before.position),right=[before.direction[1],-before.direction[0],0];
          check(norm(movement)>.04,label+"_"+key+"_actual_motion");
          if(key==="w"||key==="s")check(dot(movement,before.direction)*(key==="w"?1:-1)>.02,label+"_"+key+"_correct_forward_backward");
          if(key==="a"||key==="d")check(dot(movement,right)*(key==="d"?1:-1)>.02,label+"_"+key+"_correct_left_right");
          if(key==="q"||key==="e")check(movement[2]*(key==="e"?1:-1)>.04&&Math.hypot(movement[0],movement[1])<1e-5,label+"_"+key+"_correct_world_vertical");
          const stopped=await state(p);await p.waitForTimeout(220);const later=await state(p);
          check(after.keys.length===0&&norm(delta(later.position,stopped.position))<1e-8,label+"_"+key+"_keyup_stops");
          result.rows.push({label,key,movement,active_keys_while_moving:moving.keys});
        }
        const travel=async(fast)=>{await reset();const before=await state(p);if(fast)await p.keyboard.down("Shift");
          await p.keyboard.down("w");await p.waitForTimeout(300);await p.keyboard.up("w");if(fast)await p.keyboard.up("Shift");
          await p.waitForTimeout(120);return norm(delta((await state(p)).position,before.position));};
        const normal=await travel(false),fast=await travel(true);check(fast>normal*1.5,label+"_actual_Shift_acceleration");
        await reset();await p.getByRole("slider",{name:"采集视角",exact:true}).focus();const before=await state(p);
        await p.keyboard.press("w");await p.waitForTimeout(220);check(norm(delta((await state(p)).position,before.position))<1e-8,label+"_focused_UI_does_not_start_flight");
      } finally {await context.close();}
    }
    check(result.rows.length===12,"both_models_all_six_keys_exercised");
    if(source.sha256!==crypto.createHash("sha256").update(fs.readFileSync(__filename)).digest("hex"))throw new Error("Flight validator source changed");
    result.status="both_current_models_all_WASD_QE_Shift_and_key_release_passed";
    fs.writeFileSync(path.join(output,"RESULT.json"),JSON.stringify(result,null,2));
    console.log(JSON.stringify({status:result.status,rows:result.rows.length,checks:result.checks.length}));
  } catch(error) {result.status="failed";result.error=String(error);fs.writeFileSync(path.join(output,"FAILURE.json"),JSON.stringify(result,null,2));throw error;}
  finally{await browser.close();}
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
