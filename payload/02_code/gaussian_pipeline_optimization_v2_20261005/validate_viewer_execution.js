async (page) => {
  const base = "http://127.0.0.1:8210";
  const output = "C:/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_002";
  const browser = page.context().browser();
  const result = { status: "running", checks: [], loading: [], pixels: [], controls: [], captures: [], failures: [],
    sources: await (await page.request.get(base + "/CONTRACT.json")).json(), counted: false,
    all_static_pixel_checks_are_native_viewer_equivalence_not_GT: true };
  const median = values => [...values].sort((a, b) => a-b)[Math.floor(values.length / 2)];
  const check = (value, name, mechanism) => {
    result.checks.push({ name, mechanism, passed: Boolean(value) });
    if (!value) throw new Error(name);
  };
  const sleep = (p, ms) => p.waitForTimeout(ms);
  const state = p => p.evaluate(() => {
    const s = window.box2Viewer;
    return { ready: s.ready, frames: s.frames, position: s.getCameraPosition(), direction: s.getCameraDirection(),
      errors: s.errors, projection: s.projectionMatrix, mode: s.projectionMode, playing: s.playing,
      currentView: s.currentView, motion: s.motionResolution, keys: s.getActiveKeys() };
  });
  const distance = (a, b) => Math.hypot(...a.map((value, i) => value-b[i]));
  const pixels = p => p.evaluate(async () => {
    const r = window.box2Viewer.renderer, c = r.domElement, gl = r.getContext();
    const data = new Uint8Array(c.width*c.height*4), bins = new Uint8Array(4096);
    gl.readPixels(0, 0, c.width, c.height, gl.RGBA, gl.UNSIGNED_BYTE, data);
    for (let i=0; i<data.length; i+=4) bins[((data[i]>>4)<<8)|((data[i+1]>>4)<<4)|(data[i+2]>>4)] = 1;
    const digest = await crypto.subtle.digest("SHA-256", data);
    return { width: c.width, height: c.height, bins: bins.reduce((a,b)=>a+b,0),
      sha256: [...new Uint8Array(digest)].map(value=>value.toString(16).padStart(2,"0")).join("") };
  });
  async function ready(p, label, mode="scene") {
    await p.goto(base+"/"+label+"/cases/packed/index.html?mode="+mode);
    await p.bringToFront();
    await p.waitForFunction(() => window.box2Viewer?.ready && window.box2Viewer?.firstReadyRenderMs != null,
      undefined, {timeout:60000});
    return p.evaluate(() => ({ initialization_ms: window.box2Viewer.modelInitializationMs,
      navigation_ms: window.box2Viewer.navigationToFirstReadyRenderMs,
      integrity: globalThis.assetIntegrityTrace, cache: globalThis.assetCacheTrace,
      full_splats: window.box2Viewer.fullGaussianCount, box_splats: window.box2Viewer.boxGaussianCount,
      actual_GL_renderer: (()=>{const gl=window.box2Viewer.renderer.getContext(),ext=gl.getExtension("WEBGL_debug_renderer_info");return ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):"unavailable";})(),
      loaded: window.box2Viewer.loadedModels, errors: window.box2Viewer.errors }));
  }
  async function fresh(label, mode, operation) {
    const context = await browser.newContext({viewport:{width:1440,height:900},deviceScaleFactor:1});
    const p = await context.newPage(), errors=[];
    p.on("pageerror",error=>errors.push(String(error)));
    try { const row = await ready(p,label,mode); await operation(p,row); check(!errors.length && !row.errors.length,label+"_native_runtime_clean", "shared"); }
    finally { await context.close(); }
  }
  await page.goto("about:blank");
  result.Q68_preserved_failure = "viewer_execution_001: parallel integrity scene90.6 to94.2ms; not rerun or counted";
  const repeatFailedSpeedBranch = false;
  if (repeatFailedSpeedBranch) try {
    for (let trial=0;trial<3;trial++) {
      for (const mode of ["scene","box"]) {
        for (const label of trial===1?["checksum","baseline"]:["baseline","checksum"]) {
          await fresh(label,mode,async (p,row)=> {
            result.loading.push({trial,mode,label,...row});
            check(row.integrity.length===1 && row.loaded.join()===mode,label+"_only_requested_model_"+mode+trial,"Q68");
          });
        }
      }
    }
    for (const mode of ["scene","box"]) {
      const controls=result.loading.filter(row=>row.label==="baseline"&&row.mode===mode);
      const candidates=result.loading.filter(row=>row.label==="checksum"&&row.mode===mode);
      const reduction=1-median(candidates.map(row=>row.integrity[0].integrity_ms))/median(controls.map(row=>row.integrity[0].integrity_ms));
      result["Q68_"+mode]={reduction,control_ms:median(controls.map(row=>row.integrity[0].integrity_ms)),candidate_ms:median(candidates.map(row=>row.integrity[0].integrity_ms))};
      check(reduction>=.1,"Q68_"+mode+"_three_real_decodes_phase_reduction","Q68");
      check(candidates.every(row=>row.navigation_ms<=controls.find(x=>x.trial===row.trial).navigation_ms*1.1),"Q68_"+mode+"_full_navigation_protected","Q68");
    }
  } catch(error) { result.failures.push({mechanism:"Q68",error:String(error)}); }
  globalThis.saveViewerProgress?.(result);
  try {
    const context=await browser.newContext({viewport:{width:1440,height:900},deviceScaleFactor:1});
    const p=await context.newPage();
    try {
      result.cacheCold=[]; result.cacheWarm=[];
      for (const label of ["baseline","cache"]) result.cacheCold.push({label,...await ready(p,label)});
      for (let trial=0;trial<3;trial++) for (const label of trial===1?["cache","baseline"]:["baseline","cache"])
        result.cacheWarm.push({label,trial,...await ready(p,label)});
      const control=result.cacheWarm.filter(row=>row.label==="baseline"), candidate=result.cacheWarm.filter(row=>row.label==="cache");
      result.Q71={control_ms:median(control.map(row=>row.navigation_ms)),candidate_ms:median(candidate.map(row=>row.navigation_ms))};
      result.Q71.reduction=1-result.Q71.candidate_ms/result.Q71.control_ms;
      check(candidate.every(row=>row.cache.length===2 && row.cache.every(x=>x.hit&&x.committed_after_native_validation)),"Q71_actual_verified_warm_store_hits","Q71");
      check(result.Q71.reduction>=.1,"Q71_three_warm_full_loads_at_least10pct_faster","Q71");
      const corruption=await p.evaluate(async()=>{
        const assets=await (await fetch("../../ASSET_MANIFEST.json")).json(), name="box2-native-verified-assets-v1";
        const cache=await caches.open(name), key=new URL("/.__box2_verified_asset__/"+assets.scene.asset_sha256,location.origin).href;
        const original=await cache.match(key), data=new Uint8Array(await original.arrayBuffer()); data[data.length-1]^=1;
        await cache.put(key,new Response(data,{headers:original.headers}));
        const module=await import("./packed_splat_asset.js");
        try { await module.loadPackedAsset("./scene.spk",assets.scene); return false; } catch(error) { return String(error).includes("identity mismatch"); }
      });
      check(corruption,"Q71_corrupt_persistent_asset_rejected_without_fallback","Q71");
      const eviction=await p.evaluate(async()=>{
        const m=await import("./persistent_asset_cache.js"), oldFetch=window.fetch;
        try {
          await m.clearAssetArtifacts(); window.fetch=async()=>new Response(new Uint8Array([1,2,3,4]));
          for(let i=1;i<=4;i++) { const value=await m.readAssetArtifact("/fixture/"+i,i.toString(16).padStart(64,"0")); await value.commitVerified(); }
          const c=await caches.open("box2-native-verified-assets-v1"), keys=await c.keys();
          return {length:keys.length,firstRemoved:!await c.match(new URL("/.__box2_verified_asset__/"+"1".padStart(64,"0"),location.origin).href)};
        } finally { window.fetch=oldFetch; await m.clearAssetArtifacts(); }
      });
      check(eviction.length===3&&eviction.firstRemoved,"Q71_declared_FIFO_entry_bound_enforced","Q71");
    } finally { await context.close(); }
  } catch(error) { result.failures.push({mechanism:"Q71",error:String(error)}); }
  globalThis.saveViewerProgress?.(result);
  try {
    result.motionPacing=[];
    for(let trial=0;trial<3;trial++) {
      for(const label of trial===1?["motion","baseline"]:["baseline","motion"]) {
        await fresh(label,"scene",async p=>{
          await p.getByLabel("移动方式").selectOption("fly");
          await p.getByRole("button",{name:"无人机斜俯视",exact:true}).click(); await sleep(p,400);
          await p.locator("canvas").focus();
          const before=await state(p), started=Date.now();
          await p.keyboard.down("w"); await sleep(p,1500); const after=await state(p), elapsed=Date.now()-started;
          await p.keyboard.up("w"); await sleep(p,400);
          result.motionPacing.push({trial,label,frames:after.frames-before.frames,elapsed_ms:elapsed,
            fps:1000*(after.frames-before.frames)/elapsed,moving:after.motion,stopped:(await state(p)).motion});
        });
      }
    }
    const native=median(result.motionPacing.filter(row=>row.label==="baseline").map(row=>row.fps));
    const candidate=median(result.motionPacing.filter(row=>row.label==="motion").map(row=>row.fps));
    result.Q69_pacing={native_fps:native,candidate_fps:candidate,ratio:candidate/native};
    check(candidate>=.8*native,"Q69_three_actual_flight_intervals_frame_pacing_protected","Q69");
  } catch(error) { result.failures.push({mechanism:"Q69",error:String(error)}); }
  globalThis.saveViewerProgress?.(result);
  const identities=new Map();
  try {
    result.sourceCaptureVisibility=[];
    for(const label of ["baseline","cache","motion","shared","reference","shared-reference"]) {
      await fresh(label,"box",async(p,row)=>{
        await p.getByLabel("移动方式").selectOption("fly");
        for(const mode of ["box","scene"]) {
          await p.getByLabel("展示范围").selectOption(mode);
          await p.waitForFunction(m=>window.box2Viewer.loadedModels.includes(m),mode,{timeout:60000}); await sleep(p,500);
          const views=await p.evaluate(async()=> (await (await fetch("./camera_path.json")).json()).length);
          check(views===1016,label+"_all1016_capture_poses_available","shared");
          const keys=[];
          for(let i=0;i<24;i++) keys.push({key:"capture"+i,index:Math.round(i*(views-1)/23)});
          keys.push({key:"top",button:"无人机俯视"},{key:"drone",button:"无人机斜俯视"});
          for(const view of keys) {
            if(view.index!==undefined) await p.evaluate(index=>window.box2Viewer.setView(index),view.index);
            else await p.getByRole("button",{name:view.button,exact:true}).click();
            await sleep(p,450);
            const current=await state(p), image=await pixels(p), key=mode+"_"+view.key;
            // A source capture may face outside the display crop; equivalence still covers that frame.
            if(view.index===undefined) check(image.bins>=16,label+"_nonempty_"+key,"shared");
            else result.sourceCaptureVisibility.push({label,key,nonempty_rgb_bins:image.bins>=16,bins:image.bins});
            const control=label==="reference"||label==="shared-reference"?"reference":"baseline";
            if(label===control) identities.set(control+"_"+key,{image,projection:current.projection});
            else {
              const original=identities.get(control+"_"+key), equal=JSON.stringify(image)===JSON.stringify(original.image)&&JSON.stringify(current.projection)===JSON.stringify(original.projection);
              result.pixels.push({label,key,equal,original:original.image,candidate:image});
              check(equal,label+"_exact_static_pixels_and_projection_"+key,label==="motion"?"Q69":label.startsWith("shared")?"Q77":"Q71");
            }
            if(view.key==="drone"&&(label==="motion"||label==="shared"||label==="shared-reference")) {
              const file=output+"/"+label+"_"+mode+"_drone.png"; await p.locator("canvas").screenshot({path:file}); result.captures.push(file);
            }
          }
        }
        if(label!=="motion"&&!label.startsWith("shared")) {globalThis.saveViewerProgress?.(result);return;}
        await p.getByRole("button",{name:"重置采集视角",exact:true}).click(); await sleep(p,300);
        await p.locator("canvas").focus();
        let before=await state(p);
        await p.keyboard.down("w"); await sleep(p,450); let during=await state(p); await p.keyboard.up("w"); await sleep(p,400);
        let after=await state(p);
        check(distance(before.position,after.position)>.04,label+"_actual_free_flight_forward","shared");
        if(label==="motion") {
          const canvas=await p.evaluate(()=>({width:window.box2Viewer.renderer.domElement.width,height:window.box2Viewer.renderer.domElement.height}));
          const ratio=during.motion.physicalWidth*during.motion.physicalHeight/(canvas.width*canvas.height);
          result.Q69={physical_pixel_work_reduction:1-ratio,moving:during.motion,stopped:after.motion};
          check(1-ratio>=.1&&during.motion.moving&&!after.motion.moving&&after.motion.currentRatio===after.motion.fullRatio,"Q69_actual_motion_pixel_work_and_full_restoration","Q69");
        }
        before=after; await p.keyboard.down("e"); await sleep(p,350); await p.keyboard.up("e"); await sleep(p,300); after=await state(p);
        check(after.position[2]>before.position[2]+.04&&Math.hypot(after.position[0]-before.position[0],after.position[1]-before.position[1])<1e-5,label+"_world_Z_up","shared");
        before=after; await sleep(p,400); after=await state(p); check(distance(before.position,after.position)<1e-8,label+"_key_release_stops_flight","shared");
        await p.getByLabel("移动方式").selectOption("orbit"); await p.getByRole("button",{name:"重置采集视角",exact:true}).click();
        for(const [name,button] of [["rotate","left"],["pan","right"]]) {
          before=await state(p); await p.mouse.move(700,430); await p.mouse.down({button}); await p.mouse.move(800,475,{steps:6}); await p.mouse.up({button}); await sleep(p,500); after=await state(p);
          check(distance(before.position,after.position)>.01,label+"_orbit_"+name,"shared");
        }
        before=after; await p.mouse.wheel(0,-180); await sleep(p,500); after=await state(p); check(distance(before.position,after.position)>.01,label+"_wheel_zoom","shared");
        await p.getByRole("button",{name:"播放采集路径",exact:true}).click(); before=await state(p); await sleep(p,650); after=await state(p);
        check(after.playing&&after.frames>before.frames+5,label+"_actual_path_playback","shared");
        await p.getByRole("button",{name:"暂停播放",exact:true}).click(); await sleep(p,400);
        if(label==="motion") check(!(await state(p)).motion.moving,"Q69_path_pause_restores_full_resolution","Q69");
        const slider=p.getByRole("slider",{name:"采集视角",exact:true}); await slider.focus(); await slider.press("End"); await sleep(p,300);
        check((await state(p)).currentView===1015,label+"_final_capture_view_accessible","shared");
        result.controls.push({label,actual_free_flight_drone_orbit_path_ranges:true});
        globalThis.saveViewerProgress?.(result);
      });
    }
    for(const [label,id] of [["cache","Q71"],["motion","Q69"],["shared","Q77"],["shared-reference","Q77"]])
      check(result.pixels.filter(row=>row.label===label).length===52,label+"_all24_capture_and_two_free_camera_views_in_both_ranges",id);
    check(result.sources.Q77_shared_build.build_reduction>=.1,"Q77_actual_shared_native_build_storage_reduction","Q77");
  } catch(error) { result.failures.push({mechanism:"static_pixels_or_navigation",error:String(error)}); }
  try {
    await fresh("checksum","box",async p=>{
      const lifecycle=await p.evaluate(async()=>{
        const m=await import("./packed_splat_asset.js"), expected=(await (await fetch("../../ASSET_MANIFEST.json")).json()).box;
        const original=await (await fetch("./box.spk")).arrayBuffer(), cases=[];
        const test=async(name,buffer,contract=expected)=>{ try { await m.decodePackedAsset(buffer,contract); cases.push({name,rejected:false}); } catch(error) { cases.push({name,rejected:true,error:String(error)}); } };
        const mutate=(offset,value)=>{const b=original.slice(0);new Uint8Array(b)[offset]^=value;return b;};
        await test("magic",mutate(0,1)); await test("truncation",original.slice(0,-4));
        await test("buffer_checksum",mutate(original.byteLength-1,1));
        await test("source_build_identity",original,{...expected,spark_sha256:"0".repeat(64)});
        await test("population",original,{...expected,numSplats:expected.numSplats+1});
        return cases;
      });
      result.Q68_lifecycle=lifecycle; check(lifecycle.length===5&&lifecycle.every(row=>row.rejected),"Q68_all_original_integrity_guards_fail_closed","Q68");
    });
  } catch(error) { result.failures.push({mechanism:"Q68_lifecycle",error:String(error)}); }
  result.status=result.failures.length?"completed_with_failed_candidates":"all_candidate_browser_guards_passed_visual_pending";
  result.acceptable_mechanisms=["Q69","Q71","Q77"].filter(id=>
    result.checks.some(row=>row.mechanism===id)&&result.checks.filter(row=>row.mechanism===id||row.mechanism==="shared").every(row=>row.passed)&&
    !result.failures.some(row=>row.mechanism===id));
  return result;
}
