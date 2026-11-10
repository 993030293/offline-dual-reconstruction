async (page) => {
  const base="http://127.0.0.1:8210/release";
  const output=globalThis.effectiveViewerOutput;
  const browser=page.context().browser();
  const result={status:"running",checks:[],pixels:[],controls:[],captures:[],loads:[],native_new_model_equivalence_not_GT:true};
  const check=(value,name)=>{result.checks.push({name,passed:Boolean(value)});if(!value)throw new Error(name);};
  const state=p=>p.evaluate(()=>{const s=window.box2Viewer;return {position:s.getCameraPosition(),projection:s.projectionMatrix,
    keys:s.getActiveKeys(),motion:s.motionResolution,frames:s.frames,currentView:s.currentView,playing:s.playing,loaded:s.loadedModels,errors:s.errors};});
  const distance=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i]));
  const sleep=(p,ms=350)=>p.waitForTimeout(ms);
  const pixels=p=>p.evaluate(async()=>{
    const r=window.box2Viewer.renderer,c=r.domElement,gl=r.getContext(),data=new Uint8Array(c.width*c.height*4);
    gl.readPixels(0,0,c.width,c.height,gl.RGBA,gl.UNSIGNED_BYTE,data);
    const digest=await crypto.subtle.digest("SHA-256",data);
    return {width:c.width,height:c.height,sha256:[...new Uint8Array(digest)].map(v=>v.toString(16).padStart(2,"0")).join("")};
  });
  async function ready(p,label,mode="box") {
    await p.goto(base+"/"+label+"/cases/packed/index.html?mode="+mode);await p.bringToFront();
    await p.waitForFunction(()=>window.box2Viewer?.ready&&window.box2Viewer?.firstReadyRenderMs!=null,undefined,{timeout:60000});
    const row=await p.evaluate(async()=>{
      const s=window.box2Viewer,gl=s.renderer.getContext(),ext=gl.getExtension("WEBGL_debug_renderer_info");
      return {full:s.fullGaussianCount,box:s.boxGaussianCount,cache:globalThis.assetCacheTrace,
        actual_GL_renderer:ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):"unavailable",loaded:s.loadedModels,
        assets:await(await fetch("../../ASSET_MANIFEST.json")).json(),metadata:await(await fetch("./display_manifest.json")).json(),
        filter:s.renderFilter,navigation_ms:s.navigationToFirstReadyRenderMs};
    });
    check(row.actual_GL_renderer.includes("5060 Ti"),label+"_actual_hardware_renderer");
    check(row.metadata.image_count===1016&&row.metadata.iteration===12000&&row.metadata.input_version==="new_map_admission_v1",label+"_current_training_and_camera_version");
    check(row.assets.scene.numSplats===row.metadata.full_gaussian_count&&row.assets.box.numSplats===row.metadata.box_gaussian_count&&
      (mode==="box"?row.box===row.assets.box.numSplats&&row.full===0:row.full===row.assets.scene.numSplats&&row.box===0),label+"_native_requested_population_and_unloaded_other_model");
    result.loads.push({label,...row});return row;
  }
  const controls=new Map();
  for(const label of ["control-baseline","baseline","control-combined","combined"]) {
    const context=await browser.newContext({viewport:{width:1440,height:900},deviceScaleFactor:1});
    const p=await context.newPage(),errors=[];p.on("pageerror",e=>errors.push(String(e)));
    try {
      const first=await ready(p,label);
      await p.getByLabel("移动方式").selectOption("fly");
      for(const mode of ["box","scene"]) {
        await p.getByLabel("展示范围").selectOption(mode);
        await p.waitForFunction(m=>window.box2Viewer.loadedModels.includes(m),mode);await sleep(p);
        check(await p.evaluate(({mode,expected})=>window.box2Viewer[mode==="box"?"boxGaussianCount":"fullGaussianCount"]===expected,{mode,expected:first.assets[mode].numSplats}),label+"_actual_complete_population_"+mode);
        const length=await p.evaluate(async()=>(await(await fetch("./camera_path.json")).json()).length);
        check(length===1016,label+"_all_capture_poses_"+mode);
        const views=Array.from({length:24},(_,i)=>({key:"capture"+i,index:Math.round(i*(length-1)/23)}));
        views.push({key:"top",button:"无人机俯视"},{key:"drone",button:"无人机斜俯视"});
        for(const view of views) {
          if(view.index!==undefined)await p.evaluate(i=>window.box2Viewer.setView(i),view.index);
          else await p.getByRole("button",{name:view.button,exact:true}).click();
          await sleep(p);
          const image=await pixels(p),current=await state(p),key=label.replace("control-","")+"_"+mode+"_"+view.key;
          if(label.startsWith("control-"))controls.set(key,{image,projection:current.projection});
          else {
            const original=controls.get(key),equal=JSON.stringify(original)===JSON.stringify({image,projection:current.projection});
            result.pixels.push({label,mode,view:view.key,equal,original:original.image,candidate:image});
            check(equal,label+"_exact_static_pixels_projection_"+mode+"_"+view.key);
            if(view.key==="drone") {const file=output+"/"+label+"_"+mode+"_drone.png";await p.locator("canvas").screenshot({path:file});result.captures.push(file);}
          }
        }
      }
      if(label.startsWith("control-"))continue;
      await p.getByRole("button",{name:"重置采集视角",exact:true}).click();await sleep(p);
      await p.locator("canvas").focus();let before=await state(p);
      await p.keyboard.down("w");await sleep(p,500);let during=await state(p);await p.keyboard.up("w");await sleep(p);
      let after=await state(p);
      check(distance(before.position,after.position)>.04,label+"_free_flight_forward");
      check(during.motion.moving&&during.motion.currentRatio===.75*after.motion.fullRatio&&!after.motion.moving&&after.motion.currentRatio===after.motion.fullRatio,label+"_motion_reduction_and_static_restore");
      before=after;await p.keyboard.down("e");await sleep(p);await p.keyboard.up("e");await sleep(p);after=await state(p);
      check(after.position[2]>before.position[2]+.04&&Math.hypot(after.position[0]-before.position[0],after.position[1]-before.position[1])<1e-5,label+"_world_Z_up");
      before=after;await sleep(p,400);after=await state(p);check(distance(before.position,after.position)<1e-8,label+"_key_release_stops");
      await p.getByLabel("移动时优先流畅",{exact:true}).uncheck();await p.locator("canvas").focus();
      await p.keyboard.down("w");await sleep(p);during=await state(p);await p.keyboard.up("w");await sleep(p);
      check(!during.motion.moving&&during.motion.currentRatio===during.motion.fullRatio,label+"_optional_full_resolution_motion");
      await p.getByLabel("移动时优先流畅",{exact:true}).check();
      await p.getByLabel("移动方式").selectOption("orbit");await p.getByRole("button",{name:"重置采集视角",exact:true}).click();await sleep(p);
      for(const [name,button] of [["rotate","left"],["pan","right"]]) {
        before=await state(p);await p.mouse.move(700,430);await p.mouse.down({button});await p.mouse.move(800,475,{steps:6});await p.mouse.up({button});await sleep(p);after=await state(p);
        check(distance(before.position,after.position)>.01,label+"_orbit_"+name);
      }
      before=after;await p.mouse.wheel(0,-180);await sleep(p);after=await state(p);check(distance(before.position,after.position)>.01,label+"_wheel_zoom");
      await p.getByRole("button",{name:"播放采集路径",exact:true}).click();before=await state(p);await sleep(p,650);after=await state(p);
      check(after.playing&&after.frames>before.frames+5,label+"_actual_path_playback");
      await p.getByRole("button",{name:"暂停播放",exact:true}).click();await sleep(p);check(!(await state(p)).motion.moving,label+"_pause_restores_full_resolution");
      const slider=p.getByRole("slider",{name:"采集视角",exact:true});await slider.focus();await slider.press("End");await sleep(p);
      check((await state(p)).currentView===1015,label+"_final_capture_pose_available");
      const captureProjection=(await state(p)).projection;
      await p.setViewportSize({width:1200,height:760});await sleep(p);
      const resized=await pixels(p),layout=await p.evaluate(()=>{
        const s=window.box2Viewer,c=s.calibration;
        const scale=c?Math.min(innerWidth/c.width,innerHeight/c.height):1;
        return {expectedWidth:c?Math.floor(c.width*scale):innerWidth,expectedHeight:c?Math.floor(c.height*scale):innerHeight,
          calibration:c,viewport:s.viewport,projection:s.projectionMatrix};
      });
      check(resized.width===layout.expectedWidth&&resized.height===layout.expectedHeight&&!((await state(p)).motion.moving)&&
        JSON.stringify(captureProjection)===JSON.stringify(layout.projection),label+"_calibrated_capture_resize_preserves_letterbox_rays_and_full_resolution");
      await p.getByRole("button",{name:"无人机斜俯视",exact:true}).click();await sleep(p);
      const freeResized=await pixels(p);check(freeResized.width===1200&&freeResized.height===760&&!(await state(p)).motion.moving,label+"_free_camera_resize_and_static_restoration");
      result.controls.push({label,free_flight_drone_orbit_path_ranges_resize_and_optional_resolution:true});
      await ready(p,label,"scene");const warm=await ready(p,label,"scene");
      check(warm.cache.length===2&&warm.cache.every(x=>x.hit&&x.committed_after_native_validation),label+"_combined_verified_cache_warm_hits");
      check(warm.assets.scene.source_ply_sha256===first.assets.scene.source_ply_sha256,label+"_same_current_model_after_reload");
      const corrupt=await p.evaluate(async()=>{
        const expected=(await(await fetch("../../ASSET_MANIFEST.json")).json()).scene;
        const cache=await caches.open("box2-native-verified-assets-v1"),key=new URL("/.__box2_verified_asset__/"+expected.asset_sha256,location.origin).href;
        const entry=await cache.match(key),data=new Uint8Array(await entry.arrayBuffer());data[data.length-1]^=1;
        await cache.put(key,new Response(data,{headers:entry.headers}));
        try {const m=await import("./packed_splat_asset.js");await m.loadPackedAsset("./scene.spk",expected);return false;}
        catch(error){return String(error).includes("identity mismatch");}
        finally{await cache.delete(key);}
      });
      check(corrupt,label+"_corrupt_cache_rejected_without_fallback");
      await p.getByLabel("对照分支").selectOption(label==="baseline"?"combined":"baseline");
      await p.waitForURL(url=>url.pathname.includes("/"+(label==="baseline"?"combined":"baseline")+"/cases/packed/index.html"));
      await p.waitForFunction(()=>window.box2Viewer?.ready,undefined,{timeout:60000});
      check((await state(p)).errors.length===0,label+"_actual_two_model_navigation");
      check(!errors.length,label+"_page_runtime_clean");
      globalThis.saveViewerProgress?.(result);
    } finally {await context.close();}
  }
  check(result.pixels.length===104&&result.pixels.every(row=>row.equal),"all104_current_model_original_static_pixels_exact");
  check(result.controls.length===2,"both_current_models_actual_free_flight_combination");
  result.status="current_models_actual_combination_passed_visual_pending";
  return result;
}
