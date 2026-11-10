async (page) => {
  const base = 'http://127.0.0.1:8201';
  const root = 'C:/lidar-camera-fusion-mask-robust-downsampling/output/playwright/gaussian_optimization_p14/002';
  const browser = page.context().browser();
  const manifest = await (await page.request.get(base + '/ASSET_MANIFEST.json')).json();
  const checks = [], loading = [], captures = [], pixelComparisons = [], errors = [];
  const referencePixels = new Map();
  const check = (value, name) => { checks.push({ name, passed: Boolean(value) }); if (!value) throw new Error(name); };
  const pixel = p => p.evaluate(async () => {
    const r = window.box2Viewer.renderer, c = r.domElement, gl = r.getContext();
    const rgba = new Uint8Array(c.width*c.height*4), bins = new Uint8Array(4096);
    gl.readPixels(0, 0, c.width, c.height, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    for (let i = 0; i < rgba.length; i += 4) bins[((rgba[i]>>4)<<8)|((rgba[i+1]>>4)<<4)|(rgba[i+2]>>4)] = 1;
    const hash = await crypto.subtle.digest('SHA-256', rgba);
    return { width: c.width, height: c.height, color_bins: bins.reduce((a,b)=>a+b,0),
      sha256: [...new Uint8Array(hash)].map(x=>x.toString(16).padStart(2,'0')).join('') };
  });
  const state = p => p.evaluate(() => ({ ready: window.box2Viewer.ready, mode: window.box2Viewer.sceneMode,
    loading: window.box2Viewer.modelLoading, loaded: window.box2Viewer.loadedModels,
    position: window.box2Viewer.getCameraPosition(), direction: window.box2Viewer.getCameraDirection(),
    frames: window.box2Viewer.frames, errors: window.box2Viewer.errors,
    view: window.box2Viewer.currentView, playing: window.box2Viewer.playing }));
  const distance = (a,b) => Math.hypot(...a.map((x,i)=>x-b[i]));
  const settleMode = async (p, mode) => {
    await p.getByLabel('展示范围').selectOption(mode);
    await p.waitForFunction(value => window.box2Viewer.sceneMode === value && !window.box2Viewer.modelLoading,
      mode, { timeout: 15000 });
    await p.waitForTimeout(600);
  };
  try {
  await page.goto('about:blank');
  for (let trial=0; trial<3; trial++) {
    for (const label of (trial===1 ? ['lazy','eager'] : ['eager','lazy'])) {
      const context = await browser.newContext({ viewport: {width:1440,height:900}, deviceScaleFactor:1 });
      const resources = [], sizes = [];
      context.on('requestfinished', request => {
        if (!request.url().endsWith('.spk')) return;
        sizes.push((async () => { const size=await request.sizes(), response=await request.response();
          resources.push({name:request.url(),bytes:size.responseBodySize,status:response.status()}); })());
      });
      const p = await context.newPage();
      p.on('pageerror', error=>errors.push(String(error)));
      try {
        await p.goto(base+'/cases/'+label+'/index.html?mode=box');
        await p.bringToFront();
        await p.waitForFunction(()=>window.box2Viewer?.ready && window.box2Viewer.firstReadyRenderMs != null,
          undefined,{timeout:60000});
        await Promise.all(sizes);
        const result = await p.evaluate(()=>({initialization_ms:window.box2Viewer.modelInitializationMs,
          first_ready_render_ms:window.box2Viewer.firstReadyRenderMs,
          navigation_to_first_ready_render_ms:window.box2Viewer.navigationToFirstReadyRenderMs,
          full_splats:window.box2Viewer.fullGaussianCount,box_splats:window.box2Viewer.boxGaussianCount,
          loaded:window.box2Viewer.loadedModels}));
        result.resources=resources.map(row=>({...row}));
        loading.push({trial,label,initial_mode:'box',...result});
        check(result.box_splats===manifest.box.numSplats,label+'_box_population_'+trial);
        check(result.resources.length===(label==='lazy'?1:2) && result.resources.every(row=>row.status===200 && row.bytes>0),
          label+'_fresh_initial_requests_'+trial);
        if(label==='lazy') check(result.full_splats===0 && result.loaded.join(',')==='box','unused_scene_not_loaded_'+trial);
        await p.getByLabel('移动方式').selectOption('fly');
        await p.getByRole('button',{name:'重置采集视角',exact:true}).click();
        await p.waitForTimeout(1000);
        const first=await pixel(p);
        check(first.color_bins>=16,label+'_initial_actual_model_pixels_'+trial);
        if(trial===0) {
          for(const mode of ['box','scene']) {
            await settleMode(p,mode);
            for(const view of ['capture','top','drone']) {
              await p.getByRole('button',{name:view==='capture'?'重置采集视角':view==='top'?'无人机俯视':'无人机斜俯视',exact:true}).click();
              await p.waitForTimeout(900);
              const id=await pixel(p),key=mode+'_'+view;
              check(id.color_bins>=16,label+'_visible_'+key);
              if(label==='eager') referencePixels.set(key,id);
              else { const ref=referencePixels.get(key),equal=id.sha256===ref.sha256 && id.width===ref.width && id.height===ref.height;
                pixelComparisons.push({key,reference:ref,candidate:id,equal}); check(equal,'exact_pixels_'+key); }
              const file=root+'/'+label+'_'+key+'.png';
              await p.locator('canvas').screenshot({path:file}); captures.push(file);
            }
          }
          const counts=await p.evaluate(()=>({full:window.box2Viewer.fullGaussianCount,box:window.box2Viewer.boxGaussianCount}));
          check(counts.full===manifest.scene.numSplats && counts.box===manifest.box.numSplats,label+'_both_original_populations');
          if(label==='lazy') {
            await settleMode(p,'box'); await settleMode(p,'scene');
            const attempts=await p.evaluate(()=>window.box2Viewer.modelLoadAttempts);
            check(attempts.box===1 && attempts.scene===1,'switch_reuses_both_assets');
            await p.locator('canvas').focus();
            let a=await state(p); await p.keyboard.down('w'); await p.waitForTimeout(350); await p.keyboard.up('w');
            let b=await state(p); check(distance(a.position,b.position)>.04,'fly_forward');
            a=await state(p); await p.keyboard.down('e'); await p.waitForTimeout(350); await p.keyboard.up('e');
            b=await state(p); check(b.position[2]>a.position[2]+.04 && Math.hypot(b.position[0]-a.position[0],b.position[1]-a.position[1])<1e-5,'world_Z_up');
            await p.waitForTimeout(300); a=await state(p); await p.waitForTimeout(400);
            check(distance(a.position,(await state(p)).position)<1e-8,'key_release');
            await p.getByLabel('移动方式').selectOption('orbit');
            await p.getByRole('button',{name:'重置采集视角',exact:true}).click();
            for(const [name,button,dx] of [['rotate','left',100],['pan','right',80]]) {
              a=await state(p); await p.mouse.move(500,400); await p.mouse.down({button});
              await p.mouse.move(500+dx,460,{steps:6}); await p.mouse.up({button}); await p.waitForTimeout(400);
              check(distance(a.position,(await state(p)).position)>.01,'orbit_'+name);
            }
            a=await state(p); await p.mouse.wheel(0,-200); await p.waitForTimeout(400);
            check(distance(a.position,(await state(p)).position)>.01,'orbit_zoom');
            await p.getByRole('button',{name:'播放采集路径',exact:true}).click(); a=await state(p);
            await p.waitForTimeout(600); b=await state(p); check(b.playing && b.frames>a.frames+5,'path_play');
            await p.getByRole('button',{name:'暂停播放',exact:true}).click();
            const slider=p.getByRole('slider',{name:'采集视角',exact:true}); await slider.focus(); await slider.press('End');
            await p.waitForTimeout(100); check((await state(p)).view===211,'all_212_path_views');
            await p.setViewportSize({width:390,height:844});
            await p.waitForFunction(()=>innerWidth===390 && window.box2Viewer.viewport.width<=390);
            check(true,'mobile_calibrated_viewport');
          }
        }
        check(!(await state(p)).errors.length,label+'_no_runtime_errors_'+trial);
      } finally { await context.close(); }
    }
  }
  for (const scenario of ['race','failure','scene_first']) {
    const context=await browser.newContext({viewport:{width:1440,height:900},deviceScaleFactor:1});
    let sceneRequests=0;
    if(scenario!=='scene_first') await context.route('**/scene.spk',async route=>{
      sceneRequests++;
      if(scenario==='failure' && sceneRequests===1) return route.abort('failed');
      if(scenario==='race') await new Promise(resolve=>setTimeout(resolve,500));
      return route.continue();
    });
    const p=await context.newPage(); p.on('pageerror',error=>errors.push(String(error)));
    try {
      await p.goto(base+'/cases/lazy/index.html'+(scenario==='scene_first'?'':'?mode=box'));
      await p.bringToFront(); await p.waitForFunction(()=>window.box2Viewer?.ready,undefined,{timeout:60000});
      if(scenario==='scene_first') {
        const initial=await p.evaluate(()=>({loaded:window.box2Viewer.loadedModels,mode:window.box2Viewer.sceneMode,
          ms:window.box2Viewer.modelInitializationMs,full:window.box2Viewer.fullGaussianCount}));
        check(initial.mode==='scene' && initial.loaded.join(',')==='scene' && initial.full===manifest.scene.numSplats,'default_scene_only_loads_scene');
        loading.push({scenario,...initial}); await settleMode(p,'box');
      } else if(scenario==='race') {
        await p.getByLabel('展示范围').selectOption('scene'); await p.waitForTimeout(60);
        await p.getByLabel('展示范围').selectOption('scene'); await p.getByLabel('展示范围').selectOption('box');
        await p.waitForFunction(()=>window.box2Viewer.loadedModels.length===2);
        check((await state(p)).mode==='box' && sceneRequests===1,'late_scene_does_not_win_or_duplicate');
        await settleMode(p,'scene'); check(sceneRequests===1,'race_loaded_scene_reused');
      } else {
        await p.getByLabel('展示范围').selectOption('scene');
        await p.waitForFunction(()=>window.box2Viewer.loadFailures.length===1);
        check((await state(p)).ready && (await state(p)).mode==='box','failed_request_retains_ready_box');
        await p.getByRole('button',{name:'重试加载',exact:true}).click();
        await p.waitForFunction(()=>window.box2Viewer.sceneMode==='scene' && !window.box2Viewer.modelLoading);
        check(sceneRequests===2 && await p.getByRole('button',{name:'重试加载',exact:true}).isHidden(),'retry_succeeds_and_clears_error');
      }
      check(!(await state(p)).errors.length,scenario+'_no_global_errors');
    } finally { await context.close(); }
  }
  const median=values=>[...values].sort((a,b)=>a-b)[1];
  const eager=loading.filter(row=>row.label==='eager'),lazy=loading.filter(row=>row.label==='lazy');
  const reduction=1-median(lazy.map(row=>row.initialization_ms))/median(eager.map(row=>row.initialization_ms));
  check(reduction>=.1,'three_trial_selected_model_initialization_reduction_at_least10percent');
  check(errors.length===0,'no_uncaught_browser_errors');
  const receipt={status:'browser_checks_passed_visual_review_pending',checks,loading,pixelComparisons,captures,
    median_initialization_reduction:reduction,raw_framebuffers_equal:pixelComparisons.every(row=>row.equal),
    OS_file_cache_cleared:false,actual_geometry_improvement_claim:false,counted:false,errors};
  await page.goto(base+'/cases/lazy/index.html?mode=box');
  await page.waitForFunction(()=>window.box2Viewer?.ready);
  return receipt;
  } catch(error) {
    return {status:'failed_browser_validation',error:String(error.stack??error),checks,loading,pixelComparisons,captures,errors,counted:false};
  }
}
