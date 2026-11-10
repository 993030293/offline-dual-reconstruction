async (page) => {
  const base = 'http://127.0.0.1:8202';
  const root = 'C:/lidar-camera-fusion-mask-robust-downsampling/output/playwright/gaussian_optimization_p25/002';
  const browser = page.context().browser(), checks = [], captures = [], pixels = [], errors = [];
  const manifest = await (await page.request.get(base+'/ASSET_MANIFEST.json')).json();
  const check = (value, name) => { checks.push({name, passed: Boolean(value)}); if (!value) throw new Error(name); };
  const state = p => p.evaluate(() => ({position: window.box2Viewer.getCameraPosition(),
    direction: window.box2Viewer.getCameraDirection(), mode: window.box2Viewer.sceneMode,
    playing: window.box2Viewer.playing, frames: window.box2Viewer.frames,
    view: window.box2Viewer.currentView, errors: window.box2Viewer.errors}));
  const distance = (a,b) => Math.hypot(...a.map((x,i)=>x-b[i]));
  const settleMode = async (p, mode) => {
    await p.getByLabel('展示范围').selectOption(mode);
    await p.waitForFunction(m=>window.box2Viewer.sceneMode===m && !window.box2Viewer.modelLoading, mode);
    await p.waitForTimeout(500);
  };
  try {
    await page.goto('about:blank');
    for (const label of ['default', 'classic']) {
      const context = await browser.newContext({viewport:{width:1440,height:900},deviceScaleFactor:1});
      const p = await context.newPage();
      p.on('pageerror', error=>errors.push(String(error)));
      try {
        await p.goto(base+'/cases/'+label+'/index.html?mode=box');
        await p.bringToFront();
        await p.waitForFunction(()=>window.box2Viewer?.ready, undefined, {timeout:60000});
        await p.getByLabel('移动方式').selectOption('fly');
        for (const mode of ['box', 'scene']) {
          await settleMode(p, mode);
          for (const [view, title] of [['capture','重置采集视角'],['top','无人机俯视'],['drone','无人机斜俯视']]) {
            await p.getByRole('button',{name:title,exact:true}).click();
            await p.waitForTimeout(800);
            const image = await p.evaluate(async()=>{
              const r=window.box2Viewer.renderer,c=r.domElement,gl=r.getContext();
              const data=new Uint8Array(c.width*c.height*4), bins=new Uint8Array(4096);
              gl.readPixels(0,0,c.width,c.height,gl.RGBA,gl.UNSIGNED_BYTE,data);
              for(let i=0;i<data.length;i+=4) bins[((data[i]>>4)<<8)|((data[i+1]>>4)<<4)|(data[i+2]>>4)]=1;
              const hash=await crypto.subtle.digest('SHA-256',data);
              return {width:c.width,height:c.height,bins:bins.reduce((a,b)=>a+b,0),
                sha256:[...new Uint8Array(hash)].map(x=>x.toString(16).padStart(2,'0')).join('')};
            });
            check(image.bins>=16,label+'_'+mode+'_'+view+'_nonempty');
            pixels.push({label,mode,view,...image});
            const file=root+'/'+label+'_'+mode+'_'+view+'.png';
            await p.locator('canvas').screenshot({path:file}); captures.push(file);
          }
        }
        const counts=await p.evaluate(()=>({full:window.box2Viewer.fullGaussianCount,box:window.box2Viewer.boxGaussianCount}));
        check(counts.full===manifest.scene.numSplats && counts.box===manifest.box.numSplats,label+'_unchanged_population');
        if(label==='classic') {
          const contractChecks=await p.evaluate(async()=>{
            const {resolveRenderFilter}=await import('./render_filter_contract.js');
            const contract=await (await fetch('./render_filter.json')).json();
            const assets=await (await fetch('../../ASSET_MANIFEST.json')).json();
            const outcomes=[];
            for(const patch of [{schema:2},{rasterizeMode:'antialiased'},{eps2d:.4},{antialiasing:true},
                {trainingConfigSha256:null},{models:{scene:'0'.repeat(64),box:contract.models.box}}]) {
              let rejected=false; try {resolveRenderFilter({...contract,...patch},assets);} catch {rejected=true;} outcomes.push(rejected);
            }
            return {outcomes,live:window.box2Viewer.renderFilter};
          });
          check(contractChecks.outcomes.every(Boolean),'six_unknown_or_mismatched_contracts_rejected');
          check(contractChecks.live.preBlurAmount===.3 && contractChecks.live.blurAmount===0,'actual_model_contract_bound');
          await p.getByRole('button',{name:'重置采集视角',exact:true}).click();
          await p.locator('canvas').focus();
          let a=await state(p); await p.keyboard.down('w'); await p.waitForTimeout(350); await p.keyboard.up('w');
          let b=await state(p); check(distance(a.position,b.position)>.04,'fly_forward');
          a=await state(p); await p.keyboard.down('e'); await p.waitForTimeout(350); await p.keyboard.up('e'); b=await state(p);
          check(b.position[2]>a.position[2]+.04 && Math.hypot(b.position[0]-a.position[0],b.position[1]-a.position[1])<1e-5,'world_Z_up');
          await p.waitForTimeout(350); a=await state(p); await p.waitForTimeout(450);
          check(distance(a.position,(await state(p)).position)<1e-8,'released_key_stops_flight');
          check((await state(p)).frames===a.frames,'idle_demand_rendering_retained');
          await p.getByLabel('移动方式').selectOption('orbit');
          await p.getByRole('button',{name:'重置采集视角',exact:true}).click();
          for(const [name,button,dx] of [['rotate','left',100],['pan','right',80]]) {
            a=await state(p); await p.mouse.move(500,400); await p.mouse.down({button});
            await p.mouse.move(500+dx,460,{steps:6}); await p.mouse.up({button}); await p.waitForTimeout(400);
            check(distance(a.position,(await state(p)).position)>.01,'orbit_'+name);
          }
          a=await state(p); await p.mouse.wheel(0,-200); await p.waitForTimeout(400);
          check(distance(a.position,(await state(p)).position)>.01,'orbit_zoom');
          await p.getByRole('button',{name:'播放采集路径',exact:true}).click(); a=await state(p); await p.waitForTimeout(600); b=await state(p);
          check(b.playing && b.frames>a.frames+5,'path_playback');
          await p.getByRole('button',{name:'暂停播放',exact:true}).click();
          const slider=p.getByRole('slider',{name:'采集视角',exact:true}); await slider.focus(); await slider.press('End');
          await p.waitForTimeout(100); check((await state(p)).view===211,'all212_path_views');
          await p.setViewportSize({width:390,height:844});
          await p.waitForFunction(()=>innerWidth===390 && window.box2Viewer.viewport.width<=390);
          check(true,'mobile_calibrated_resize');
        }
        check(!(await state(p)).errors.length,label+'_no_runtime_errors');
      } finally {await context.close();}
    }
    const context=await browser.newContext({viewport:{width:1440,height:900},deviceScaleFactor:1});
    let sceneRequests=0;
    await context.route('**/scene.spk',async route=>{sceneRequests++; return sceneRequests===1?route.abort('failed'):route.continue();});
    const retryPage=await context.newPage();
    try {
      await retryPage.goto(base+'/cases/classic/index.html?mode=box');
      await retryPage.waitForFunction(()=>window.box2Viewer?.ready);
      await retryPage.getByLabel('展示范围').selectOption('scene');
      await retryPage.waitForFunction(()=>window.box2Viewer.loadFailures.length===1);
      check((await state(retryPage)).mode==='box','failed_scene_retains_box');
      await retryPage.getByRole('button',{name:'重试加载',exact:true}).click();
      await retryPage.waitForFunction(()=>window.box2Viewer.sceneMode==='scene' && !window.box2Viewer.modelLoading);
      check(sceneRequests===2,'actual_retry_success');
    } finally {await context.close();}
    check(pixels.slice(6).some((x,i)=>x.sha256!==pixels[i].sha256),'actual_model_filter_changes_pixels');
    check(errors.length===0,'no_uncaught_errors');
    return {status:'browser_checks_passed_visual_review_pending',checks,captures,pixels,errors,
      scope:'display_filter_correctness_not_training_quality_or_geometry_accuracy',counted:false};
  } catch(error) {
    return {status:'failed_browser_validation',error:String(error.stack??error),checks,captures,pixels,errors,counted:false};
  }
}
