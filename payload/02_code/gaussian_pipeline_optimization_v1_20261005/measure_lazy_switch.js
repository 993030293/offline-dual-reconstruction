async (page) => {
  const rows=[], browser=page.context().browser();
  await page.goto('about:blank');
  for(let trial=0;trial<3;trial++) {
    for(const initial of ['scene','box']) {
      for(const label of (trial===1?['lazy','eager']:['eager','lazy'])) {
        const context=await browser.newContext({viewport:{width:1440,height:900},deviceScaleFactor:1});
        const p=await context.newPage();
        try {
          await p.goto('http://127.0.0.1:8201/cases/'+label+'/index.html?mode='+initial);
          await p.bringToFront(); await p.waitForFunction(()=>window.box2Viewer?.ready);
          const initialization_ms=await p.evaluate(()=>window.box2Viewer.modelInitializationMs);
          let switch_measurement=null;
          if(initial==='box') switch_measurement=await p.evaluate(async()=>{
            const begin=performance.now(),initialFrames=window.box2Viewer.frames;
            const select=document.getElementById('scene-mode');
            select.value='scene'; select.dispatchEvent(new Event('change',{bubbles:true}));
            return new Promise((resolve,reject)=>{
              function poll(){
                const state=window.box2Viewer;
                if(state.sceneMode==='scene' && !state.modelLoading && state.frames>initialFrames)
                  return resolve({event_to_ready_frame_ms:performance.now()-begin,full_splats:state.fullGaussianCount});
                if(performance.now()-begin>15000) return reject(new Error('Scene switch timeout'));
                requestAnimationFrame(poll);
              }
              requestAnimationFrame(poll);
            });
          });
          rows.push({trial,initial,label,initialization_ms,switch_measurement});
        }finally{await context.close();}
      }
    }
  }
  return {status:'auxiliary_latency_measured',rows,time_boundary:'change_event_to_ready_render_polled_by_requestAnimationFrame',
    browser_cache_empty_each_context:true,OS_file_cache_cleared:false,counted:false};
}
