// Inspect actual focused and overhead surface views using only local resources.
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.argv[2]);
(async () => {
  const browser = await chromium.launch({channel:'msedge',headless:true,args:['--disable-background-networking']});
  const denied=[];
  try {
    const context=await browser.newContext({viewport:{width:1440,height:960}});
    await context.route('**/*',route=>{
      const host=new URL(route.request().url()).hostname;
      if(['127.0.0.1','localhost','[::1]'].includes(host))return route.continue();
      denied.push(host);return route.abort('blockedbyclient');
    });
    const page=await context.newPage();
    await page.goto(process.argv[3]+'surface/index.html');
    await page.waitForFunction(()=>window.surfaceViewer?.ready,{timeout:60000});
    await page.locator('#scene-mode').selectOption('focus');
    const settle=()=>page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    await settle();
    await page.screenshot({path:path.join(process.argv[4],'surface_focus.png')});
    await page.locator('#top').click();
    await settle();
    await page.screenshot({path:path.join(process.argv[4],'surface_top.png')});
    const result=await page.evaluate(()=>({points:window.surfaceViewer.points,focusPoints:window.surfaceViewer.focusPoints,
      errors:window.surfaceViewer.errors,worldCoordinatesChanged:false}));
    if(result.errors.length||denied.length)throw new Error('Local surface inspection failed');
    fs.writeFileSync(path.join(process.argv[4],'SURFACE_DISPLAY_DIAGNOSTIC.json'),JSON.stringify({...result,
      status:'actual_focus_and_top_views_inspected',denied,geometric_accuracy_verified:false},null,2)+'\n');
    if(process.argv[5]) {
      await page.goto(process.argv[3]+process.argv[5]+'/cases/packed/index.html');
      await page.waitForFunction(()=>window.box2Viewer?.ready,{timeout:60000});
      const views=[];
      for(const preset of ['top','drone']) {
        await page.locator('[data-preset="'+preset+'"]').click();
        await settle();
        const view=await page.evaluate(()=>({position:window.box2Viewer.getCameraPosition(),
          direction:window.box2Viewer.getCameraDirection(),mode:window.box2Viewer.navigationMode,
          errors:window.box2Viewer.errors}));
        if(view.errors.length||view.mode!=='fly'||!view.position.every(Number.isFinite)||view.direction[2]>-.1)
          throw new Error('The real drone view did not aim downwards');
        views.push({preset,...view});
        await page.screenshot({path:path.join(process.argv[4],'gaussian_'+preset+'.png')});
      }
      if(denied.length)throw new Error('External drone view resources attempted');
      fs.writeFileSync(path.join(process.argv[4],'DRONE_DIAGNOSTIC.json'),JSON.stringify({
        status:'actual_gaussian_top_and_oblique_drone_presets_verified',views,denied},null,2)+'\n');
    }
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
