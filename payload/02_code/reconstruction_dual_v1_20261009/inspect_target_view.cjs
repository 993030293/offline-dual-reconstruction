// Inspect reconstructed shape and target framing, without using screenshots for text review.
const fs=require('node:fs'), path=require('node:path');
const {chromium}=require(process.argv[2]);
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true,args:['--disable-background-networking']});
  const denied=[],errors=[],results=[];
  try{
    const context=await browser.newContext({viewport:{width:1440,height:960}});
    await context.route('**/*',route=>{
      const host=new URL(route.request().url()).hostname;
      if(['127.0.0.1','localhost','[::1]'].includes(host))return route.continue();
      denied.push(host);return route.abort('blockedbyclient');
    });
    const page=await context.newPage();page.on('pageerror',error=>errors.push(String(error)));
    const targetName=process.argv[5]||'target';
    await page.goto(process.argv[3]+(targetName==='target'?'surface/index.html':targetName+'/index.html'));
    await page.waitForFunction(()=>window.surfaceViewer?.ready,{timeout:60000});
    if(!page.url().endsWith('/'+targetName+'/index.html'))throw new Error('The expected target page did not load');
    for(const preset of ['overview','top']){
      await page.locator('#'+preset).click();
      await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
      results.push(await page.evaluate(preset=>({preset,position:window.surfaceViewer.getCameraPosition(),
        direction:window.surfaceViewer.getCameraDirection(),points:window.surfaceViewer.points,
        focusPoints:window.surfaceViewer.focusPoints,errors:window.surfaceViewer.errors}),preset));
      await page.screenshot({path:path.join(process.argv[4],'target_'+preset+'.png')});
    }
    await page.locator('#overview').click();
    await page.locator('#navigation-mode').selectOption('fly');
    const canvas=page.locator('canvas');await canvas.focus();
    const before=await page.evaluate(()=>window.surfaceViewer.getCameraPosition());
    await page.keyboard.down('w');
    await page.waitForFunction(before=>window.surfaceViewer.getCameraPosition().some((v,i)=>Math.abs(v-before[i])>.1),before);
    await page.keyboard.up('w');
    if(results.some(row=>row.errors.length)||errors.length||denied.length)throw new Error('Target UI or external resource failure');
    const freeFlight=await page.evaluate(()=>window.surfaceViewer.getCameraPosition());
    await page.goto(process.argv[3]+targetName+'/reference_style.html',{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>document.querySelector('#pointcloud-viewer')?.data?.length>0,{timeout:90000});
    await page.locator('[data-quality="ultra"]').click();
    await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    await page.screenshot({path:path.join(process.argv[4],'target_original_style.png')});
    const native=await page.locator('#status').innerText();
    fs.writeFileSync(path.join(process.argv[4],'TARGET_DISPLAY_REVIEW.json'),JSON.stringify({
      status:'actual_target_overview_top_flight_and_original_preview_style_verified',results,freeFlight,
      targetName,legacyWrongURLRedirectVerified:targetName==='target',originalStyleStatus:native,pageErrors:errors,denied,
      screenshotsPurpose:'object_shape_background_residual_and_framing_only',
      independentGeometryAccuracyVerified:false},null,2)+'\n');
    console.log('target visual display checked');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
