// Validate actual orbit, translation, zoom, free flight, and path playback in both views.
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.argv[2]);
const base=process.argv[3], output=process.argv[4], branch=process.argv[5], pointPage=process.argv[6]||'surface/index.html';
const distance=(a,b)=>Math.sqrt(a.reduce((sum,value,index)=>sum+(value-b[index])**2,0));
(async()=>{
  let browser, page, key, phase='browser_launch', currentKind=null, domLoaded=false, result;
  const rows=[], denied=[], errors=[], consoleErrors=[], badResponses=[], failedRequests=[], requests=[];
  try{
    browser=await chromium.launch({channel:'msedge',headless:true,args:['--disable-background-networking','--disable-background-timer-throttling','--disable-renderer-backgrounding']});
    const context=await browser.newContext({viewport:{width:1440,height:960}});
    await context.route('**/*',route=>{ const url=new URL(route.request().url());if(['127.0.0.1','localhost','[::1]'].includes(url.hostname))return route.continue();denied.push(url.hostname);return route.abort('blockedbyclient'); });
    page=await context.newPage();page.on('pageerror',error=>errors.push(String(error)));
    page.on('console',message=>{if(message.type()==='error')consoleErrors.push(message.text());});
    page.on('request',request=>requests.push({url:request.url(),type:request.resourceType()}));
    page.on('response',response=>{if(response.status()>=400)badResponses.push({url:response.url(),status:response.status()});});
    page.on('requestfailed',request=>failedRequests.push({url:request.url(),error:request.failure()?.errorText}));
    for(const kind of ['gaussian','surface']){
      currentKind=kind;key=kind==='gaussian'?'box2Viewer':'surfaceViewer';phase='initialization';domLoaded=false;
      await page.goto(base+(kind==='gaussian'?branch+'/cases/packed/index.html':pointPage),{waitUntil:'domcontentloaded'});
      domLoaded=true;
      // Readiness polling must not depend on the display animation scheduling policy.
      await page.waitForFunction(key=>window[key]?.ready||window[key]?.errors?.length,key,{timeout:90000,polling:100});
      const read=()=>page.evaluate(key=>{const s=window[key];return{ready:s.ready,position:s.getCameraPosition(),direction:s.getCameraDirection(),errors:s.errors,navigation:s.navigationMode,scene:s.sceneMode,view:s.currentView,playing:s.playing};},key);
      const first=await read();if(!first.ready||first.errors.length||errors.length||consoleErrors.length||badResponses.length||failedRequests.length)throw new Error('Viewer initialization failed '+JSON.stringify({first,errors,consoleErrors,badResponses,failedRequests}));
      phase='interaction';
      const canvas=page.locator('canvas'); const box=await canvas.boundingBox();
      if(!box||box.width<100||box.height<100)throw new Error('Visible interactive canvas is missing');
      const x=box.x+box.width*.42,y=box.y+box.height*.5;
      await page.mouse.move(x,y);await page.mouse.down();await page.mouse.move(x+100,y+35,{steps:12});await page.mouse.up();
      await page.waitForFunction(({key,direction})=>{const now=window[key].getCameraDirection();return now.some((v,i)=>Math.abs(v-direction[i])>1e-3);},{key,direction:first.direction});
      const orbit=await read();
      await page.mouse.move(x,y);await page.mouse.down({button:'right'});await page.mouse.move(x+90,y+20,{steps:12});await page.mouse.up({button:'right'});
      await page.waitForFunction(({key,position})=>window[key].getCameraPosition().some((v,i)=>Math.abs(v-position[i])>1e-3),{key,position:orbit.position});
      const pan=await read();await page.mouse.wheel(0,-250);
      await page.waitForFunction(({key,position})=>window[key].getCameraPosition().some((v,i)=>Math.abs(v-position[i])>1e-3),{key,position:pan.position});
      const zoom=await read();
      await page.locator('#navigation-mode').selectOption('fly');await canvas.focus();const beforeFlight=await read();
      await page.keyboard.down('w');await page.waitForFunction(({key,position})=>window[key].getCameraPosition().some((v,i)=>Math.abs(v-position[i])>.08),{key,position:beforeFlight.position});await page.keyboard.up('w');
      const flight=await read();
      await page.keyboard.down('e');await page.waitForFunction(({key,z})=>window[key].getCameraPosition()[2]>z+.06,{key,z:flight.position[2]});await page.keyboard.up('e');const ascent=await read();
      await page.locator('#navigation-mode').selectOption('orbit');
      await page.locator('#scene-mode').selectOption(kind==='gaussian'?'box':'focus');
      await page.waitForFunction(({key,scene})=>window[key].sceneMode===scene,{key,scene:kind==='gaussian'?'box':'focus'});
      await page.locator('#scene-mode').selectOption('scene');
      if(kind==='surface')await page.locator('#top').click();
      await page.locator('#reset').click();const reset=await read();await page.locator('#play').click();
      await page.waitForFunction(({key,view})=>window[key].playing&&window[key].currentView!==view,{key,view:reset.view});
      await page.locator('#play').click();
      await page.screenshot({path:path.join(output,kind+'_viewer.png')});
      const last=await read();if(last.errors.length||errors.length||consoleErrors.length||badResponses.length||failedRequests.length)throw new Error('Actual interaction raised errors');
      rows.push({kind,rotationDirectionChange:distance(first.direction,orbit.direction),panMeters:distance(orbit.position,pan.position),zoomMeters:distance(pan.position,zoom.position),flightMeters:distance(beforeFlight.position,flight.position),ascentMeters:ascent.position[2]-flight.position[2],cropSwitchVerified:true,capturePathPlaybackVerified:true,topViewVerified:kind==='surface',final:last});
    }
    if(denied.length)throw new Error('External browser destinations were attempted '+denied.join(','));
    result={status:'both_viewers_actual_interactions_verified',rows,pageErrors:errors,consoleErrors,badResponses,failedRequests,
      externalRequestsDenied:denied,onlyLoopbackRequestsAllowed:true,requests};
  }catch(error){
    let snapshot=null,timer;
    if(page){
      try{
        snapshot=await Promise.race([page.evaluate(key=>({url:location.href,documentReadyState:document.readyState,
          status:document.getElementById('status')?.textContent,error:document.getElementById('error')?.textContent,
          viewer:window[key]?{ready:window[key].ready,errors:window[key].errors,frames:window[key].frames}:null}),key),
          new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('Bounded snapshot timeout')),2500);})]);
      }catch(snapshotError){snapshot={snapshotError:String(snapshotError)};}
      finally{clearTimeout(timer);}
    }
    result={status:'failed_viewer_interaction',error:String(error.stack??error),phase,kind:currentKind,domLoaded,
      failure_kind:error.name==='TimeoutError'&&phase==='initialization'&&domLoaded?'readiness_timeout':'viewer_error',
      readiness_timeout_ms:90000,snapshot,rows,denied,pageErrors:errors,consoleErrors,badResponses,failedRequests,requests};
    process.exitCode=1;
  }finally{
    let closed=false;
    if(browser){try{await browser.close();closed=true;}catch(error){result={...result,status:'failed_viewer_interaction',
      failure_kind:'browser_close_failed',closeError:String(error)};process.exitCode=1;}}
    result.browserClosed=closed;
    fs.writeFileSync(path.join(output,result.status==='both_viewers_actual_interactions_verified'?'RESULT.json':'FAILURE.json'),JSON.stringify(result,null,2)+'\n');
    console.log(result.status);
  }
})();
