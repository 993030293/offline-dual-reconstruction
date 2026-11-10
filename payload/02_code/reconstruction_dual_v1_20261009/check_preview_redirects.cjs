// Check actual navigation from retained URLs without additional screenshots.
const fs=require('node:fs');
const {chromium}=require(process.argv[2]);
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true,args:['--disable-background-networking']});
  try{
    const context=await browser.newContext();
    const denied=[];
    await context.route('**/*',route=>{
      const host=new URL(route.request().url()).hostname;
      if(['127.0.0.1','localhost','[::1]'].includes(host))return route.continue();
      denied.push(host);return route.abort('blockedbyclient');
    });
    const page=await context.newPage(),rows=[];
    for(const source of ['target/index.html','surface/index.html']){
      await page.goto(process.argv[3]+source);
      await page.waitForFunction(()=>window.surfaceViewer?.ready,{timeout:60000});
      if(!page.url().endsWith('/target_adaptive/index.html'))throw new Error('Wrong compatibility navigation: '+page.url());
      rows.push({source,actual:page.url(),points:await page.evaluate(()=>window.surfaceViewer.points)});
    }
    if(denied.length)throw new Error('An external request was attempted');
    fs.writeFileSync(process.argv[4],JSON.stringify({status:'legacy_preview_urls_redirect_to_current_adaptive_target',rows,denied},null,2)+'\n');
    console.log('compatibility target URLs verified');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
