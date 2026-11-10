async (page) => {
  await page.waitForFunction(() => window.box2Viewer?.ready,undefined,{timeout:45000});
  if (!page.url().startsWith("http://127.0.0.1:8195/")) throw new Error("Wrong active port");
  const receipt=await page.evaluate(async () => (await fetch("/DISPLAY_EXPORT_RECEIPT.json")).json());
  if (receipt.navigation_version!=="free_flight_v2") throw new Error("Old preview is still active");
  await page.getByRole("button",{name:"无人机斜俯视",exact:true}).click();
  const before=await page.evaluate(() => window.box2Viewer.getCameraPosition());
  await page.keyboard.down("a"); await page.waitForTimeout(300); await page.keyboard.up("a");
  const after=await page.evaluate(() => window.box2Viewer.getCameraPosition());
  if (Math.hypot(...after.map((value,i)=>value-before[i]))<.05) throw new Error("Active preview does not accept flight input");
  await page.getByRole("button",{name:"重置采集视角",exact:true}).click();
  await page.getByLabel("移动方式").selectOption("orbit");
  await page.getByLabel("展示范围").selectOption("box");
  await page.waitForTimeout(200);
  const result=await page.evaluate(() => ({ready:window.box2Viewer.ready,errors:window.box2Viewer.errors,full:window.box2Viewer.fullGaussianCount,currentView:window.box2Viewer.currentView,navigation:window.box2Viewer.navigationMode}));
  if (!result.ready || result.errors.length || result.full!==543436) throw new Error("Active model or runtime errors");
  return {status:"passed_active_endpoint",url:page.url(),...result,model_retrained:false};
}
