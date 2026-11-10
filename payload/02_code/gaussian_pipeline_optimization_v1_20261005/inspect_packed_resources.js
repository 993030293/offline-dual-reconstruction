async (page) => {
  const context = await page.context().browser().newContext();
  const records = [];
  const pending = [];
  context.on("requestfinished", request => {
    if (!/\.(ply|spk)$/.test(request.url())) return;
    pending.push((async () => records.push({ url: request.url(), sizes: await request.sizes() }))());
  });
  const p = await context.newPage();
  await p.goto("http://127.0.0.1:8198/cases/ply/index.html");
  await p.waitForFunction(() => window.box2Viewer?.ready);
  await Promise.all(pending);
  const result = await p.evaluate(() => ({
    ready: window.box2Viewer.ready,
    initialization_ms: window.box2Viewer.modelInitializationMs,
    resources: performance.getEntriesByType("resource").map(r => ({
      name: r.name, type: r.initiatorType, transfer: r.transferSize, decoded: r.decodedBodySize
    }))
  }));
  await context.close();
  return { ...result, network: records };
}
