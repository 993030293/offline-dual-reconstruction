async (page) => {
  const context = await page.context().browser().newContext({ viewport: { width: 1440, height: 900 } });
  const p = await context.newPage();
  const read = () => p.evaluate(() => ({ inner: [innerWidth, innerHeight], viewport: window.box2Viewer.viewport, frames: window.box2Viewer.frames }));
  try {
    await p.goto("http://127.0.0.1:8198/cases/packed/index.html");
    await p.waitForFunction(() => window.box2Viewer?.ready);
    await p.getByRole("slider", { name: "采集视角", exact: true }).press("End");
    const before = await read();
    await p.setViewportSize({ width: 390, height: 844 });
    const immediate = await read();
    await p.waitForFunction(() => window.box2Viewer.viewport.width <= 390, undefined, { timeout: 5000 });
    const eventual = await read();
    return { before, immediate, eventual };
  } finally {
    await context.close();
  }
}
