async (page) => {
  const results = [];
  for (const label of ["ply", "packed"]) {
    await page.goto("http://127.0.0.1:8198/cases/" + label + "/index.html");
    await page.waitForFunction(() => window.box2Viewer?.ready);
    await page.waitForTimeout(1000);
    results.push(await page.evaluate(() => ({
      full: window.box2Viewer.fullGaussianCount, box: window.box2Viewer.boxGaussianCount,
      frames: window.box2Viewer.frames, keys: Object.keys(window.box2Viewer)
    })));
  }
  return results;
}
