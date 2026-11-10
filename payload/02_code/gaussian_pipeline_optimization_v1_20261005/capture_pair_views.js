async (page) => {
  const root = "C:/lidar-camera-fusion-mask-robust-downsampling/output/playwright/gaussian_optimization_p0";
  const records = [];
  for (const label of ["legacy", "full"]) {
    await page.getByLabel("对照分支").selectOption(label);
    await page.waitForURL("**/cases/" + label + "/index.html");
    await page.waitForFunction(() => window.box2Viewer?.ready, undefined, { timeout: 60000 });
    await page.getByLabel("展示范围").selectOption("box");
    for (const preset of ["capture", "top", "drone"]) {
      if (preset === "capture") await page.getByRole("button", { name: "重置采集视角", exact: true }).click();
      else await page.getByRole("button", { name: preset === "top" ? "无人机俯视" : "无人机斜俯视", exact: true }).click();
      await page.waitForTimeout(500);
      const file = root + "/" + label + "_" + preset + ".png";
      await page.screenshot({ path: file, scale: "css" });
      records.push({ label, preset, file, camera: await page.evaluate(() => ({
        position: window.box2Viewer.getCameraPosition(), direction: window.box2Viewer.getCameraDirection(),
        projection: window.box2Viewer.projectionMatrix, mode: window.box2Viewer.projectionMode })) });
    }
  }
  await page.getByRole("button", { name: "重置采集视角", exact: true }).click();
  return { status: "images_captured_visual_review_pending", records, independently_known_free_view_reference: false };
}
