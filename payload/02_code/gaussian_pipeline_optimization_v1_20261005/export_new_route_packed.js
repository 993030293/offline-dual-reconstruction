async (page) => {
  const initial = new URL(page.url());
  if (initial.hostname !== "127.0.0.1" || initial.port !== "8203") throw new Error("Unexpected staging server");
  const root = "C:/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v1_20261005/new_route_viewer_001";
  const records = [];
  for (const model of ["reference", "combined"]) {
    await page.goto(initial.origin + "/" + model + "/cases/packed/export.html");
    for (const label of ["scene", "box"]) {
      const result = await page.evaluate(async label => {
        const started = performance.now();
        const sources = await (await fetch("../../ASSET_SOURCES.json")).json();
        const source = sources[label];
        const { PackedSplats } = await import("@sparkjsdev/spark");
        const { encodePackedAsset, decodePackedAsset, digestBytes, verifySparkBuild } = await import("./packed_splat_asset.js");
        await verifySparkBuild("./vendor/spark.module.js", source.spark_sha256);
        const response = await fetch("./" + label + ".ply");
        if (!response.ok) throw new Error("Source PLY request failed");
        const sourceBytes = await response.arrayBuffer();
        if (sourceBytes.byteLength !== source.source_bytes || await digestBytes(sourceBytes) !== source.source_ply_sha256)
          throw new Error("Served source PLY identity mismatch");
        const native = new PackedSplats({ fileBytes: sourceBytes, fileName: label + ".ply", lod: false });
        await native.initialized;
        if (native.numSplats !== source.numSplats || native.getNumSh() !== 3) throw new Error("Native decoder changed population or SH");
        const encoded = await encodePackedAsset(native, { spark_sha256: source.spark_sha256, source_ply_sha256: source.source_ply_sha256 });
        const roundtrip = await decodePackedAsset(encoded.bytes.buffer, source);
        const checks = [];
        for (const name of ["packedArray", "sh1", "sh2", "sh3"]) {
          const a = name === "packedArray" ? native.packedArray : native.extra[name];
          const b = name === "packedArray" ? roundtrip.packed.packedArray : roundtrip.packed.extra[name];
          if (a.length !== b.length || !a.every((value, index) => value === b[index])) throw new Error("Roundtrip differs: " + name);
          checks.push({ name: "exact_" + name, passed: true, words: a.length });
        }
        async function rejects(name, bytes, expected = source) {
          let rejected = false;
          try { await decodePackedAsset(bytes, expected); } catch { rejected = true; }
          if (!rejected) throw new Error("Invalid packed asset accepted: " + name);
          checks.push({ name, passed: true });
        }
        const corrupt = encoded.bytes.slice(); corrupt[corrupt.length - 1] ^= 1;
        await rejects("corrupt_payload_rejected", corrupt.buffer);
        await rejects("truncated_asset_rejected", encoded.bytes.buffer.slice(0, -4));
        await rejects("wrong_build_rejected", encoded.bytes.buffer, { ...source, spark_sha256: "0".repeat(64) });
        await rejects("wrong_source_rejected", encoded.bytes.buffer, { ...source, source_ply_sha256: "0".repeat(64) });
        await rejects("wrong_population_rejected", encoded.bytes.buffer, { ...source, numSplats: source.numSplats + 1 });
        window.__packedExportBytes = encoded.bytes;
        native.dispose(); roundtrip.packed.dispose();
        return { label, source_bytes: sourceBytes.byteLength, packed_bytes: encoded.bytes.byteLength,
          numSplats: encoded.metadata.numSplats, maxSh: encoded.metadata.maxSh, checks,
          encode_and_contract_validation_ms: performance.now() - started };
      }, label);
      const downloadPromise = page.waitForEvent("download");
      await page.evaluate(label => {
        const url = URL.createObjectURL(new Blob([window.__packedExportBytes], { type: "application/octet-stream" }));
        window.__packedExportUrl = url;
        const link = document.createElement("a"); link.href = url; link.download = label + ".spk";
        document.body.appendChild(link); link.click(); link.remove();
      }, label);
      const output = root + "/" + model + "/cases/packed/" + label + ".spk";
      await (await downloadPromise).saveAs(output);
      await page.evaluate(() => { URL.revokeObjectURL(window.__packedExportUrl); delete window.__packedExportBytes; delete window.__packedExportUrl; });
      records.push({ model, ...result, saved: output });
    }
  }
  return { status: "four_native_assets_exported_and_roundtrip_verified", records,
    original_float32_PLY_modified: false, browser_pixels_or_latency_validated_here: false, counted: false };
}
