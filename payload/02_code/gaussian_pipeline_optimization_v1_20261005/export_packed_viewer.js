async (page) => {
  const output = "C:/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v1_20261005/p10_packed_viewer_001/cases/packed";
  const records = [];
  await page.waitForFunction(() => window.box2Viewer?.ready, undefined, { timeout: 60000 });
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
      if (sourceBytes.byteLength !== source.source_bytes || await digestBytes(sourceBytes) !== source.source_ply_sha256) throw new Error("Served source PLY identity mismatch");
      const native = new PackedSplats({ fileBytes: sourceBytes, fileName: label + ".ply", lod: false });
      await native.initialized;
      if (native.numSplats !== source.numSplats || native.getNumSh() !== 3) throw new Error("Native source decoder changed the population or SH");
      const provenance = { spark_sha256: source.spark_sha256, source_ply_sha256: source.source_ply_sha256 };
      const encoded = await encodePackedAsset(native, provenance);
      const roundtrip = await decodePackedAsset(encoded.bytes.buffer, source);
      const checks = [];
      for (const name of ["packedArray", "sh1", "sh2", "sh3"]) {
        const a = name === "packedArray" ? native.packedArray : native.extra[name];
        const b = name === "packedArray" ? roundtrip.packed.packedArray : roundtrip.packed.extra[name];
        const equal = a.length === b.length && a.every((value, index) => value === b[index]);
        if (!equal) throw new Error("Roundtrip array differs: " + name);
        checks.push({ name: "exact_" + name, passed: true, words: a.length });
      }
      async function rejects(name, bytes, expected = source) {
        let rejected = false;
        try { await decodePackedAsset(bytes, expected); } catch { rejected = true; }
        if (!rejected) throw new Error("Invalid packed asset was accepted: " + name);
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
        numSplats: encoded.metadata.numSplats, maxSh: encoded.metadata.maxSh,
        checks, encode_and_contract_validation_ms: performance.now() - started };
    }, label);
    const downloadPromise = page.waitForEvent("download");
    await page.evaluate(label => {
      const url = URL.createObjectURL(new Blob([window.__packedExportBytes], { type: "application/octet-stream" }));
      window.__packedExportUrl = url;
      const link = document.createElement("a"); link.href = url; link.download = label + ".spk";
      document.body.appendChild(link); link.click(); link.remove();
    }, label);
    const download = await downloadPromise;
    await download.saveAs(output + "/" + label + ".spk");
    await page.evaluate(() => { URL.revokeObjectURL(window.__packedExportUrl); delete window.__packedExportBytes; delete window.__packedExportUrl; });
    records.push({ ...result, saved: output + "/" + label + ".spk" });
  }
  return { status: "native_buffers_exported_and_roundtrip_verified", records,
    original_float32_PLY_modified: false, browser_pixels_or_latency_validated_here: false, counted: false };
}
