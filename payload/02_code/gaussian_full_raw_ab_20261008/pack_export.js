import { PackedSplats } from "@sparkjsdev/spark";
import { encodePackedAsset, decodePackedAsset, digestBytes, verifySparkBuild } from "./packed_splat_asset.js";

const status = document.getElementById("status");
document.getElementById("pack").addEventListener("click", async event => {
  event.target.disabled = true;
  try {
    const completeStarted = performance.now();
    const sourceReply = await fetch("../../ASSET_SOURCES.json");
    if (!sourceReply.ok) throw new Error("The fresh asset source manifest is unavailable");
    const sources = await sourceReply.json();
    const branch = location.pathname.split("/")[1];
    const records = [];
    for (const label of ["scene", "box"]) {
      status.textContent = "正在生成 " + label;
      const source = sources[label];
      await verifySparkBuild("../../../shared-build/vendor/spark.module.js", source.spark_sha256);
      const response = await fetch("./"+label+".ply");
      if (!response.ok) throw new Error("The fresh source PLY is unavailable");
      const bytes = await response.arrayBuffer();
      if (bytes.byteLength !== source.source_bytes || await digestBytes(bytes) !== source.source_ply_sha256) throw new Error("The source PLY identity changed");
      const native = new PackedSplats({ fileBytes: bytes, fileName: label+".ply", lod: false });
      await native.initialized;
      if (native.numSplats !== source.numSplats || native.getNumSh() !== 3) throw new Error("The native population or SH orders changed");
      const started = performance.now();
      const encoded = await encodePackedAsset(native, { spark_sha256: source.spark_sha256, source_ply_sha256: source.source_ply_sha256 });
      const decoded = await decodePackedAsset(encoded.bytes.buffer, source);
      for (const name of ["packedArray", "sh1", "sh2", "sh3"]) {
        const a = name === "packedArray" ? native.packedArray : native.extra[name];
        const b = name === "packedArray" ? decoded.packed.packedArray : decoded.packed.extra[name];
        if (a.length !== b.length || !a.every((value, index) => value === b[index])) throw new Error("A complete native packed array changed: "+name);
      }
      const publication = await fetch("/api/packed/"+branch+"/"+label, { method: "POST", body: encoded.bytes,
        headers: { "Content-Type": "application/octet-stream" } });
      if (!publication.ok) throw new Error("The validated local asset publication failed: "+await publication.text());
      records.push({ label, numSplats: native.numSplats, complete_native_arrays_equal: true,
        source_bytes: bytes.byteLength, asset_bytes: encoded.bytes.byteLength,
        encode_roundtrip_publish_ms: performance.now()-started, publication: await publication.json() });
      native.dispose();
      decoded.packed.dispose();
    }
    const result = { status: "both_native_assets_published", records,
      whole_pack_wall_ms: performance.now()-completeStarted };
    const receipt = await fetch("/api/packing-receipt/"+branch, { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(result) });
    const receiptBody = await receipt.text();
    if (!receipt.ok) throw new Error("The native oracle receipt was not accepted: "+receiptBody);
    const receiptResult = JSON.parse(receiptBody);
    if (receiptResult.status !== "native_array_and_complete_asset_oracles_recorded")
      throw new Error("The native oracle receipt acknowledgement changed");
    status.textContent = JSON.stringify(result, null, 2);
    status.dataset.complete = "true";
  } catch (error) {
    status.textContent = String(error?.stack ?? error);
    status.dataset.failed = "true";
  }
});
