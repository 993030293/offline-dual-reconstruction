import { readAssetArtifact, assetCacheTrace } from "./persistent_asset_cache.js";
import { PackedSplats } from "@sparkjsdev/spark";

const MAGIC = "SPKPK001";
const LIMIT = 256 * 1024 * 1024;
const NAMES = ["packedArray", "sh1", "sh2", "sh3"];

export async function digestBytes(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map(x => x.toString(16).padStart(2, "0")).join("");
}

function checkEndian() {
  if (new Uint8Array(new Uint32Array([0x01020304]).buffer)[0] !== 4) {
    throw new Error("The versioned packed asset requires a little-endian browser");
  }
}

function sourceArrays(packed) {
  return [packed.packedArray, packed.extra.sh1, packed.extra.sh2, packed.extra.sh3];
}

export async function encodePackedAsset(packed, provenance) {
  checkEndian();
  await packed.initialized;
  if (packed.lod || packed.getNumSh() !== 3 || packed.maxSh !== 3) {
    throw new Error("This exact-display asset requires all three SH orders and no LoD");
  }
  const arrays = sourceArrays(packed);
  let offset = 0;
  const buffers = [];
  for (let i = 0; i < arrays.length; i++) {
    const array = arrays[i];
    if (!(array instanceof Uint32Array)) throw new Error("Unexpected packed buffer type");
    const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
    buffers.push({ name: NAMES[i], offset, length: array.length, byteLength: array.byteLength, sha256: await digestBytes(bytes) });
    offset += array.byteLength;
  }
  const metadata = { schema: 1, provenance, numSplats: packed.numSplats, maxSh: packed.maxSh,
    splatEncoding: packed.splatEncoding, buffers, payloadBytes: offset };
  const header = new TextEncoder().encode(JSON.stringify(metadata));
  const payloadStart = 16 + Math.ceil(header.byteLength / 4) * 4;
  if (header.byteLength > 16384 || payloadStart + offset > LIMIT) throw new Error("Packed export exceeds its declared diagnostic budget");
  const output = new Uint8Array(payloadStart + offset);
  output.set(new TextEncoder().encode(MAGIC), 0);
  const words = new DataView(output.buffer);
  words.setUint32(8, header.byteLength, true);
  words.setUint32(12, payloadStart, true);
  output.set(header, 16);
  for (let i = 0; i < arrays.length; i++) {
    output.set(new Uint8Array(arrays[i].buffer, arrays[i].byteOffset, arrays[i].byteLength), payloadStart + buffers[i].offset);
  }
  return { bytes: output, metadata };
}

export async function decodePackedAsset(buffer, expected) {
  checkEndian();
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 16 || buffer.byteLength > LIMIT) throw new Error("Invalid packed asset byte budget");
  const bytes = new Uint8Array(buffer), words = new DataView(buffer);
  if (new TextDecoder().decode(bytes.subarray(0, 8)) !== MAGIC) throw new Error("Unsupported packed asset magic/version");
  const headerLength = words.getUint32(8, true), payloadStart = words.getUint32(12, true);
  if (headerLength > 16384 || payloadStart !== 16 + Math.ceil(headerLength / 4) * 4 || payloadStart > buffer.byteLength) {
    throw new Error("Invalid packed asset header bounds");
  }
  const metadata = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes.subarray(16, 16 + headerLength)));
  if (metadata.schema !== 1 || metadata.maxSh !== 3 || !Number.isSafeInteger(metadata.numSplats) || metadata.numSplats <= 0) {
    throw new Error("Invalid packed asset schema or SH/count contract");
  }
  if (expected.numSplats !== undefined && metadata.numSplats !== expected.numSplats) throw new Error("Packed asset population differs from its source");
  if (metadata.provenance?.spark_sha256 !== expected.spark_sha256 || metadata.provenance?.source_ply_sha256 !== expected.source_ply_sha256) {
    throw new Error("Packed asset source/build identity mismatch");
  }
  if (!Array.isArray(metadata.buffers) || metadata.buffers.length !== NAMES.length || metadata.payloadBytes !== buffer.byteLength - payloadStart) {
    throw new Error("Missing buffers or truncated packed asset");
  }
  const encodingKeys = ["rgbMin", "rgbMax", "lnScaleMin", "lnScaleMax", "sh1Max", "sh2Max", "sh3Max"];
  if (!encodingKeys.every(key => Number.isFinite(metadata.splatEncoding?.[key])) || metadata.splatEncoding.lodOpacity !== false) {
    throw new Error("Unsupported or nonfinite packed encoding ranges");
  }
  const arrays = {};
  const integrityStarted = performance.now();
  const integrityParts = [];
  let offset = 0;
  for (let i = 0; i < NAMES.length; i++) {
    const row = metadata.buffers[i];
    if (row.name !== NAMES[i] || row.offset !== offset || !Number.isSafeInteger(row.length) || row.length <= 0
        || row.byteLength !== row.length * 4 || payloadStart + offset + row.byteLength > buffer.byteLength) {
      throw new Error("Invalid packed buffer layout");
    }
    const part = bytes.subarray(payloadStart + offset, payloadStart + offset + row.byteLength);
    if (await digestBytes(part) !== row.sha256) throw new Error("Packed buffer checksum mismatch");
    arrays[row.name] = new Uint32Array(buffer, payloadStart + offset, row.length);
    offset += row.byteLength;
  }
  if (offset !== metadata.payloadBytes || arrays.packedArray.length < metadata.numSplats * 4
      || arrays.sh1.length < metadata.numSplats * 2 || arrays.sh2.length < metadata.numSplats * 4 || arrays.sh3.length < metadata.numSplats * 4) {
    throw new Error("Packed buffer capacity would truncate the requested scene");
  }
  (globalThis.assetIntegrityTrace ??= []).push({
    payloadBytes: metadata.payloadBytes, buffers: metadata.buffers.length,
    integrity_ms: performance.now() - integrityStarted, all_native_checks_retained: true });
  const packed = new PackedSplats({ packedArray: arrays.packedArray, numSplats: metadata.numSplats,
    extra: { sh1: arrays.sh1, sh2: arrays.sh2, sh3: arrays.sh3 }, splatEncoding: metadata.splatEncoding, lod: false });
  packed.maxSh = metadata.maxSh;
  if (packed.numSplats !== metadata.numSplats || packed.getNumSh() !== 3) throw new Error("Native PackedSplats changed the asset population");
  return { packed, metadata };
}

export async function loadPackedAsset(url, expected) {
  const artifact = await readAssetArtifact(url, expected.asset_sha256);
  const buffer = artifact.buffer;
  if (typeof expected.asset_sha256 !== "string" || await digestBytes(buffer) !== expected.asset_sha256) {
    throw new Error("Packed asset file/header identity mismatch");
  }
  const result = await decodePackedAsset(buffer, expected);
  await artifact.commitVerified();
  globalThis.assetCacheTrace = assetCacheTrace;
  return result.packed;
}

export async function verifySparkBuild(url, expectedHash) {
  const artifact = await readAssetArtifact(url, expectedHash);
  if (await digestBytes(artifact.buffer) !== expectedHash) {
    throw new Error("The served Spark build differs from the packed asset contract");
  }
  await artifact.commitVerified();
}

