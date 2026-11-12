// Cache complete immutable artifacts by their content identity, with native verification at every use.
const CACHE = "box2-native-verified-assets-v1";
const LIMIT = 64 * 1024 * 1024;
const MAX_ENTRIES = 3;
const MAX_BYTES = 128 * 1024 * 1024;
export const assetCacheTrace = [];

export async function readAssetArtifact(url, sha256) {
  if (!/^[0-9a-f]{64}$/.test(sha256) || !globalThis.caches) throw new Error("Verified artifact storage is unavailable or has an invalid identity");
  const cache = await caches.open(CACHE);
  const key = new URL("/.__box2_verified_asset__/" + sha256, location.origin).href;
  const started = performance.now();
  const cached = await cache.match(key);
  let response = cached;
  if (!response) {
    response = await fetch(url);
    if (!response.ok) throw new Error("Native artifact request failed: " + response.status);
  }
  const buffer = await response.arrayBuffer();
  if (!buffer.byteLength || buffer.byteLength > LIMIT) throw new Error("Cached native artifact exceeded its original byte budget");
  const row = { url, sha256, hit: Boolean(cached), bytes: buffer.byteLength,
    read_ms: performance.now() - started, committed_after_native_validation: false };
  assetCacheTrace.push(row);
  return {
    buffer,
    async commitVerified() {
      if (!cached) {
        const keys = await cache.keys();
        let bytes = 0;
        for (const request of keys) {
          const entry = await cache.match(request);
          const size = Number(entry?.headers.get("X-Native-Artifact-Bytes"));
          if (!Number.isSafeInteger(size) || size <= 0 || size > LIMIT) throw new Error("An immutable artifact cache entry has invalid bounds");
          bytes += size;
        }
        if (keys.length >= MAX_ENTRIES || bytes + buffer.byteLength > MAX_BYTES) {
          // Explicit FIFO eviction keeps the optional persistent store within its declared budget.
          for (const request of keys) {
            const entry = await cache.match(request);
            bytes -= Number(entry.headers.get("X-Native-Artifact-Bytes"));
            await cache.delete(request);
            if (bytes + buffer.byteLength <= MAX_BYTES && (await cache.keys()).length < MAX_ENTRIES) break;
          }
        }
        await cache.put(key, new Response(buffer, { headers: {
          "Content-Type": "application/octet-stream", "X-Native-Artifact-Bytes": String(buffer.byteLength),
          "X-Native-Artifact-SHA256": sha256 } }));
      }
      row.committed_after_native_validation = true;
    },
  };
}

export async function clearAssetArtifacts() {
  await caches.delete(CACHE);
}
