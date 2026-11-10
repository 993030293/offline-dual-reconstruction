// Observe committed native allocation dimensions after readPause at the existing readback boundary.
export function installOrderingAllocationCostV2({ spark }) {
  const sortSource = spark.driveSort.toString(), original = spark.readbackDepth, events = [];
  const allocation = "const ordering = new Uint32Array(this.maxSplats);";
  if (sortSource.split(allocation).length !== 2 || sortSource.split("await this.readbackDepth({").length !== 2 || sortSource.indexOf(allocation) > sortSource.indexOf("await this.readbackDepth({")) throw new Error("The native allocation-before-readback source boundary changed");
  let active = 0, disposed = false;
  const totals = { actualNativeAllocationAtReadbackCalls: 0, verifiedNativeNewArrayBytes: 0, nativeErrors: 0, minimumAllocationBytes: null, maximumAllocationBytes: 0 };
  const wrapped = function (...args) {
    if (disposed) throw new Error("The native allocation observer was already disposed");
    const input = args[0], words = this.maxSplats;
    if (!this.sorting || !input || input.renderer !== this.renderer || input.readback !== this.readback32 || !Number.isSafeInteger(words) || words < 16384 || events.length >= 4096) throw new Error("The original committed native allocation call contract changed");
    const bytes = words * Uint32Array.BYTES_PER_ELEMENT;
    totals.actualNativeAllocationAtReadbackCalls++; totals.verifiedNativeNewArrayBytes += bytes;
    totals.minimumAllocationBytes = totals.minimumAllocationBytes === null ? bytes : Math.min(totals.minimumAllocationBytes, bytes);
    totals.maximumAllocationBytes = Math.max(totals.maximumAllocationBytes, bytes);
    events.push({ bytes, rendererMaxSplats: words, currentMaxSplats: input.current.maxSplats,
      numSplats: input.numSplats, mappingVersion: input.current.mappingVersion, originalReadPause: this.readPause });
    active++;
    let result;
    try { result = original.apply(this, args); }
    catch (error) { active--; totals.nativeErrors++; throw error; }
    return Promise.resolve(result).then(value => value, error => { totals.nativeErrors++; throw error; }).finally(() => { active--; });
  };
  spark.readbackDepth = wrapped;
  const snapshot = () => ({ totals: { ...totals }, events: events.map(row => ({ ...row })), active, disposed,
    retainedOriginalImageDataBytes: spark.orderingTexture?.image.data.byteLength ?? 0,
    byte_count_is_source_verified_native_allocation_requests_not_allocator_wall_time_or_peak_RSS: true });
  return {
    snapshot,
    dispose() {
      if (disposed || active || spark.readbackDepth !== wrapped) throw new Error("The native allocation observer cannot restore its alias");
      spark.readbackDepth = original; disposed = true;
      return { ...snapshot(), originalAliasRestored: spark.readbackDepth === original };
    },
  };
}
