// Count native ordering allocations from the verified synchronous allocation site without changing it.
export function installOrderingAllocationCost({ spark }) {
  const original = spark.driveSort, source = original.toString(), events = [];
  const allocation = "const ordering = new Uint32Array(this.maxSplats);";
  if (source.split(allocation).length !== 2 || !source.includes("this.lastSortTime = now;")) throw new Error("The native ordering allocation site changed");
  let active = 0, disposed = false;
  const totals = { callsIncludingNoop: 0, actualNativeSortStarts: 0, verifiedNativeNewArrayBytes: 0, nativeErrors: 0, minimumAllocationBytes: null, maximumAllocationBytes: 0 };
  const wrapped = function (...args) {
    if (disposed) throw new Error("The ordering allocation observer was already disposed");
    const previous = this.lastSortTime;
    totals.callsIncludingNoop++; active++;
    let result;
    try {
      result = original.apply(this, args);
      if (this.lastSortTime !== previous) {
        const words = this.maxSplats;
        if (!Number.isSafeInteger(words) || words < 16384 || events.length >= 4096) throw new Error("The actual native allocation dimensions exceeded the registered bound");
        const bytes = words * Uint32Array.BYTES_PER_ELEMENT;
        totals.actualNativeSortStarts++; totals.verifiedNativeNewArrayBytes += bytes;
        totals.minimumAllocationBytes = totals.minimumAllocationBytes === null ? bytes : Math.min(totals.minimumAllocationBytes, bytes);
        totals.maximumAllocationBytes = Math.max(totals.maximumAllocationBytes, bytes);
        events.push({ bytes, rendererMaxSplats: this.maxSplats, currentMaxSplats: this.current.maxSplats,
          numSplats: this.current.numSplats, mappingVersion: this.current.mappingVersion });
      }
    } catch (error) { active--; totals.nativeErrors++; throw error; }
    return Promise.resolve(result).then(value => value, error => { totals.nativeErrors++; throw error; }).finally(() => { active--; });
  };
  spark.driveSort = wrapped;
  const snapshot = () => ({ totals: { ...totals }, events: events.map(row => ({ ...row })), active, disposed,
    retainedOriginalImageDataBytes: spark.orderingTexture?.image.data.byteLength ?? 0,
    byte_count_is_source_verified_native_allocation_requests_not_allocator_wall_time_or_peak_RSS: true });
  return {
    snapshot,
    dispose() {
      if (disposed || active || spark.driveSort !== wrapped) throw new Error("The native allocation observer cannot restore its alias");
      spark.driveSort = original; disposed = true;
      return { ...snapshot(), originalAliasRestored: spark.driveSort === original };
    },
  };
}
