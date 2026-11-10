// Retained texture data and all views of its buffer must never become transferable scratch.
export function createOrderingScratchPool() {
  const protectedArrays = new WeakSet(), protectedBuffers = new WeakSet(), events = [];
  const totals = { newRequests: 0, newRequestBytes: 0, reusedRequests: 0, fullClearWords: 0,
    protectedReturns: 0, capacityDiscards: 0, unknownStateDiscards: 0, abortedLeases: 0 };
  let free = null, active = null, sequence = 0, disposed = false;

  function attached(array) {
    if (!(array instanceof Uint32Array)) return false;
    try { new Uint32Array(array.buffer); return true; } catch { return false; }
  }
  function protectedData(array) {
    return protectedArrays.has(array) || protectedBuffers.has(array.buffer);
  }
  function protect(array) {
    if (disposed) throw new Error("The ordering pool is disposed");
    if (array === undefined || array === null) return;
    if (!attached(array)) throw new Error("Retained ordering data is already detached or has an invalid type");
    protectedArrays.add(array); protectedBuffers.add(array.buffer);
    if (free && protectedData(free)) free = null;
  }
  function record(row) {
    if (events.length >= 4096) throw new Error("The registered scratch observation limit was exceeded");
    events.push(row);
  }
  function acquire(words, retained = null) {
    if (disposed || active) throw new Error("Ordering scratch is unavailable while disposed or leased");
    if (!Number.isSafeInteger(words) || words < 0) throw new Error("Ordering capacity is invalid");
    protect(retained);
    let origin = "new", reason = "empty", ordering;
    if (free) {
      if (!attached(free) || free.byteOffset !== 0 || free.byteLength !== free.buffer.byteLength || protectedData(free)) {
        totals.unknownStateDiscards++; free = null; reason = "unknown_state";
      } else if (free.length !== words) {
        totals.capacityDiscards++; free = null; reason = "capacity_changed";
      } else {
        ordering = free; free = null; ordering.fill(0);
        origin = "reuse"; reason = "exact_capacity";
        totals.reusedRequests++; totals.fullClearWords += words;
      }
    }
    if (!ordering) {
      ordering = new Uint32Array(words);
      totals.newRequests++; totals.newRequestBytes += ordering.byteLength;
    }
    active = { id: ++sequence, words, ordering, origin };
    record({ kind: "acquire", id: active.id, words, bytes: words * 4, origin, reason });
    return { ...active };
  }
  function complete(id, result, { retained = null, consumed = false } = {}) {
    if (disposed || !active || active.id !== id || !consumed) throw new Error("Ordering return requires its active lease and completed original consumption");
    if (!attached(result) || result.byteOffset !== 0 || result.length !== active.words || result.byteLength !== result.buffer.byteLength) throw new Error("Returned native ordering has an incompatible capacity or ownership");
    protect(retained);
    const held = protectedData(result);
    if (held) totals.protectedReturns++;
    else free = result;
    record({ kind: "complete", id, words: result.length, retained: held });
    active = null;
  }
  function abort(id) {
    if (disposed || !active || active.id !== id) throw new Error("Ordering abort has no matching lease");
    record({ kind: "abort", id }); totals.abortedLeases++;
    active = null; free = null;
  }
  function snapshot() {
    return { totals: { ...totals }, events: events.map(row => ({ ...row })),
      activeLease: active?.id ?? null, freeWords: free?.length ?? null, disposed };
  }
  return { acquire, complete, abort, protect, snapshot,
    dispose() {
      if (disposed || active) throw new Error("Ordering scratch cannot dispose while leased or already disposed");
      free = null; disposed = true; return snapshot();
    } };
}
