import { createOrderingScratchPool } from "./ordering_scratch_pool.js";

// Execute the release's real WASM sort32 worker; the independent worker is always disposed.
export async function runOrderingScratchGeneric(nativeWorker) {
  if (!nativeWorker || typeof nativeWorker.call !== "function") throw new Error("A real native sort worker is required");
  const worker = new nativeWorker.constructor(), pool = createOrderingScratchPool();
  const cases = [], checks = {}, protectedRows = [];
  let oracleNewArrayBytes = 0, referenceNewOrderingBytes = 0, referenceNewOrderingRequests = 0;
  function equal(a, b) {
    if (a.constructor !== b.constructor || a.length !== b.length) return false;
    for (let i = 0; i < a.length; ++i) if (a[i] !== b[i]) return false;
    return true;
  }
  function isZero(a) { for (let i = 0; i < a.length; ++i) if (a[i] !== 0) return false; return true; }
  function expectThrow(fn, label) {
    try { fn(); } catch (error) { checks[label] = true; return String(error); }
    checks[label] = false; throw new Error("Expected ownership error was not forwarded: " + label);
  }
  function depths(words, num, pattern) {
    const values = new Uint32Array(words); oracleNewArrayBytes += values.byteLength;
    for (let i = 0; i < num; ++i) {
      if (pattern === "zeros") values[i] = 0;
      else if (pattern === "duplicate") values[i] = i % 3 ? 0x3f800000 : 0;
      else if (pattern === "high_bits") values[i] = [0, 0xffffffff, 0x80000000, 0x7fffffff, 0x3f800000, 0xbf800000, 0x00800000][i % 7];
      else values[i] = i % 17 ? 0x3f000000 + ((Math.imul(i, 2654435761) >>> 0) & 0xffffff) : 0;
    }
    return values;
  }
  async function test(words, num, pattern, retain = false) {
    const input = depths(Math.max(words, num), num, pattern), referenceReadback = input.slice();
    oracleNewArrayBytes += referenceReadback.byteLength;
    const referenceOrdering = new Uint32Array(words);
    referenceNewOrderingBytes += referenceOrdering.byteLength; referenceNewOrderingRequests++;
    const lease = pool.acquire(words);
    const baselineZero = isZero(referenceOrdering), candidateZero = isZero(lease.ordering);
    const candidateInput = input.slice(); oracleNewArrayBytes += candidateInput.byteLength;
    const reference = await worker.call("sortSplats32", { numSplats: num, readback: referenceReadback, ordering: referenceOrdering });
    const candidate = await worker.call("sortSplats32", { numSplats: num, readback: candidateInput, ordering: lease.ordering });
    const allExact = equal(reference.ordering, candidate.ordering) && equal(reference.readback, candidate.readback) && reference.activeSplats === candidate.activeSplats;
    const detached = words === 0 || lease.ordering.byteLength === 0;
    const originalReadback = equal(candidate.readback, input);
    const row = { words, num, pattern, origin: lease.origin, activeSplats: candidate.activeSplats,
      baselineZero, candidateZero, fullOrderingExact: equal(reference.ordering, candidate.ordering),
      fullReadbackExact: equal(reference.readback, candidate.readback), originalReadback,
      activeSplatsExact: reference.activeSplats === candidate.activeSplats, actualTransferDetachedBorrow: detached };
    if (!allExact || !baselineZero || !candidateZero || !detached || !originalReadback) throw new Error("Real sort32 scratch equivalence failed: " + JSON.stringify(row));
    if (retain) {
      const copy = candidate.ordering.slice(); oracleNewArrayBytes += copy.byteLength;
      protectedRows.push({ array: candidate.ordering, copy });
    }
    pool.complete(lease.id, candidate.ordering, { consumed: true, retained: retain ? candidate.ordering : null });
    for (const protectedRow of protectedRows) {
      if (protectedRow.array.byteLength !== protectedRow.copy.byteLength || !equal(protectedRow.array, protectedRow.copy)) throw new Error("Retained original image data changed or detached");
    }
    cases.push(row);
  }
  try {
    // Capacities cover zero, exact row boundaries, partial populations, high bits, growth and shrink.
    for (const [words, num, pattern, held] of [
      [0, 0, "zeros", false], [0, 0, "zeros", false],
      [16384, 0, "zeros", true], [16384, 1, "zeros", false],
      [16384, 16383, "mixed", false], [16384, 16384, "duplicate", false],
      [32768, 16385, "high_bits", true], [32768, 32767, "mixed", false],
      [32768, 32768, "duplicate", false], [131072, 100003, "mixed", false],
      [131072, 131071, "high_bits", false], [16384, 7, "mixed", false],
      [16384, 16384, "mixed", false], [540672, 526886, "mixed", true],
      [540672, 526886, "duplicate", false], [540672, 60451, "high_bits", false],
      [540672, 526886, "zeros", false], [32768, 19991, "mixed", false],
    ]) await test(words, num, pattern, held);
    checks.full_real_worker_outputs_inputs_and_transfer_exact = cases.every(row => row.fullOrderingExact && row.fullReadbackExact && row.activeSplatsExact && row.baselineZero && row.candidateZero && row.actualTransferDetachedBorrow && row.originalReadback);
    checks.protected_original_image_data_remains_attached_and_exact = protectedRows.every(row => equal(row.array, row.copy) && row.array.byteLength === row.copy.byteLength);
    const before = pool.snapshot();
    const lease = pool.acquire(32768);
    expectThrow(() => pool.acquire(32768), "second_borrow_rejected");
    expectThrow(() => pool.complete(lease.id, lease.ordering), "premature_return_rejected");
    expectThrow(() => pool.dispose(), "dispose_while_leased_rejected");
    expectThrow(() => pool.complete(lease.id + 1, lease.ordering, { consumed: true }), "unknown_lease_rejected");
    expectThrow(() => pool.complete(lease.id, new Uint32Array(1), { consumed: true }), "wrong_capacity_rejected");
    oracleNewArrayBytes += 4;
    let actualWorkerError;
    try { await worker.call("R164_unknown_native_RPC", { ordering: lease.ordering }); }
    catch (error) { actualWorkerError = String(error); }
    checks.actual_native_worker_callback_error_forwarded = actualWorkerError?.includes("Unknown worker RPC") === true;
    checks.actual_error_transfer_detaches_only_borrow = lease.ordering.byteLength === 0;
    pool.abort(lease.id);
    await test(32768, 17, "mixed");
    await test(32768, 19991, "mixed");
    checks.known_error_invalidates_scratch_without_swallowing = pool.snapshot().totals.abortedLeases === 1;
    const retained = protectedRows[0].array, alias = new Uint32Array(retained.buffer);
    pool.protect(alias);
    const isolated = createOrderingScratchPool(), a = isolated.acquire(16384);
    isolated.protect(alias);
    isolated.complete(a.id, alias, { consumed: true });
    const b = isolated.acquire(16384);
    checks.all_retained_backing_buffer_aliases_excluded = b.origin === "new" && retained.byteLength === 65536;
    isolated.complete(b.id, b.ordering, { consumed: true });
    const lost = isolated.acquire(16384); isolated.complete(lost.id, lost.ordering, { consumed: true });
    structuredClone(lost.ordering, { transfer: [lost.ordering.buffer] });
    const replaced = isolated.acquire(16384);
    checks.detached_unknown_free_state_allocates_original_fresh_and_records = replaced.origin === "new" && isolated.snapshot().totals.unknownStateDiscards === 1;
    isolated.complete(replaced.id, replaced.ordering, { consumed: true });
    const isolatedCost = isolated.dispose();
    const final = pool.dispose();
    checks.actual_reuse_and_full_zero_clear_present = before.totals.reusedRequests > 0 && final.totals.fullClearWords > 0;
    checks.capacity_changes_original_fresh_allocation = final.totals.capacityDiscards >= 4;
    checks.no_owned_array_lease_at_dispose = final.activeLease === null && final.disposed && isolatedCost.disposed;
    return { checks, cases, pool: final, isolated: isolatedCost, actualWorkerError,
      referenceNewOrderingBytes, referenceNewOrderingRequests, oracleNewArrayBytes,
      totalKnownNewArrayRequestBytes: final.totals.newRequestBytes + isolatedCost.totals.newRequestBytes + referenceNewOrderingBytes + oracleNewArrayBytes,
      status: Object.values(checks).every(Boolean) ? "real_sort32_component_passed0" : "real_sort32_component_failed0",
      new_optimization_count: 0, full_pipeline_unknown_object_GT_and_speed_verified: false };
  } finally {
    window.r164ScratchGenericProgress = { checks, cases, pool: pool.snapshot(), referenceNewOrderingBytes,
      referenceNewOrderingRequests, oracleNewArrayBytes, new_optimization_count: 0 };
    worker.dispose();
  }
}
