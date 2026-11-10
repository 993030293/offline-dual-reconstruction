import * as THREE from "three";
import { Readback, utils } from "@sparkjsdev/spark";
import { createOrderingScratchPool } from "./ordering_scratch_pool.js";

// The dynamically bound method retains the installed Spark implementation and its MIT attribution.
// Source of driveSort: Spark, Copyright (c) 2025 WORLD LABS TECHNOLOGIES, INC.
// The unmodified full license is retained beside every executed source snapshot as SPARK-LICENSE.txt.
export function installOrderingScratchNative({ spark, enabled = false }) {
  const originalSort = spark.driveSort, originalReadback = spark.readbackDepth;
  const nativeSource = originalSort.toString(), allocation = "const ordering = new Uint32Array(this.maxSplats);";
  const recycling = "this.sorting = false;\n\t\tthis.setDirty();";
  const pool = createOrderingScratchPool(), retained = [], events = [];
  const totals = { allocationRequests: 0, newArrayRequests: 0, newArrayRequestBytes: 0,
    reusedRequests: 0, fullClearWords: 0, initialNativeRequests: 0, firstNativeWorkerCallsUnobserved: 0,
    oracleNewArrayRequestBytes: 0, retainedCopyNewArrayBytes: 0, fullOrderingComparisons: 0,
    fullReadbackComparisons: 0, activeSplatsComparisons: 0, fullZeroInputComparisons: 0,
    nativeErrors: 0, consumedReturns: 0, protectedDataComparisons: 0 };
  if (nativeSource.split(allocation).length !== 2 || nativeSource.split(recycling).length !== 2 || !nativeSource.startsWith("async driveSort()") || typeof utils.uploadU32DataTextureRows !== "function") throw new Error("The exact installed native sort or exported dependency contract changed");
  let compiled = null, referenceWorker = null, workerBinding = null, currentLease = null, disposed = false, active = 0, serial = 0;
  function equal(a, b) {
    if (a.constructor !== b.constructor || a.length !== b.length) return false;
    for (let i = 0; i < a.length; ++i) if (a[i] !== b[i]) return false;
    return true;
  }
  function protectAndCheck() {
    const imageData = spark.orderingTexture?.image.data;
    if (imageData && !retained.some(row => row.array === imageData)) {
      pool.protect(imageData);
      const copy = imageData.slice(); totals.retainedCopyNewArrayBytes += copy.byteLength;
      retained.push({ array: imageData, copy });
    }
    for (const row of retained) {
      totals.protectedDataComparisons++;
      if (row.array.byteLength !== row.copy.byteLength || !equal(row.array, row.copy)) throw new Error("Original retained texture data changed or detached");
    }
  }
  function attach() {
    const worker = spark.sortWorker;
    if (!worker) return;
    if (workerBinding) {
      if (workerBinding.worker !== worker) throw new Error("The native sort worker changed during the owned context");
      return;
    }
    const ctor = worker.constructor;
    if (typeof ctor !== "function" || typeof worker.call !== "function") throw new Error("The actual native worker interface changed");
    referenceWorker = new ctor();
    const originalCall = worker.call;
    const wrappedCall = async function (...args) {
      if (args[0] !== "sortSplats32") return originalCall.apply(this, args);
      const input = args[1];
      if (!(input.ordering instanceof Uint32Array) || !(input.readback instanceof Uint32Array)) throw new Error("The original full sort32 input types changed");
      for (let i = 0; i < input.ordering.length; ++i) if (input.ordering[i] !== 0) throw new Error("The complete native ordering input is not all zero");
      totals.fullZeroInputComparisons++;
      const referenceOrdering = new Uint32Array(input.ordering.length), referenceReadback = input.readback.slice();
      totals.oracleNewArrayRequestBytes += referenceOrdering.byteLength + referenceReadback.byteLength;
      const result = await originalCall.apply(this, args);
      const oracle = await referenceWorker.call("sortSplats32", { numSplats: input.numSplats, readback: referenceReadback, ordering: referenceOrdering });
      if (!equal(result.ordering, oracle.ordering) || !equal(result.readback, oracle.readback) || result.activeSplats !== oracle.activeSplats) throw new Error("The complete native sort32 ordering/readback/active population differs from the real original worker");
      totals.fullOrderingComparisons++; totals.fullReadbackComparisons++; totals.activeSplatsComparisons++;
      return result;
    };
    worker.call = wrappedCall; workerBinding = { worker, originalCall, wrappedCall };
    const body = nativeSource.replace("async driveSort()", "async function driveSort()")
      .replace(allocation, "const ordering = requestOrdering(this.maxSplats);")
      .replace(recycling, "returnOrdering(result.ordering);\n\t\t" + recycling);
    const inverse = body.replace("async function driveSort()", "async driveSort()")
      .replace("const ordering = requestOrdering(this.maxSplats);", allocation)
      .replace("returnOrdering(result.ordering);\n\t\t" + recycling, recycling);
    if (inverse !== nativeSource) throw new Error("The local native method has changes beyond allocation and consumed-result recycling");
    compiled = new Function("THREE", "Readback", "SplatWorker", "uploadU32DataTextureRows", "requestOrdering", "returnOrdering", "return (" + body + ");")(
      THREE, Readback, ctor, utils.uploadU32DataTextureRows, requestOrdering, returnOrdering);
  }
  function requestOrdering(words) {
    if (disposed || currentLease) throw new Error("The owned native scratch request overlaps another lease");
    protectAndCheck();
    let array, origin;
    if (enabled) {
      const lease = pool.acquire(words, spark.orderingTexture?.image.data);
      currentLease = lease; array = lease.ordering; origin = lease.origin;
    } else {
      array = new Uint32Array(words); origin = "new";
      currentLease = { id: ++serial, words, ordering: array, origin };
    }
    totals.allocationRequests++;
    if (origin === "reuse") { totals.reusedRequests++; totals.fullClearWords += words; }
    else { totals.newArrayRequests++; totals.newArrayRequestBytes += array.byteLength; }
    if (events.length >= 4096) throw new Error("The registered native scratch event bound was exceeded");
    events.push({ kind: "request", words, bytes: words * 4, origin, readPause: spark.readPause });
    return array;
  }
  function returnOrdering(ordering) {
    if (!currentLease || ordering.length !== currentLease.words) throw new Error("The native result has no compatible consumed scratch lease");
    protectAndCheck();
    if (enabled) pool.complete(currentLease.id, ordering, { consumed: true, retained: spark.orderingTexture?.image.data });
    currentLease = null; totals.consumedReturns++;
  }
  const readback = function (...args) {
    if (disposed) throw new Error("The native scratch observer is disposed");
    if (!currentLease) {
      const bytes = this.maxSplats * 4;
      if (!Number.isSafeInteger(this.maxSplats) || this.maxSplats < 16384 || !this.sorting) throw new Error("The original first native allocation boundary changed");
      totals.allocationRequests++; totals.newArrayRequests++; totals.initialNativeRequests++; totals.newArrayRequestBytes += bytes;
      events.push({ kind: "initial_native_request", words: this.maxSplats, bytes, origin: "new", readPause: this.readPause });
    }
    return originalReadback.apply(this, args);
  };
  const sort = async function (...args) {
    if (disposed) throw new Error("The native scratch wrapper is disposed");
    active++;
    try {
      const nativeRoute = !compiled;
      const prior = totals.initialNativeRequests;
      const result = await (compiled ?? originalSort).apply(this, args);
      if (nativeRoute) {
        totals.firstNativeWorkerCallsUnobserved += totals.initialNativeRequests - prior;
        protectAndCheck(); attach();
      }
      return result;
    } catch (error) {
      totals.nativeErrors++;
      if (currentLease) {
        if (enabled) pool.abort(currentLease.id);
        currentLease = null;
      }
      throw error;
    } finally { active--; }
  };
  spark.readbackDepth = readback; spark.driveSort = sort;
  function snapshot() {
    return { enabled, totals: { ...totals }, events: events.map(row => ({ ...row })), pool: pool.snapshot(),
      active, activeLease: currentLease?.id ?? null, disposed, privateNativeConstructorBound: Boolean(compiled),
      fullOriginalMethodInverseExact: true, protectedOriginalImageDataCount: retained.length,
      byte_count_is_new_array_requests_not_OS_memory_RSS_GPU_or_speed: true };
  }
  return { snapshot, attachWorker: attach,
    dispose() {
      if (disposed || active || currentLease || spark.driveSort !== sort || spark.readbackDepth !== readback) throw new Error("Owned native scratch work or aliases are not drained");
      if (workerBinding) {
        if (workerBinding.worker.call !== workerBinding.wrappedCall) throw new Error("The owned native worker alias changed before restoration");
        workerBinding.worker.call = workerBinding.originalCall;
      }
      referenceWorker?.dispose(); protectAndCheck();
      spark.driveSort = originalSort; spark.readbackDepth = originalReadback;
      const poolFinal = pool.dispose(); disposed = true;
      return { ...snapshot(), pool: poolFinal, originalSortAliasRestored: spark.driveSort === originalSort,
        originalReadbackAliasRestored: spark.readbackDepth === originalReadback,
        originalWorkerAliasRestored: !workerBinding || workerBinding.worker.call === workerBinding.originalCall };
    },
  };
}
