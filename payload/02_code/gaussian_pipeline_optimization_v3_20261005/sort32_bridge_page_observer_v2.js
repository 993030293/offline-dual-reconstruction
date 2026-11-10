// Intercept only the exact frozen SDK source, including its native URL revocation prefix.
export function installSort32BridgeObserver({ originalWorkerSource, observedWorkerSource }) {
  const enabled = new URLSearchParams(location.search).get("hooks") === "1";
  const nativeBlob = globalThis.Blob, nativeWorker = globalThis.Worker;
  const nativeCreate = URL.createObjectURL, nativeRevoke = URL.revokeObjectURL;
  const prefix = "(self.URL || self.webkitURL).revokeObjectURL(self.location.href);";
  const knownBlobs = new WeakSet(), knownURLs = new Set(), ownedWorkers = [], traces = [];
  const totals = { matchingBlobs: 0, matchingWorkers: 0, observedWorkers: 0, nativeOtherWorkers: 0,
    statusTraces: 0, originalURLsRevoked: 0, observedImportCopyBytes: 0, observedJSBackingRequestsBytes: 0,
    scalarReadCalls: 0, scalarWriteCalls: 0 };
  let disposed = false;
  const blobProxy = new Proxy(nativeBlob, {
    construct(target, args, newTarget) {
      const value = Reflect.construct(target, args, newTarget);
      if (Array.isArray(args[0]) && args[0].length === 2 && args[0][0] === prefix && args[0][1] === originalWorkerSource) {
        if (args[1]?.type !== "text/javascript;charset=utf-8") throw new Error("The frozen SDK worker blob MIME changed");
        knownBlobs.add(value); totals.matchingBlobs++;
      }
      return value;
    },
  });
  const create = function (...args) {
    const value = nativeCreate.apply(this, args);
    if (knownBlobs.has(args[0])) knownURLs.add(value);
    return value;
  };
  const workerProxy = new Proxy(nativeWorker, {
    construct(target, args, newTarget) {
      const matches = knownURLs.has(args[0]);
      let constructionArgs = args;
      if (matches) {
        totals.matchingWorkers++;
        if (enabled) {
          const blob = new nativeBlob([prefix, observedWorkerSource], { type: "text/javascript;charset=utf-8" });
          const url = nativeCreate.call(URL, blob);
          constructionArgs = [url, ...args.slice(1)];
          totals.observedWorkers++;
        }
      } else totals.nativeOtherWorkers++;
      const worker = Reflect.construct(target, constructionArgs, newTarget);
      if (matches && enabled) { nativeRevoke.call(URL, args[0]); totals.originalURLsRevoked++; }
      const listener = event => {
        const row = event.data?.status?.sort32_bridge_008;
        if (!row) return;
        if (!matches || !enabled || traces.length >= 4096) throw new Error("The bridge status ownership or diagnostic bound changed");
        traces.push(row); totals.statusTraces++;
        for (const key of ["observedImportCopyBytes", "observedJSBackingRequestsBytes", "scalarReadCalls", "scalarWriteCalls"]) totals[key] += row[key];
      };
      worker.addEventListener("message", listener);
      ownedWorkers.push({ worker, listener, matches });
      return worker;
    },
  });
  globalThis.Blob = blobProxy; URL.createObjectURL = create; globalThis.Worker = workerProxy;
  function snapshot() { return { enabled, disposed, totals: { ...totals }, traces: traces.map(row => structuredClone(row)),
    internalWasmCopiesAndAllocationsUnobserved: true, timingNotProductionSpeedEvidence: true }; }
  return {
    snapshot,
    dispose() {
      if (disposed || globalThis.Blob !== blobProxy || globalThis.Worker !== workerProxy || URL.createObjectURL !== create) throw new Error("The bridge observer alias ownership changed");
      for (const { worker, listener } of ownedWorkers) worker.removeEventListener("message", listener);
      globalThis.Blob = nativeBlob; globalThis.Worker = nativeWorker; URL.createObjectURL = nativeCreate;
      disposed = true;
      return { ...snapshot(), originalAliasesRestored: globalThis.Blob === nativeBlob && globalThis.Worker === nativeWorker && URL.createObjectURL === nativeCreate };
    },
  };
}
