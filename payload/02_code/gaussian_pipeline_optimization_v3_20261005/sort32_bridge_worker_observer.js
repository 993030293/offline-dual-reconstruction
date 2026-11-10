// Count actual WASM imports within the original synchronous sort RPC.
// This block is inserted into a private copy of the MIT-licensed original worker.
{
  let activeBridgeTrace = null;
  let serial = 0;
  const nativeGetImports = __wbg_get_imports;
  const nativeHandler = rpcHandlers.sortSplats32;
  const bulkInto = new Map([
    ["__wbg_prototypesetcall_3e05eb9545565046", 1],
    ["__wbg_prototypesetcall_64c287a27cc24d27", 2],
    ["__wbg_prototypesetcall_e42275e601e14eeb", 4],
  ]);
  const bulkOut = new Map([
    ["__wbg_set_448126769bf7c181", 4],
    ["__wbg_set_a98c8da6557e63de", 4],
  ]);
  const allocations = new Map([
    ["__wbg_new_with_length_41a22191b9bdfd66", 4],
    ["__wbg_new_with_length_26bffbe236bf73f9", 4],
  ]);
  const scalarReads = new Set(["__wbg_get_4848e350b40afc16", "__wbg_get_unchecked_7d7babe32e9e6a54"]);
  const scalarWrites = new Set(["__wbg_set_6c60b2e8ad0e9383", "__wbg_set_index_338a66e40fc45dee"]);
  function describe(value, trace) {
    if (!ArrayBuffer.isView(value)) return { type: typeof value, arrayView: false };
    const shares = input => value.buffer === input.buffer && value.byteOffset >= input.byteOffset && value.byteOffset + value.byteLength <= input.byteOffset + input.byteLength;
    return { type: Object.prototype.toString.call(value), arrayView: true, length: value.length,
      byteLength: value.byteLength, byteOffset: value.byteOffset,
      originalReadbackView: value === trace.readback, originalOrderingView: value === trace.ordering,
      sharesReadbackBuffer: shares(trace.readback), sharesOrderingBuffer: shares(trace.ordering),
      wasmMemoryBuffer: value.buffer === wasm.memory.buffer };
  }
  __wbg_get_imports = function (...args) {
    const imports = nativeGetImports.apply(this, args);
    const table = imports["./spark_rs_bg.js"];
    if (!table || typeof table !== "object") throw new Error("The original WASM import namespace changed");
    for (const [name, native] of Object.entries(table)) {
      if (typeof native !== "function") throw new Error("The original WASM import callable changed");
      table[name] = function (...values) {
        const trace = activeBridgeTrace;
        if (!trace) return native.apply(this, values);
        const row = trace.imports[name] ??= { calls: 0 };
        row.calls++; trace.importCalls++;
        if (trace.importCalls > 200000000) throw new Error("The registered bridge call diagnostic bound was exceeded");
        let copy = null;
        if (bulkInto.has(name)) {
          const width = bulkInto.get(name), input = values[2];
          if (!ArrayBuffer.isView(input)) throw new Error("The original typed-array copy source changed");
          copy = { direction: "JS_to_WASM", bytes: input.byteLength, destinationBytes: values[1] * width,
            destinationPointer: values[0], source: describe(input, trace) };
        } else if (bulkOut.has(name)) {
          const width = bulkOut.get(name);
          copy = { direction: "WASM_to_JS", bytes: values[2] * width,
            sourcePointer: values[1], destination: describe(values[0], trace) };
        } else if (name === "__wbg_set_81b4174352e6a095") {
          if (!ArrayBuffer.isView(values[0]) || !ArrayBuffer.isView(values[1])) throw new Error("The generic typed-array set interface changed");
          const source = describe(values[1], trace), destination = describe(values[0], trace);
          copy = { direction: source.wasmMemoryBuffer ? "WASM_to_JS" : destination.wasmMemoryBuffer ? "JS_to_WASM" : "JS_to_JS",
            bytes: values[1].byteLength, source, destination, destinationElementOffset: values[2] >>> 0 };
        }
        if (copy) {
          if (trace.copies.length >= 128) throw new Error("The registered per-sort copy event bound was exceeded");
          const at = performance.now();
          try { return native.apply(this, values); }
          finally {
            copy.nativeImportWallMs = performance.now() - at;
            trace.copies.push(copy); trace.observedImportCopyBytes += copy.bytes;
          }
        }
        if (scalarReads.has(name) || scalarWrites.has(name)) {
          if (row.firstArray === undefined) row.firstArray = describe(values[0], trace);
          const index = values[1] >>> 0;
          row.minIndex = row.minIndex === undefined ? index : Math.min(row.minIndex, index);
          row.maxIndex = row.maxIndex === undefined ? index : Math.max(row.maxIndex, index);
          if (scalarReads.has(name)) trace.scalarReadCalls++; else trace.scalarWriteCalls++;
        }
        const result = native.apply(this, values);
        if (allocations.has(name)) {
          const bytes = (values[0] >>> 0) * allocations.get(name);
          row.requestedJSBackingBytes = (row.requestedJSBackingBytes ?? 0) + bytes;
          trace.observedJSBackingRequestsBytes += bytes;
          if (row.firstResult === undefined) row.firstResult = describe(result, trace);
        } else if (name.startsWith("__wbg_subarray_") && row.firstSource === undefined) {
          row.firstSource = describe(values[0], trace);
          row.firstStart = values[1] >>> 0; row.firstEnd = values[2] >>> 0;
          row.firstResult = describe(result, trace);
        } else if (name.startsWith("__wbindgen_cast_") && row.firstResult === undefined) {
          row.firstResult = describe(result, trace);
        }
        return result;
      };
    }
    return imports;
  };
  rpcHandlers.sortSplats32 = function (args, options) {
    if (activeBridgeTrace) throw new Error("The original synchronous sort RPC became reentrant");
    if (++serial > 4096) throw new Error("The registered sort diagnostic event bound was exceeded");
    const trace = { serial, numSplats: args.numSplats, readback: args.readback, ordering: args.ordering,
      imports: {}, importCalls: 0, copies: [], observedImportCopyBytes: 0,
      observedJSBackingRequestsBytes: 0, scalarReadCalls: 0, scalarWriteCalls: 0 };
    const readbackInput = describe(args.readback, trace), orderingInput = describe(args.ordering, trace);
    activeBridgeTrace = trace;
    const started = performance.now();
    let result, primaryError = null;
    try { result = nativeHandler.call(this, args, options); return result; }
    catch (error) { primaryError = String(error); throw error; }
    finally {
      activeBridgeTrace = null;
      const { readback, ordering, ...metadata } = trace;
      options.sendStatus({ sort32_bridge_008: { ...metadata, readbackInput, orderingInput,
        originalHandlerWallWithObserverMs: performance.now() - started, primaryError,
        activeSplats: result?.activeSplats, resultReadbackSameView: result?.readback === readback,
        resultOrderingSameView: result?.ordering === ordering,
        internalWasmCopiesAndAllocationsUnobserved: true } });
    }
  };
}
