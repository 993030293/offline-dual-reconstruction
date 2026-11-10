import * as THREE from "three";
import { OrbitControls } from "../shared-build/vendor/OrbitControls.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { loadPackedAsset } from "../box2-optimized/cases/packed/packed_splat_asset.js";
import { resolveRenderFilter } from "../box2-optimized/cases/packed/render_filter_contract.js";
import { createBehindCameraCuller, packedCenterBounds, installDrawCounter } from "./behind_camera_draw.js";
import { createOrbitConvergenceDag } from "./orbit_convergence_dag_v2.js";
import { installOrderingScratchNative } from "./ordering_scratch_native_v2.js";

const output = document.getElementById("result");
const nextFrame = () => new Promise(resolve => requestAnimationFrame(resolve));
const readJSON = async path => { const reply = await fetch(path); if (!reply.ok) throw new Error(path + ": " + reply.status); return reply.json(); };
const equal = (first, second) => first.constructor === second.constructor && first.length === second.length && first.every((value, index) => value === second[index]);
function applyPose(camera, view) {
  const r = view.rotation_c2w;
  const matrix = new THREE.Matrix4().set(r[0][0], -r[0][1], -r[0][2], 0,
    r[1][0], -r[1][1], -r[1][2], 0, r[2][0], -r[2][1], -r[2][2], 0, 0, 0, 0, 1);
  camera.position.fromArray(view.position); camera.quaternion.setFromRotationMatrix(matrix); camera.updateMatrixWorld(true);
}
function signature(camera, controls) {
  camera.updateMatrixWorld(true);
  return JSON.stringify([...camera.position.toArray(), ...camera.quaternion.toArray(), ...controls.target.toArray(),
    ...camera.matrixWorld.elements, ...camera.projectionMatrix.elements].map(value => Object.is(value, -0) ? "-0" : value));
}
function orbitCheck(views, focus, canvas) {
  const records = [];
  for (let trial = 0; trial < 3; trial++) {
    let exact = 0, supported = 0, nativeUpdates = 0, helperUpdates = 0;
    const unsupported = [], failures = [];
    for (const view of views) {
      const cameras = [0, 1, 2].map(() => { const camera = new THREE.PerspectiveCamera(50, 812 / 620, 0.02, 200);
        camera.up.set(0, 0, 1); applyPose(camera, view); return camera; });
      const controls = cameras.map(camera => { const control = new OrbitControls(camera, canvas);
        control.enableDamping = true; control.dampingFactor = 0.12; control.minDistance = 0.02;
        control.target.fromArray(focus); return control; });
      controls[2].update(); const singleSignature = signature(cameras[2], controls[2]);
      let previous = signature(cameras[0], controls[0]), fixed = false;
      const seen = new Set([previous]);
      for (let step = 0; step < 4096; step++) {
        controls[0].update(); nativeUpdates++;
        const value = signature(cameras[0], controls[0]);
        if (value === previous) { fixed = true; break; }
        if (seen.has(value)) break;
        seen.add(value); previous = value;
      }
      const helper = createOrbitConvergenceDag({ controls: controls[1], camera: cameras[1],
        state: { navigationMode: "orbit", playing: false } });
      helper.markDiscrete(); helper.update(); const trace = helper.dispose();
      helperUpdates += trace.shadowUpdates + trace.replayUpdates;
      const supportedHere = trace.converged === 1;
      if (supportedHere) supported++; else unsupported.push(view.image_name);
      const reference = supportedHere && fixed ? signature(cameras[0], controls[0]) : singleSignature;
      if (signature(cameras[1], controls[1]) === reference && trace.nativeUpdateUnchanged) exact++;
      else failures.push(view.image_name);
      for (const control of controls) control.dispose();
    }
    records.push({ trial, complete_camera_denominator: views.length, exact_final_or_native_fallback_states: exact,
      supported_discrete_views: supported, unsupported_views_retained: unsupported, failures,
      measured_native_CPU_updates: nativeUpdates, measured_helper_CPU_updates: helperUpdates,
      GPU_frame_or_speed_gain_not_measured: true });
  }
  return records;
}
async function modelCheck(entry) {
  const prefix = "../" + entry.branch;
  const [manifest, views, filter, assets] = await Promise.all([
    readJSON(prefix + "/cases/packed/display_manifest.json"), readJSON(prefix + "/cases/packed/camera_path.json"),
    readJSON(prefix + "/cases/packed/render_filter.json"), readJSON(prefix + "/ASSET_MANIFEST.json")]);
  if (assets.scene.asset_sha256 !== entry.full_asset.asset_sha256 || manifest.full_gaussian_count !== entry.full_asset.numSplats) throw new Error("The frozen full-model input changed");
  const canvas = document.getElementById("view");
  const orbit = orbitCheck(views, manifest.focus, canvas);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
  renderer.setSize(812, 620, false); renderer.setPixelRatio(1);
  const scene = new THREE.Scene(); scene.background = new THREE.Color("#080b0f");
  const camera = new THREE.PerspectiveCamera(50, 812 / 620, 0.02, 200); camera.up.set(0, 0, 1);
  const spark = new SparkRenderer({ renderer, ...resolveRenderFilter(filter, assets) }); scene.add(spark);
  const packed = await loadPackedAsset(prefix + "/cases/packed/scene.spk", assets.scene);
  const model = new SplatMesh({ splats: packed, lod: false });
  await model.initialized; scene.add(model);
  if (model.numSplats !== entry.full_asset.numSplats) throw new Error("The native mesh truncated the complete model");
  const selected = Array.from({ length: 6 }, (_, number) => views[Math.floor((number + 0.5) * views.length / 6)]);
  const scratchRows = [], behindRows = [];
  try {
    for (let trial = 0; trial < 3; trial++) {
      for (const enabled of (trial % 2 ? [true, false] : [false, true])) {
        const scratch = installOrderingScratchNative({ spark, enabled });
        for (const view of selected) {
          applyPose(camera, view);
          for (let frame = 0; frame < 12; frame++) { renderer.render(scene, camera); await nextFrame(); }
        }
        for (let frame = 0; frame < 240 && (spark.sorting || scratch.snapshot().active); frame++) { renderer.render(scene, camera); await nextFrame(); }
        const trace = scratch.dispose();
        scratchRows.push({ trial, enabled, full_model_count: model.numSplats, trace });
      }
      const bounds = packedCenterBounds(model.splats);
      if (!bounds.supported) throw new Error("The complete fresh packed-center layout is unsupported");
      const span = Math.max(1, ...bounds.maximum.map((value, axis) => value - bounds.minimum[axis]));
      camera.position.set(bounds.maximum[0] + 2 * span, (bounds.minimum[1] + bounds.maximum[1]) / 2,
        (bounds.minimum[2] + bounds.maximum[2]) / 2);
      camera.lookAt(camera.position.clone().add(new THREE.Vector3(1, 0, 0))); camera.updateMatrixWorld(true);
      for (let frame = 0; frame < 20; frame++) { renderer.render(scene, camera); await nextFrame(); }
      const counter = installDrawCounter(renderer.getContext());
      const before = counter.trace.calls;
      renderer.render(scene, camera);
      const native = new Uint8Array(812 * 620 * 4); renderer.getContext().readPixels(0, 0, 812, 620,
        renderer.getContext().RGBA, renderer.getContext().UNSIGNED_BYTE, native);
      const nativeCalls = counter.trace.calls - before;
      const culler = createBehindCameraCuller({ scene, spark, SplatMesh });
      const current = counter.trace.calls; culler.beforeFrame(camera); renderer.render(scene, camera);
      const candidate = new Uint8Array(native.length); renderer.getContext().readPixels(0, 0, 812, 620,
        renderer.getContext().RGBA, renderer.getContext().UNSIGNED_BYTE, candidate);
      behindRows.push({ trial, full_model_count: model.numSplats, complete_RGBA_bytes: native.length,
        all_pixels_literal_equal: equal(native, candidate), native_draw_calls: nativeCalls,
        candidate_draw_calls: counter.trace.calls - current, culler_trace: structuredClone(culler.trace) });
      culler.dispose(); counter.dispose();
    }
    return { branch: entry.branch, full_gaussian_count: model.numSplats, camera_count: views.length,
      orbit_rows: orbit, scratch_rows: scratchRows, behind_rows: behindRows,
      all_original_full_SH_orders_retained: packed.getNumSh() === 3 };
  } finally { scene.remove(model); model.dispose(); spark.dispose(); renderer.dispose(); }
}
document.getElementById("run").addEventListener("click", async () => {
  document.getElementById("run").disabled = true;
  const state = { status: "running", rows: [], errors: [], started_unix_ms: Date.now() };
  const publish = () => { output.textContent = JSON.stringify(state, null, 2); };
  publish();
  try {
    const contract = await readJSON("./CONTRACT.json");
    for (const entry of contract.branches) {
      state.current_branch = entry.branch; publish();
      try { state.rows.push(await modelCheck(entry)); }
      catch (error) { state.errors.push({ branch: entry.branch, error: String(error?.stack ?? error) }); }
      publish();
    }
    state.status = "completed_outcomes_recorded"; state.elapsed_ms = Date.now() - state.started_unix_ms;
  } catch (error) { state.status = "failed_fixture"; state.errors.push({ error: String(error?.stack ?? error) }); }
  publish();
});
