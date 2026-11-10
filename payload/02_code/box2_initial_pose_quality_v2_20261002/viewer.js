import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

const ui = id => document.getElementById(id);
const state = { ready: false, errors: [], frames: 0, currentView: 0, playing: false, sceneMode: "scene", navigationMode: "orbit", flightSpeed: 0.7 };
window.box2Viewer = state;
const worldUp = new THREE.Vector3(0, 0, 1);
function showError(error) {
  const message = String(error?.stack ?? error);
  state.errors.push(message);
  ui("status").textContent = "加载失败";
  ui("error").style.display = "block";
  ui("error").textContent = message;
}
window.addEventListener("error", event => showError(event.error ?? event.message));
window.addEventListener("unhandledrejection", event => showError(event.reason));
async function loadJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(path + ": HTTP " + response.status);
  return response.json();
}

try {
  const [metadata, views] = await Promise.all([loadJSON("./display_manifest.json"), loadJSON("./camera_path.json")]);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#080b0f");
  const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.02, 200);
  camera.up.copy(worldUp);
  const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.setSize(innerWidth, innerHeight);
  const canvas = renderer.domElement;
  canvas.tabIndex = 0;
  canvas.setAttribute("aria-label", "三维场景操作区域");
  ui("viewport").appendChild(canvas);
  scene.add(new SparkRenderer({ renderer }));
  const focus = new THREE.Vector3(...metadata.focus);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.minDistance = 0.02;
  controls.maxDistance = Infinity;
  controls.screenSpacePanning = true;
  const radius = Math.max(1, new THREE.Vector3(...metadata.roi_maximum).distanceTo(new THREE.Vector3(...metadata.roi_minimum)));
  const referenceFront = new THREE.Vector3(...views[metadata.default_view].position).sub(focus);
  referenceFront.z = 0;
  referenceFront.normalize();
  const referenceRight = referenceFront.clone().cross(worldUp).normalize();
  const poses = views.map(view => {
    const r = view.rotation_c2w;
    // Convert COLMAP optical axes to Three.js camera axes without changing world coordinates.
    const matrix = new THREE.Matrix4().set(r[0][0], -r[0][1], -r[0][2], 0, r[1][0], -r[1][1], -r[1][2], 0, r[2][0], -r[2][1], -r[2][2], 0, 0, 0, 0, 1);
    return { position: new THREE.Vector3(...view.position), quaternion: new THREE.Quaternion().setFromRotationMatrix(matrix), fov: THREE.MathUtils.radToDeg(2 * Math.atan(view.height / (2 * view.fy))) };
  });
  const keys = new Set();
  let yaw = 0, pitch = 0, drag = null, playbackStart = 0, lastTime = 0;
  const forward = new THREE.Vector3(), right = new THREE.Vector3(), movement = new THREE.Vector3();
  function stopPlayback() {
    state.playing = false;
    controls.enabled = state.navigationMode === "orbit";
    ui("play").textContent = "播放采集路径";
    ui("play").setAttribute("aria-pressed", "false");
  }
  function syncFlightAngles() {
    camera.getWorldDirection(forward);
    yaw = Math.atan2(forward.y, forward.x);
    pitch = Math.asin(THREE.MathUtils.clamp(forward.z, -1, 1));
  }
  function orientFlight() {
    pitch = THREE.MathUtils.clamp(pitch, -Math.PI / 2 + 0.005, Math.PI / 2 - 0.005);
    forward.set(Math.cos(pitch) * Math.cos(yaw), Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch));
    camera.up.copy(worldUp);
    camera.lookAt(camera.position.clone().add(forward));
  }
  function freeView() {
    stopPlayback();
    ui("view-label").textContent = state.navigationMode === "fly" ? "自由飞行" : "自由视角";
  }
  function setMode(mode) {
    stopPlayback();
    keys.clear();
    drag = null;
    state.navigationMode = mode;
    ui("navigation-mode").value = mode;
    controls.enabled = mode === "orbit";
    if (mode === "fly") {
      syncFlightAngles();
      orientFlight();
    } else {
      camera.getWorldDirection(forward);
      controls.target.copy(camera.position).addScaledVector(forward, Math.max(0.3, camera.position.distanceTo(focus)));
      controls.update();
    }
    ui("help").textContent = mode === "fly" ? "点击场景 · WASD 移动 · Q/E 升降 · Shift 加速 · 左键转向 · 右键平移" : "左键旋转 · 右键平移 · 滚轮缩放 · 可切换自由飞行";
  }
  function applyView(index, fraction = 0) {
    const a = poses[index], b = poses[Math.min(index + 1, poses.length - 1)];
    camera.position.copy(a.position).lerp(b.position, fraction);
    camera.quaternion.copy(a.quaternion).slerp(b.quaternion, fraction);
    camera.fov = a.fov;
    camera.updateProjectionMatrix();
    camera.getWorldDirection(forward);
    controls.target.copy(camera.position).addScaledVector(forward, Math.max(0.3, camera.position.distanceTo(focus)));
    syncFlightAngles();
    state.currentView = index;
    ui("path-slider").value = String(index);
    ui("view-label").textContent = (index + 1) + " / " + views.length;
  }
  function preset(name) {
    stopPlayback();
    const offset = new THREE.Vector3();
    const distance = radius * 1.15;
    if (name === "front") offset.copy(referenceFront).multiplyScalar(distance);
    if (name === "back") offset.copy(referenceFront).multiplyScalar(-distance);
    if (name === "right") offset.copy(referenceRight).multiplyScalar(distance);
    if (name === "left") offset.copy(referenceRight).multiplyScalar(-distance);
    offset.z = radius * 0.18;
    if (name === "drone") offset.copy(referenceFront).multiplyScalar(radius * 0.7).addScaledVector(worldUp, radius * 1.1);
    if (name === "top") offset.copy(referenceFront).multiplyScalar(radius * 0.006).addScaledVector(worldUp, radius * 1.25);
    camera.position.copy(focus).add(offset);
    camera.up.copy(worldUp);
    camera.lookAt(focus);
    camera.fov = 50;
    camera.updateProjectionMatrix();
    controls.target.copy(focus);
    syncFlightAngles();
    if (name === "top" || name === "drone") setMode("fly");
    else if (state.navigationMode === "orbit") controls.update();
    ui("view-label").textContent = name === "top" ? "无人机俯视" : name === "drone" ? "无人机斜俯视" : "自由视角";
    canvas.focus({ preventScroll: true });
  }
  controls.addEventListener("start", freeView);
  ui("navigation-mode").addEventListener("change", event => { setMode(event.target.value); canvas.focus({ preventScroll: true }); });
  document.querySelectorAll("[data-preset]").forEach(button => button.addEventListener("click", () => preset(button.dataset.preset)));
  ui("flight-speed").addEventListener("input", event => {
    state.flightSpeed = Number(event.target.value);
    ui("speed-value").textContent = state.flightSpeed.toFixed(2) + " m/s";
  });
  const navigationKeys = new Set(["KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE", "ShiftLeft", "ShiftRight"]);
  addEventListener("keydown", event => {
    if (state.navigationMode !== "fly" || !navigationKeys.has(event.code) || /^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(event.target.tagName)) return;
    event.preventDefault();
    keys.add(event.code);
    freeView();
  });
  addEventListener("keyup", event => keys.delete(event.code));
  addEventListener("blur", () => { keys.clear(); drag = null; });
  document.addEventListener("visibilitychange", () => { keys.clear(); drag = null; });
  canvas.addEventListener("contextmenu", event => event.preventDefault());
  canvas.addEventListener("pointerdown", event => {
    canvas.focus({ preventScroll: true });
    if (state.navigationMode !== "fly") return;
    freeView();
    drag = { x: event.clientX, y: event.clientY, button: event.button };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", event => {
    if (state.navigationMode !== "fly" || !drag) return;
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    drag.x = event.clientX; drag.y = event.clientY;
    if (drag.button === 0) { yaw -= dx * 0.004; pitch -= dy * 0.004; orientFlight(); }
    if (drag.button === 2) {
      right.set(1, 0, 0).applyQuaternion(camera.quaternion);
      const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
      const factor = Math.max(0.001, camera.position.distanceTo(focus) / innerHeight);
      camera.position.addScaledVector(right, -dx * factor).addScaledVector(up, dy * factor);
    }
  });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("pointercancel", () => { drag = null; });
  canvas.addEventListener("wheel", event => {
    if (state.navigationMode !== "fly") return;
    event.preventDefault();
    freeView();
    camera.getWorldDirection(forward);
    camera.position.addScaledVector(forward, -event.deltaY * state.flightSpeed * 0.002);
  }, { passive: false });
  ui("path-slider").max = String(views.length - 1);
  ui("path-slider").addEventListener("input", event => { keys.clear(); stopPlayback(); applyView(Number(event.target.value)); });
  ui("previous").addEventListener("click", () => { keys.clear(); stopPlayback(); applyView(Math.max(0, state.currentView - 15)); });
  ui("next").addEventListener("click", () => { keys.clear(); stopPlayback(); applyView(Math.min(views.length - 1, state.currentView + 15)); });
  ui("reset").addEventListener("click", () => { keys.clear(); stopPlayback(); applyView(metadata.default_view); });
  ui("play").addEventListener("click", () => {
    if (state.playing) return stopPlayback();
    keys.clear(); drag = null;
    state.playing = true; controls.enabled = false; playbackStart = performance.now();
    ui("play").textContent = "暂停播放"; ui("play").setAttribute("aria-pressed", "true");
  });
  ui("fullscreen").addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen().catch(showError);
  });
  ui("collapse").addEventListener("click", () => {
    const collapsed = ui("settings").classList.toggle("collapsed");
    ui("collapse").textContent = collapsed ? "+" : "−";
    ui("collapse").setAttribute("aria-expanded", String(!collapsed));
    ui("collapse").setAttribute("aria-label", collapsed ? "展开控制面板" : "收起控制面板");
  });
  const grid = new THREE.GridHelper(20, 40, "#658898", "#273c49");
  grid.rotation.x = Math.PI / 2;
  grid.position.set(focus.x, focus.y, metadata.roi_minimum[2] - 0.03);
  grid.visible = false;
  scene.add(grid);
  ui("grid").addEventListener("click", () => { grid.visible = !grid.visible; ui("grid").setAttribute("aria-pressed", String(grid.visible)); ui("grid").textContent = grid.visible ? "隐藏参考网格" : "显示参考网格"; });
  let light = false;
  ui("background").addEventListener("click", () => { light = !light; scene.background.set(light ? "#d9e2e4" : "#080b0f"); ui("background").textContent = light ? "深色背景" : "浅色背景"; });
  const fullModel = new SplatMesh({ url: "./scene.ply", lod: false });
  const boxModel = new SplatMesh({ url: "./box.ply", lod: false });
  scene.add(fullModel, boxModel);
  boxModel.visible = false;
  ui("scene-mode").addEventListener("change", event => {
    state.sceneMode = event.target.value;
    fullModel.visible = event.target.value === "scene"; boxModel.visible = event.target.value === "box";
  });
  ui("case-select").addEventListener("change", event => { location.href = "../" + event.target.value + "/index.html"; });
  applyView(metadata.default_view);
  addEventListener("resize", () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
  state.getCameraPosition = () => camera.position.toArray();
  state.getCameraDirection = () => camera.getWorldDirection(new THREE.Vector3()).toArray();
  state.getActiveKeys = () => [...keys];
  state.setView = index => { stopPlayback(); keys.clear(); applyView(index); };
  state.renderer = renderer;
  renderer.setAnimationLoop(time => {
    const dt = Math.min(0.05, Math.max(0, (time - lastTime) / 1000));
    lastTime = time;
    if (state.playing) {
      const phase = ((time - playbackStart) / 16000) % 2;
      const progress = (phase <= 1 ? phase : 2 - phase) * (views.length - 1);
      applyView(Math.floor(progress), progress % 1);
    } else if (state.navigationMode === "orbit") {
      controls.update();
    } else if (keys.size) {
      camera.getWorldDirection(forward);
      right.set(1, 0, 0).applyQuaternion(camera.quaternion);
      movement.set(0, 0, 0).addScaledVector(forward, Number(keys.has("KeyW")) - Number(keys.has("KeyS")))
        .addScaledVector(right, Number(keys.has("KeyD")) - Number(keys.has("KeyA")))
        .addScaledVector(worldUp, Number(keys.has("KeyE")) - Number(keys.has("KeyQ")));
      const boost = keys.has("ShiftLeft") || keys.has("ShiftRight") ? 4 : 1;
      if (movement.lengthSq()) camera.position.addScaledVector(movement.normalize(), dt * state.flightSpeed * boost);
    }
    renderer.render(scene, camera); state.frames++;
  });
  await Promise.all([fullModel.initialized, boxModel.initialized]);
  state.ready = true; state.fullGaussianCount = fullModel.numSplats; state.boxGaussianCount = boxModel.numSplats;
  ui("status").textContent = "模型已加载 · 可自由浏览或切换飞行";
} catch (error) { showError(error); }
