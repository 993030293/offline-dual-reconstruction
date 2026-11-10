import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { createCameraProjection } from './calibrated_camera.js';

const ui = id => document.getElementById(id);
const state = { ready:false, errors:[], frames:0, navigationMode:'orbit', sceneMode:'scene', playing:false, currentView:0 };
window.surfaceViewer = state;
const fail = error => { state.errors.push(String(error?.stack ?? error)); ui('error').style.display='block'; ui('error').textContent=state.errors.join('\n'); };
addEventListener('error', event => fail(event.error ?? event.message));
addEventListener('unhandledrejection', event => fail(event.reason));
async function readJSON(path) { const response = await fetch(path); if(!response.ok) throw new Error(path+': '+response.status); return response.json(); }
try {
  const [meta, views, response] = await Promise.all([readJSON('./manifest.json'),readJSON('./camera_path.json'),fetch('./points.bin')]);
  if(!response.ok) throw new Error('Point asset HTTP '+response.status);
  const data = new Float32Array(await response.arrayBuffer());
  if(data.length!==meta.points*6) throw new Error('Complete point asset has a wrong denominator');
  const positions = new Float32Array(meta.points*3), colors = new Float32Array(meta.points*3), focused=[];
  const focus = new THREE.Vector3(...meta.focus), up = new THREE.Vector3(0,0,1);
  for(let i=0;i<meta.points;i++) {
    for(let axis=0;axis<3;axis++) {
      const value=data[i*6+axis]; if(!Number.isFinite(value)) throw new Error('Nonfinite point coordinate');
      positions[i*3+axis]=value; colors[i*3+axis]=data[i*6+axis+3];
    }
    if(new THREE.Vector3(positions[i*3],positions[i*3+1],positions[i*3+2]).distanceTo(focus)<=meta.focusRadius) focused.push(i);
  }
  if(!focused.length) throw new Error('The fused point cloud has no support around the common capture focus');
  const scene=new THREE.Scene(); scene.background=new THREE.Color('#080b0f');
  const camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,.02,200); camera.up.copy(up);
  const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,1.5)); renderer.setSize(innerWidth,innerHeight);
  const canvas=renderer.domElement; canvas.tabIndex=0; canvas.setAttribute('aria-label','三维表面点云操作区域'); ui('viewport').appendChild(canvas);
  const controls=new OrbitControls(camera,canvas); controls.target.copy(focus); controls.enableDamping=true; controls.screenSpacePanning=true; controls.maxDistance=Infinity;
  const geometry=new THREE.BufferGeometry(); geometry.setAttribute('position',new THREE.BufferAttribute(positions,3)); geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));
  const material=new THREE.PointsMaterial({size:.008,vertexColors:true,sizeAttenuation:true}); scene.add(new THREE.Points(geometry,material));
  const projection=createCameraProjection(camera,renderer,state);
  const poses=views.map(view=>{ const r=view.rotation_c2w; const m=new THREE.Matrix4().set(r[0][0],-r[0][1],-r[0][2],0,r[1][0],-r[1][1],-r[1][2],0,r[2][0],-r[2][1],-r[2][2],0,0,0,0,1); return {position:new THREE.Vector3(...view.position),quaternion:new THREE.Quaternion().setFromRotationMatrix(m)}; });
  let last=performance.now(), yaw=0, pitch=0, drag=null, pathTime=0;
  const keys=new Set(), forward=new THREE.Vector3(), right=new THREE.Vector3();
  function stop() { state.playing=false; ui('play').textContent='播放采集路径'; }
  function angles() { camera.getWorldDirection(forward); yaw=Math.atan2(forward.y,forward.x); pitch=Math.asin(THREE.MathUtils.clamp(forward.z,-1,1)); }
  function capture(index) {
    state.currentView=index; camera.position.copy(poses[index].position); camera.quaternion.copy(poses[index].quaternion); camera.up.copy(up); projection.capture(views[index]);
    controls.target.copy(camera.position.clone().add(camera.getWorldDirection(forward).multiplyScalar(Math.max(.1,camera.position.distanceTo(focus)))));
    ui('path').value=index; ui('view-label').textContent=(index+1)+' / '+views.length; angles();
  }
  ui('path').max=views.length-1;
  capture(meta.defaultView);
  function overview() {
    stop();projection.free();camera.up.copy(up);
    const offset=camera.position.clone().sub(focus).normalize();
    if(offset.lengthSq()<.5)offset.set(1,-1,.5).normalize();
    const vertical=THREE.MathUtils.degToRad(camera.fov)/2;
    const horizontal=Math.atan(Math.tan(vertical)*camera.aspect);
    const distance=meta.focusRadius/Math.sin(Math.min(vertical,horizontal))*1.12;
    camera.position.copy(focus).addScaledVector(offset,distance);camera.lookAt(focus);controls.target.copy(focus);angles();
  }
  ui('overview').addEventListener('click',overview);
  if(meta.defaultNavigation==='overview')overview();
  ui('scene-mode').addEventListener('change',event=>{state.sceneMode=event.target.value;geometry.setIndex(state.sceneMode==='focus'?focused:null);});
  ui('navigation-mode').addEventListener('change',event=>{stop();state.navigationMode=event.target.value;controls.enabled=state.navigationMode==='orbit';projection.free();keys.clear();angles();});
  ui('point-size').addEventListener('input',event=>material.size=Number(event.target.value)/1000);
  ui('top').addEventListener('click',()=>{stop();projection.free();camera.up.set(0,1,0);camera.position.copy(focus).add(new THREE.Vector3(0,0,meta.focusRadius*3));camera.lookAt(focus);controls.target.copy(focus);angles();});
  ui('reset').addEventListener('click',()=>{stop();capture(meta.defaultView);});
  ui('path').addEventListener('input',event=>{stop();capture(Number(event.target.value));});
  ui('play').addEventListener('click',()=>{state.playing=!state.playing;pathTime=performance.now();ui('play').textContent=state.playing?'暂停采集路径':'播放采集路径';});
  canvas.addEventListener('pointerdown',event=>{stop();projection.free();canvas.focus();if(state.navigationMode==='fly'){drag={x:event.clientX,y:event.clientY};canvas.setPointerCapture(event.pointerId);}});
  canvas.addEventListener('pointermove',event=>{if(drag){yaw-=(event.clientX-drag.x)*.004;pitch-= (event.clientY-drag.y)*.004;pitch=THREE.MathUtils.clamp(pitch,-Math.PI/2+.005,Math.PI/2-.005);drag={x:event.clientX,y:event.clientY};camera.up.copy(up);forward.set(Math.cos(pitch)*Math.cos(yaw),Math.cos(pitch)*Math.sin(yaw),Math.sin(pitch));camera.lookAt(camera.position.clone().add(forward));}});
  canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('contextmenu',event=>event.preventDefault());
  canvas.addEventListener('wheel',()=>{stop();projection.free();},{passive:true});
  canvas.addEventListener('keydown',event=>{if(['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','ShiftLeft','ShiftRight'].includes(event.code)){keys.add(event.code);event.preventDefault();}});
  canvas.addEventListener('keyup',event=>keys.delete(event.code));canvas.addEventListener('blur',()=>keys.clear());
  addEventListener('resize',()=>projection.resize());
  state.getCameraPosition=()=>camera.position.toArray();state.getCameraDirection=()=>camera.getWorldDirection(new THREE.Vector3()).toArray();state.getActiveKeys=()=>[...keys];state.points=meta.points;state.focusPoints=focused.length;state.renderer=renderer;
  state.ready=true;ui('status').textContent=meta.points.toLocaleString()+' 点 · 共用米制坐标';
  function frame(time) {
    const dt=Math.min((time-last)/1000,.08);last=time;
    if(state.playing) capture((Math.floor((time-pathTime)/120)+meta.defaultView)%views.length);
    else if(state.navigationMode==='orbit') controls.update();
    else {
      const speed=(keys.has('ShiftLeft')||keys.has('ShiftRight')?3:1)*.7*dt;
      camera.getWorldDirection(forward);right.crossVectors(forward,up).normalize();
      if(keys.has('KeyW'))camera.position.addScaledVector(forward,speed);if(keys.has('KeyS'))camera.position.addScaledVector(forward,-speed);
      if(keys.has('KeyD'))camera.position.addScaledVector(right,speed);if(keys.has('KeyA'))camera.position.addScaledVector(right,-speed);
      if(keys.has('KeyE'))camera.position.addScaledVector(up,speed);if(keys.has('KeyQ'))camera.position.addScaledVector(up,-speed);
    }
    renderer.render(scene,camera);state.frames++;requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
} catch(error) { fail(error); }
