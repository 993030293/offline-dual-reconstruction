import * as THREE from "three";
import { OrbitControls } from "../../../shared-build/vendor/OrbitControls.js";
import { createOrbitConvergenceDag } from "./orbit_convergence_dag.js";

// Compare the real original controller's complete CPU sequence across all input poses.
export function runOrbitConvergenceGeneric({ views, publicViews, focus }) {
  const encode = values => JSON.stringify(values.map(value => Object.is(value, -0) ? "-0" : value));
  const pose = (controls, camera) => encode([...camera.position.toArray(), ...camera.quaternion.toArray(), ...controls.target.toArray()]);
  const complete = (controls, camera) => encode([...camera.position.toArray(), ...camera.quaternion.toArray(), ...controls.target.toArray(),
    ...camera.matrixWorld.toArray(), ...controls._lastPosition.toArray(), ...controls._lastQuaternion.toArray(),
    ...controls._lastTargetPosition.toArray(), controls._spherical.radius, controls._spherical.phi, controls._spherical.theta,
    controls._sphericalDelta.radius, controls._sphericalDelta.phi, controls._sphericalDelta.theta,
    ...controls._panOffset.toArray(), controls._scale]);
  function setup(fixture) {
    const camera = new THREE.PerspectiveCamera(50, 1.6, .02, 200); camera.up.fromArray(fixture.up ?? [0, 0, 1]);
    const canvas = document.createElement("canvas"), controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true; controls.dampingFactor = .12; controls.minDistance = .02; controls.maxDistance = Infinity;
    controls.screenSpacePanning = true; camera.position.fromArray(fixture.position);
    if (fixture.rotation_c2w) {
      const r = fixture.rotation_c2w, matrix = new THREE.Matrix4().set(r[0][0], -r[0][1], -r[0][2], 0, r[1][0], -r[1][1], -r[1][2], 0, r[2][0], -r[2][1], -r[2][2], 0, 0, 0, 0, 1);
      camera.quaternion.setFromRotationMatrix(matrix);
      const forward = camera.getWorldDirection(new THREE.Vector3());
      controls.target.copy(camera.position).addScaledVector(forward, Math.max(.3, camera.position.distanceTo(new THREE.Vector3(...fixture.focus))));
    } else {
      controls.target.fromArray(fixture.target); camera.lookAt(controls.target);
    }
    return { camera, controls };
  }
  function oracle(instance) {
    const seen = new Set(); let previous = pose(instance.controls, instance.camera), changed = false;
    seen.add(previous);
    for (let steps = 1; steps <= 4096; ++steps) {
      changed = instance.controls.update() || changed; instance.camera.updateMatrixWorld();
      const next = pose(instance.controls, instance.camera);
      if (next === previous) return { fixed: true, steps, changed, complete: complete(instance.controls, instance.camera) };
      if (seen.has(next)) return { fixed: false, steps, route: "cycle" };
      seen.add(next); previous = next;
    }
    return { fixed: false, steps: 4096, route: "bound" };
  }
  const publicFocus = publicViews.reduce((sum, view) => sum.map((value, i) => value + view.position[i] / publicViews.length), [0, 0, 0]);
  const generated = Array.from({ length: 128 }, (_, i) => ({ position: [Math.sin(i * .73) * 13, Math.cos(i * .31) * 7, 1 + Math.sin(i * .17) * 4],
    target: [Math.cos(i * .13) * 2, Math.sin(i * .37), i % 7 * .13], up: i % 2 ? [0, 1, 0] : [0, 0, 1] }));
  const populations = [{ name: "all_original1016", fixtures: views.map(view => ({ ...view, focus })) },
    { name: "all_public360", fixtures: publicViews.map(view => ({ ...view, focus: publicFocus })) }, { name: "generic128", fixtures: generated }];
  const summaries = [], failures = [], routes = { converged: 0, cycle: 0, bound: 0 };
  for (const population of populations) {
    let exact = 0, converged = 0, maxSteps = 0;
    for (const [i, fixture] of population.fixtures.entries()) {
      const native = setup(fixture), candidate = setup(fixture), single = setup(fixture);
      try {
        const expected = oracle(native), state = { navigationMode: "orbit", playing: false };
        single.controls.update(); single.camera.updateMatrixWorld();
        const helper = createOrbitConvergenceDag({ ...candidate, state }); helper.markDiscrete(); helper.update();
        if (!expected.fixed) candidate.camera.updateMatrixWorld();
        const trace = helper.snapshot(), target = expected.fixed ? expected.complete : complete(single.controls, single.camera);
        const equal = complete(candidate.controls, candidate.camera) === target && (expected.fixed ? trace.converged === 1 : trace.cycleUnsupported + trace.boundUnsupported === 1);
        exact += Number(equal); converged += Number(expected.fixed); maxSteps = Math.max(maxSteps, expected.steps);
        routes[expected.fixed ? "converged" : expected.route]++;
        if (!equal) failures.push({ population: population.name, i, expected, trace, actual: complete(candidate.controls, candidate.camera) });
        const restored = helper.dispose(); if (!restored.nativeUpdateUnchanged || restored.pending) throw new Error("The original generic controller alias changed");
      } finally { native.controls.dispose(); candidate.controls.dispose(); single.controls.dispose(); }
    }
    summaries.push({ population: population.name, denominator: population.fixtures.length, exact, converged, maxSteps });
  }
  const unsupported = ["inertia", "pan", "scale", "cursor", "state", "autoRotate", "fly", "playing", "parent", "matrixAutoUpdate", "matrixWorldAutoUpdate"];
  let unsupportedExact = 0;
  for (const type of unsupported) {
    const native = setup(generated[0]), candidate = setup(generated[0]);
    const nativeState = { navigationMode: "orbit", playing: false }, state = { ...nativeState };
    const modify = (instance, value) => {
      if (type === "inertia") instance.controls._sphericalDelta.theta = .1;
      if (type === "pan") instance.controls._panOffset.set(.1, .2, .3);
      if (type === "scale") instance.controls._scale = .9;
      if (type === "cursor") instance.controls._performCursorZoom = true;
      if (type === "state") instance.controls.state = 0;
      if (type === "autoRotate") instance.controls.autoRotate = true;
      if (type === "fly") value.navigationMode = "fly";
      if (type === "playing") value.playing = true;
      if (type === "parent") new THREE.Object3D().add(instance.camera);
      if (type === "matrixAutoUpdate") instance.camera.matrixAutoUpdate = false;
      if (type === "matrixWorldAutoUpdate") instance.camera.matrixWorldAutoUpdate = false;
    };
    try {
      modify(native, nativeState); modify(candidate, state);
      const helper = createOrbitConvergenceDag({ ...candidate, state }); helper.markDiscrete();
      const before = complete(candidate.controls, candidate.camera);
      const markedInputUnchanged = complete(candidate.controls, candidate.camera) === before;
      native.controls.update(); helper.update();
      const trace = helper.snapshot();
      const equal = markedInputUnchanged && complete(native.controls, native.camera) === complete(candidate.controls, candidate.camera) && trace.converged === 0 && trace.shadowUpdates === 0;
      unsupportedExact += Number(equal); if (!equal) failures.push({ unsupported: type, trace }); helper.dispose();
    } finally { native.controls.dispose(); candidate.controls.dispose(); }
  }
  return { status: failures.length ? "failed_full_pose_generic" : "full_original_public_and_generic_pose_contract_passed0",
    checks: { all_full_pose_populations_exact: summaries.every(value => value.exact === value.denominator),
      original1016_and_public360_denominators_preserved: views.length === 1016 && publicViews.length === 360,
      all11_unsupported_original_routes_exact: unsupportedExact === unsupported.length, failures_empty: failures.length === 0 },
    summaries, routes, unsupportedExact, failures };
}
