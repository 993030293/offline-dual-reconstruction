import assert from 'node:assert/strict';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const source = resolve(dirname(fileURLToPath(import.meta.url)), 'lazy_models.js');
const code = await readFile(source, 'utf8');
const { createLazyModels } = await import('data:text/javascript;base64,' + Buffer.from(code).toString('base64'));
const output = process.argv[2];
if (!output) throw new Error('An independent output directory is required');
await mkdir(output, { recursive: false });
const checks = [];
const check = (condition, name) => { assert.ok(condition, name); checks.push({ name, passed: true }); };
const mock = count => ({ numSplats: count, visible: false, initialized: Promise.resolve() });
let release;
const state = {}, added = [];
const loader = createLazyModels({ state, scene: { add: model => added.push(model) }, requestRender: () => {},
  load: mode => mode === 'box' ? Promise.resolve(mock(10)) : new Promise(done => { release = done; }) });
await loader.select('box');
check(state.ready && state.sceneMode === 'box' && state.modelLoadAttempts.scene === 0, 'initial_box_has_no_scene_request');
const first = loader.select('scene'), second = loader.select('scene');
await loader.select('box');
check(state.modelLoadAttempts.scene === 1, 'inflight_model_is_not_duplicated');
release(mock(100));
const raced = await Promise.all([first, second]);
check(raced.every(result => !result.applied) && state.sceneMode === 'box', 'late_scene_result_cannot_replace_latest_box');
check(added.filter(model => model.visible).length === 1 && state.boxGaussianCount === 10 && state.fullGaussianCount === 100,
  'loaded_counts_exact_single_visible_model');
await loader.select('scene');
check(state.modelLoadAttempts.scene === 1 && state.sceneMode === 'scene', 'completed_scene_reused_without_request');
check(added.filter(model => model.visible).length === 1 && added.find(model => model.numSplats === 100).visible,
  'switch_visibility_is_consistent');
const retryState = {}, retryModels = [];
let fail = true;
const retryLoader = createLazyModels({ state: retryState, scene: { add: model => retryModels.push(model) }, requestRender: () => {},
  load: async mode => { if (mode === 'scene' && fail) throw new Error('Injected scene request failure'); return mock(mode === 'box' ? 10 : 100); } });
await retryLoader.select('box');
await assert.rejects(retryLoader.select('scene'), /Injected scene request failure/);
check(retryState.ready && retryState.sceneMode === 'box' && !retryState.modelLoading && retryModels[0].visible,
  'failure_preserves_working_model');
fail = false;
await retryLoader.select('scene');
check(retryState.modelLoadAttempts.scene === 2 && retryState.sceneMode === 'scene', 'failed_promise_removed_and_retry_succeeds');
await assert.rejects(retryLoader.select('unknown'), /Unknown model mode/);
check(retryState.sceneMode === 'scene', 'invalid_mode_cannot_replace_visible_model');
const result = { status: 'passed', checks, count: checks.length, source,
  actual_browser_verified: false, performance_claim: false, counted: false };
await writeFile(resolve(output, 'RESULT.json'), JSON.stringify(result, null, 2)+'\n');
console.log(JSON.stringify(result));
