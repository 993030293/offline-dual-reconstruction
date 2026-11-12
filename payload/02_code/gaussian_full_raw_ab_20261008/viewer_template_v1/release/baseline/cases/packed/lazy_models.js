export function createLazyModels({ scene, state, load, requestRender }) {
  const models = new Map(), pending = new Map();
  let version = 0;
  state.loadedModels = [];
  state.modelLoadAttempts = { scene: 0, box: 0 };
  state.fullGaussianCount = 0;
  state.boxGaussianCount = 0;
  async function ensure(mode) {
    if (!['scene', 'box'].includes(mode)) throw new Error('Unknown model mode');
    if (models.has(mode)) return models.get(mode);
    if (pending.has(mode)) return pending.get(mode);
    state.modelLoadAttempts[mode]++;
    const task = (async () => {
      const model = await load(mode);
      await model.initialized;
      model.visible = false;
      scene.add(model);
      models.set(mode, model);
      state.loadedModels = [...models.keys()].sort();
      state[mode === 'scene' ? 'fullGaussianCount' : 'boxGaussianCount'] = model.numSplats;
      return model;
    })();
    pending.set(mode, task);
    try { return await task; }
    finally { if (pending.get(mode) === task) pending.delete(mode); }
  }
  async function select(mode) {
    const token = ++version;
    state.requestedSceneMode = mode;
    state.modelLoading = true;
    try {
      const model = await ensure(mode);
      if (token !== version) return { applied: false, mode };
      for (const [name, value] of models) value.visible = name === mode;
      state.sceneMode = mode;
      state.modelLoading = false;
      state.ready = true;
      requestRender();
      return { applied: true, mode, count: model.numSplats };
    } catch (error) {
      if (token === version) state.modelLoading = false;
      error.modelMode = mode;
      error.isCurrentRequest = token === version;
      throw error;
    }
  }
  return { select, ensure, loaded: () => [...models.keys()].sort() };
}
