// Resolve only model-bound filter rules verified against the local training configuration.
export function resolveRenderFilter(contract, assets) {
  if (contract?.schema !== 1 || contract.rasterizeMode !== "classic" || contract.eps2d !== 0.3
      || contract.opacityCompensation !== false || contract.antialiasing !== false) {
    throw new Error("Unsupported or ambiguous training filter contract");
  }
  if (!/^[a-f0-9]{64}$/.test(contract.trainingConfigSha256 ?? "")) {
    throw new Error("Missing model training configuration identity");
  }
  for (const label of ["scene", "box"]) {
    const expected = assets?.[label]?.source_ply_sha256;
    if (!/^[a-f0-9]{64}$/.test(expected ?? "") || contract.models?.[label] !== expected) {
      throw new Error("The filter contract belongs to a different display model");
    }
  }
  return { preBlurAmount: contract.eps2d, blurAmount: 0 };
}
