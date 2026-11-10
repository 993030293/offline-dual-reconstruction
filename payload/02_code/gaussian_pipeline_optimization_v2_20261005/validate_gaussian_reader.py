"""Load a complete actual Gaussian model and fingerprint every native parameter."""

import argparse
import hashlib
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import activate_reference, atomic_json, data_identity
from vectorized_gaussian_read import vectorized_gaussian_read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "model", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--mode", choices=["baseline", "vectorized"], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = activate_reference(args.runtime)
    import torch
    from scene.gaussian_model import GaussianModel
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    model = GaussianModel(3)
    with vectorized_gaussian_read(args.mode == "vectorized") as trace:
        before = time.perf_counter()
        model.load_ply(str(args.model))
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-before
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        gpu = torch.cuda.max_memory_reserved()
        fields = {}
        for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
            value = getattr(model, name)
            fields[name] = {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype),
                "device": str(value.device), "requires_grad": value.requires_grad,
                "sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            if not torch.isfinite(value).all():
                raise FloatingPointError("A complete Gaussian model contains a nonfinite parameter")
    if peak > 32*1024**3 or gpu > 12*1024**3:
        raise MemoryError("A complete Gaussian read exceeded its native RAM or CUDA budget")
    result = {"status": "actual_model_loaded_pending_pair_oracle", "mode": args.mode, "load_seconds": elapsed,
        "peak_RSS_bytes": peak, "peak_CUDA_reserved_bytes": gpu, "parameters": fields,
        "active_degree": model.active_sh_degree, "maximum_degree": model.max_sh_degree,
        "model": data_identity(args.model), "runtime": metadata, "trace": trace,
        "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("vectorized_gaussian_read.py"))], "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    print({key: result[key] for key in ("mode", "load_seconds", "peak_RSS_bytes", "peak_CUDA_reserved_bytes")}, flush=True)


if __name__ == "__main__":
    main()
