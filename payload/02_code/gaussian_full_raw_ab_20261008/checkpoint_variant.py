"""Exercise complete native recovery on the current fresh training input."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V3 = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005"
sys.path[:0] = [str(V1), str(V3)]
from runtime import atomic_json, data_identity
import checkpoint_adapter
import train_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-mechanism", choices=("native", "private", "deferred", "combined"), required=True)
    args, remaining = parser.parse_known_args()
    output = Path(remaining[remaining.index("--output")+1])
    resume = Path(remaining[remaining.index("--resume-file")+1]) if "--resume-file" in remaining else None
    if args.restore_mechanism != "native" and resume is None:
        raise ValueError("A restore mechanism requires a complete fresh checkpoint")
    traces = {}
    base = checkpoint_adapter.RecoveryState
    sources = [Path(__file__), Path(train_variant.__file__), V1 / "checkpoint_adapter.py",
               V3 / "private_checkpoint_load.py", V3 / "deferred_checkpoint_equal_001.py", V3 / "deferred_checkpoint_equal_v2.py"]
    atomic_json(output.parent / (output.name+"_recovery_contract.json"), {"mechanism": args.restore_mechanism,
        "sources": [data_identity(path) for path in sources], "input_checkpoint": data_identity(resume) if resume else None,
        "all_original_state_checks_and_training_formulas_retained": True,
        "sampling_oracle": "all12000 native random-pop camera names exactly equal",
        "final_PLY_diagnostic_tolerance": {"absolute": 1e-6, "relative": 1e-5},
        "atomic_training_nondeterminism_does_not_relax_saved_state_exactness": True})

    class InstrumentedRecovery(base):
        def prepare(self, gaussians, options, scene):
            with ExitStack() as stack:
                if self.resume is not None:
                    from private_checkpoint_load import private_checkpoint_load
                    from deferred_checkpoint_equal_v2 import deferred_checkpoint_equal
                    traces["private"] = stack.enter_context(private_checkpoint_load(self.resume,
                        enabled=args.restore_mechanism in ("private", "combined")))
                    traces["deferred"] = stack.enter_context(deferred_checkpoint_equal(checkpoint_adapter,
                        enabled=args.restore_mechanism in ("deferred", "combined")))
                return super().prepare(gaussians, options, scene)

    old = sys.argv
    checkpoint_adapter.RecoveryState = InstrumentedRecovery
    try:
        sys.argv = [str(Path(train_variant.__file__)), *remaining]
        train_variant.main()
    finally:
        checkpoint_adapter.RecoveryState = base
        sys.argv = old
    atomic_json(output / "RECOVERY_MODULE_TRACE.json", {"mechanism": args.restore_mechanism,
        "traces": traces, "original_RecoveryState_binding_restored": checkpoint_adapter.RecoveryState is base})


if __name__ == "__main__":
    main()
