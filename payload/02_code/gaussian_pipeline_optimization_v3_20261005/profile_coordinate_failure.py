"""Preserve complete native error and real SIGINT prefixes for operand reuse."""

import hashlib
import inspect
import os
from pathlib import Path
import signal
import sys

import profile_native_zero_fault as preserved
import profile_coordinate_scene as coordinate


def main():
    index = sys.argv.index("--fault-kind")
    kind = sys.argv[index + 1]
    if kind not in ("error", "sigint"):
        raise ValueError("The native fault kind must be explicit")
    sys.argv = sys.argv[:index] + sys.argv[index + 2:]
    source = inspect.getsource(preserved.main)
    edits = [
        ('parent.native_feature_zero(path, enabled)', 'parent.native_coordinate_reuse(path, enabled)'),
        ('namespace.update(native_feature_zero=fault_zero', 'namespace.update(native_coordinate_reuse=fault_zero'),
        ('"R157 registered original distCUDA2 failure"', '"R169 registered native distance failure"'),
        ('evidence["actual_distance_argument"] = {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}',
         'evidence["actual_distance_argument"] = {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device), "stride": list(value.stride()), "bytes": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}'),
        ('                        evidence["actual_original_model_prefix"] = prefix(model)',
         '                        import torch\n                        torch.cuda.synchronize()\n                        evidence["CUDA_drain_returned"] = True\n                        evidence["actual_original_model_prefix"] = prefix(model)'),
        ('    (output / "EXECUTED_FAULT_OBSERVER.py").write_text',
         '    checks["actual_CUDA_drain_returned"] = evidence["CUDA_drain_returned"]\n    checks["no_success_result_published"] = not (output / "RESULT.json").exists()\n    (output / "EXECUTED_COORDINATE_FAILURE.py").write_text'),
    ]
    for before, after in edits:
        if before.startswith('"R157'):
            if source.count(before) != 2:
                raise ValueError("The preserved error text boundary changed")
        elif source.count(before) != 1:
            raise ValueError("The preserved complete failure boundary changed: " + before[:80])
        source = source.replace(before, after)
    if kind == "sigint":
        edits = [
            ('raise OSError("R169 registered native distance failure")',
             'os.kill(os.getpid(), signal.SIGINT)\n            raise AssertionError("The real SIGINT was not delivered")'),
            ('    except OSError as error:\n        if str(error) != "R169 registered native distance failure":\n            raise\n',
             '    except KeyboardInterrupt as error:\n        pass\n'),
            ('"expected_original_native_failure"', '"expected_original_native_SIGINT"'),
        ]
        for before, after in edits:
            if source.count(before) != 1:
                raise ValueError("The preserved real signal boundary changed")
            source = source.replace(before, after)
    namespace = dict(preserved.main.__globals__)
    namespace.update(parent=coordinate, ZeroSceneGuard=coordinate.CoordinateSceneGuard,
                     __file__=str(Path(__file__).resolve()), os=os, signal=signal, hashlib=hashlib)
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
