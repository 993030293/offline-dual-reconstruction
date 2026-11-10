"""Send a real SIGINT at the complete original native distance boundary."""

import inspect
import os
from pathlib import Path
import signal

import profile_native_zero_fault as parent


def main():
    source = inspect.getsource(parent.main)
    replacements = [
        ('raise OSError("R157 registered original distCUDA2 failure")',
         'os.kill(os.getpid(), signal.SIGINT)\n            raise AssertionError("The native SIGINT was not delivered")'),
        ('    except OSError as error:\n        if str(error) != "R157 registered original distCUDA2 failure":\n            raise\n',
         '    except KeyboardInterrupt as error:\n        pass\n'),
        ('                        evidence["actual_original_model_prefix"] = prefix(model)',
         '                        import torch\n                        torch.cuda.synchronize()\n                        evidence["CUDA_drain_returned"] = True\n                        evidence["actual_original_model_prefix"] = prefix(model)'),
        ('        failure = {"type": type(error).__name__, "message": str(error)}',
         '        failure = {"type": type(error).__name__, "message": str(error)}'),
        ('    (output / "EXECUTED_FAULT_OBSERVER.py").write_text',
         '    checks["real_SIGINT_delivered_and_CUDA_drained"] = failure["type"] == "KeyboardInterrupt" and evidence["CUDA_drain_returned"]\n    checks["no_success_result_published"] = not (output / "RESULT.json").exists()\n    (output / "EXECUTED_SIGINT_OBSERVER.py").write_text'),
        ('"expected_original_native_failure"', '"expected_original_native_SIGINT"'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('The preserved complete native fault observer changed')
        source = source.replace(before, after)
    namespace = dict(parent.main.__globals__)
    namespace.update(__file__=str(Path(__file__).resolve()), os=os, signal=signal)
    exec(compile(source, '<same_complete_Scene_real_SIGINT_observer>', 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
