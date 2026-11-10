"""Inject a private value mismatch or a real SIGINT in native restore comparison."""

from contextlib import contextmanager
import ctypes
import inspect
import os
from pathlib import Path
import signal
import textwrap
import threading


@contextmanager
def actual_restore_fault(adapter, torch, output, kind):
    if kind not in ("mismatch", "cancel"):
        raise ValueError("The actual restore fault kind is not registered")
    original_prepare = adapter.RecoveryState.prepare
    original_equal, original_eq = torch.equal, torch.eq
    ready, stop = threading.Event(), threading.Event()
    trace = {"kind": kind, "comparison_scope_live": False, "actual_CUDA_compare_API_calls": 0,
        "private_second_state_mutated": False, "actual_SIGINT_sent": False}
    source = textwrap.dedent(inspect.getsource(original_prepare))
    before = "    independent = torch.load(self.resume, weights_only=False)"
    after = before + '\n    _fault_trace["comparison_scope_live"] = True'
    if kind == "mismatch":
        after += '\n    independent["gaussian_capture"][1].reshape(-1)[0].add_(1.0)\n    _fault_trace["private_second_state_mutated"] = True'
    if source.count(before) != 1:
        raise ValueError("The frozen original RecoveryState.prepare changed")
    source = source.replace(before, after)
    namespace = dict(original_prepare.__globals__)
    namespace["_fault_trace"] = trace
    namespace["exact_equal"] = lambda left, right: adapter.exact_equal(left, right)
    exec(compile(source, "<private_native_restore_fault>", "exec"), namespace)

    def trigger(left):
        if trace["comparison_scope_live"] and isinstance(left, torch.Tensor) and left.is_cuda:
            trace["actual_CUDA_compare_API_calls"] += 1
            ready.set()

    def observed_equal(left, right):
        value = original_equal(left, right)
        trigger(left)
        return value

    def observed_eq(left, right, **kwargs):
        value = original_eq(left, right, **kwargs)
        trigger(left)
        return value

    def deliver():
        ready.wait()
        if not stop.is_set():
            library = ctypes.CDLL(None)
            library.kill.argtypes = [ctypes.c_int, ctypes.c_int]
            library.kill.restype = ctypes.c_int
            trace["actual_SIGINT_sent"] = library.kill(os.getpid(), signal.SIGINT) == 0

    thread = threading.Thread(target=deliver, name="owned-R168-native-SIGINT") if kind == "cancel" else None
    adapter.RecoveryState.prepare = namespace["prepare"]
    torch.equal, torch.eq = observed_equal, observed_eq
    if thread is not None:
        thread.start()
    try:
        yield trace
    except BaseException as error:
        trace["actual_exception_type"] = type(error).__name__
        trace["actual_exception_message"] = str(error)
        raise
    finally:
        stop.set()
        ready.set()
        if thread is not None:
            thread.join(timeout=2)
        torch.equal, torch.eq = original_equal, original_eq
        adapter.RecoveryState.prepare = original_prepare
        trace["original_prepare_and_tensor_API_bindings_restored"] = (
            adapter.RecoveryState.prepare is original_prepare and torch.equal is original_equal and torch.eq is original_eq)
        trace["owned_signal_thread_closed"] = thread is None or not thread.is_alive()
        adapter.atomic_json(Path(output) / "ACTUAL_FAULT_TRACE.json", trace)
        if not trace["owned_signal_thread_closed"]:
            raise RuntimeError("The owned signal thread did not drain")

