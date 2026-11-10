"""Reuse private native OpenCV destination planes within each scoring thread."""

from contextlib import contextmanager
import inspect
import threading
import time

import cv2


class NativeQualityWorkspace:
    def __init__(self):
        self.local = threading.local()
        self.lock = threading.Lock()
        self.states = []
        self.closed = False

    def state(self):
        if self.closed:
            raise RuntimeError("A closed private quality workspace cannot execute")
        state = getattr(self.local, "state", None)
        if state is None:
            state = {"owner_thread": threading.get_ident(), "key": None, "gray": None,
                     "derivative": None, "reduced": None, "calls": 0, "reused": 0,
                     "created": 0, "shape_changes": 0, "maximum_owned_plane_bytes": 0}
            self.local.state = state
            with self.lock:
                self.states.append(state)
        if state["owner_thread"] != threading.get_ident():
            raise RuntimeError("A native quality plane crossed its private thread owner")
        return state

    @staticmethod
    def measure(state):
        total = sum(state[name].nbytes for name in ("gray", "derivative", "reduced") if state[name] is not None)
        state["maximum_owned_plane_bytes"] = max(state["maximum_owned_plane_bytes"], total)

    def gray(self, image):
        state = self.state()
        key = image.shape, image.dtype.str
        if state["key"] != key:
            for name in ("gray", "derivative", "reduced"):
                state[name] = None
            state["key"] = key
            state["shape_changes"] += 1
        destination = state["gray"]
        result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, dst=destination)
        if destination is not None and result is not destination:
            raise RuntimeError("The original supported grayscale destination was not reused")
        state["gray"] = result
        state["calls"] += 1
        state["created" if destination is None else "reused"] += 1
        self.measure(state)
        return result

    def derivative(self, gray):
        state = self.state()
        if gray is not state["gray"]:
            raise ValueError("The original derivative must use its owned current grayscale plane")
        destination = state["derivative"]
        result = cv2.Laplacian(gray, cv2.CV_16S, dst=destination)
        if destination is not None and result is not destination:
            raise RuntimeError("The original supported derivative destination was not reused")
        state["derivative"] = result
        state["calls"] += 1
        state["created" if destination is None else "reused"] += 1
        self.measure(state)
        return result

    def reduced(self, gray):
        state = self.state()
        if gray is not state["gray"]:
            raise ValueError("The original resize must use its owned current grayscale plane")
        destination = state["reduced"]
        result = cv2.resize(gray, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA, dst=destination)
        if destination is not None and result is not destination:
            raise RuntimeError("The original supported resize destination was not reused")
        state["reduced"] = result
        state["calls"] += 1
        state["created" if destination is None else "reused"] += 1
        self.measure(state)
        return result

    def receipt(self):
        fields = ("owner_thread", "calls", "reused", "created", "shape_changes", "maximum_owned_plane_bytes")
        return {"closed": self.closed, "thread_states": [{key: state[key] for key in fields} for state in self.states],
                "thread_count": len(self.states), "calls": sum(state["calls"] for state in self.states),
                "reused": sum(state["reused"] for state in self.states), "created": sum(state["created"] for state in self.states),
                "owned_plane_bytes_after_close": sum(state[name].nbytes for state in self.states
                    for name in ("gray", "derivative", "reduced") if state[name] is not None)}

    def close(self):
        if self.closed:
            return
        for state in self.states:
            for name in ("gray", "derivative", "reduced"):
                state[name] = None
        self.closed = True


@contextmanager
def native_quality_workspace_execution(enabled):
    import quality_histogram as histogram

    started = time.perf_counter()
    original = histogram.make_quality
    workspace = NativeQualityWorkspace() if enabled else None
    source = inspect.getsource(original)
    before = '    namespace["_gray_moments"] = gray_moments\n'
    compile_before = '    exec(compile(modified, "<original_quality_histogram_moments>", "exec"), namespace)\n'
    if source.count(before) != 1 or source.count(compile_before) != 1:
        raise ValueError("The original accepted histogram builder anchors changed")
    namespace = dict(histogram.__dict__)
    namespace["_private_quality_workspace"] = workspace
    source = source.replace(before, before + '    namespace["_native_quality_workspace"] = _private_quality_workspace\n')
    composition = '''    changes = (
        ("gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)", "gray = _native_quality_workspace.gray(image)"),
        ("derivative = cv2.Laplacian(gray, cv2.CV_16S)", "derivative = _native_quality_workspace.derivative(gray)"),
        ("reduced = cv2.resize(gray, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)", "reduced = _native_quality_workspace.reduced(gray)"),
    )
    for before_plane, after_plane in changes:
        if modified.count(before_plane) != 1:
            raise ValueError("The original complete native quality plane anchor changed")
        modified = modified.replace(before_plane, after_plane)
'''
    source = source.replace(compile_before, composition + compile_before)
    exec(compile(source, "<private_native_quality_workspace_builder>", "exec"), namespace)
    candidate = namespace[original.__name__]
    trace = {"enabled": enabled, "setup_seconds": time.perf_counter() - started,
             "native_functions_and_all_parameters_preserved": True, "counted": False}
    if enabled:
        histogram.make_quality = candidate
    try:
        yield trace
    finally:
        before_restore = time.perf_counter()
        expected = candidate if enabled else original
        changed = histogram.make_quality is not expected
        histogram.make_quality = original
        if workspace:
            workspace.close()
        trace.update(restore_seconds=time.perf_counter() - before_restore,
                     original_builder_restored=histogram.make_quality is original,
                     workspace=workspace.receipt() if workspace else None)
        if changed:
            raise RuntimeError("Another writer changed the owned original quality builder")
