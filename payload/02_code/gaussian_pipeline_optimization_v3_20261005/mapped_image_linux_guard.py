"""Measure and bound the actual Linux process tree through profile exit."""

import os
from pathlib import Path
import resource
import threading
import time

import psutil


class LinuxRSSGuard:
    def __init__(self, output, publish, maximum_seconds=1750, maximum_bytes=32*1024**3):
        self.output, self.publish = Path(output), publish
        self.maximum_seconds, self.maximum_bytes = maximum_seconds, maximum_bytes
        self.process = psutil.Process()
        self.started = time.perf_counter()
        self.stop = threading.Event()
        self.row = {"scope": "WSL", "pid": self.process.pid, "creation_time": self.process.create_time(),
            "status": "running", "maximum_seconds": maximum_seconds, "maximum_RSS_bytes": maximum_bytes,
            "peak_tree_RSS_bytes": 0, "elapsed_seconds": 0., "counted": False}
        self.thread = threading.Thread(target=self.watch, name="owned-Linux-RSS", daemon=True)

    def sample(self):
        rss = self.process.memory_info().rss
        for child in self.process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        self.row.update(peak_tree_RSS_bytes=max(self.row["peak_tree_RSS_bytes"], rss,
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024), elapsed_seconds=time.perf_counter()-self.started)
        if rss > self.maximum_bytes or self.row["elapsed_seconds"] > self.maximum_seconds:
            raise RuntimeError("An original Linux memory or time guard was exceeded")

    def watch(self):
        try:
            while not self.stop.wait(.1):
                self.sample()
                self.publish(self.output/"WSL_RESOURCE.json", self.row)
        except BaseException as error:
            self.row.update(status="failed", error=repr(error))
            self.publish(self.output/"WSL_RESOURCE.json", self.row)
            os._exit(73)

    def __enter__(self):
        self.sample()
        self.publish(self.output/"WSL_RESOURCE.json", self.row)
        self.thread.start()
        return self

    def __exit__(self, kind, error, trace):
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("The owned Linux RSS observer did not join")
        self.sample()
        self.row.update(status="completed" if kind is None else "failed", error=None if error is None else repr(error))
        self.publish(self.output/"WSL_RESOURCE.json", self.row)

