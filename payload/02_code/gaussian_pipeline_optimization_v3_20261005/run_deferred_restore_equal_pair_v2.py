"""Add owned failure cleanup to the frozen complete-pair observation driver."""

from pathlib import Path
import subprocess
import sys

import psutil


class OwnedProcesses:
    CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW
    check_output = staticmethod(subprocess.check_output)

    def __init__(self):
        self.items = []

    def Popen(self, command, **kwargs):
        process = subprocess.Popen(command, **kwargs)
        self.items.append((process, psutil.Process(process.pid).create_time(), command))
        return process

    def close_failed(self):
        for process, born, command in reversed(self.items):
            if process.poll() is not None:
                process.wait()
                continue
            folder = command[command.index("--output") + 1]
            code = subprocess.run(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
                "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
                "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/stop_owned_restore_linux_001.py", folder],
                timeout=25, creationflags=subprocess.CREATE_NO_WINDOW).returncode
            if code != 0:
                raise RuntimeError("Native cleanup failed; the saved process requires investigation")
            try:
                current = psutil.Process(process.pid)
                if current.create_time() == born and process.poll() is None:
                    process.terminate()
                process.wait(timeout=10)
            except psutil.NoSuchProcess:
                pass


def main():
    source = Path(__file__).with_name("run_deferred_restore_equal_pair_001.py").read_text(encoding="utf-8")
    before = '(Path(__file__).name, "deferred_checkpoint_equal_001.py",'
    after = '(Path(__file__).name, "run_deferred_restore_equal_pair_001.py", "stop_owned_restore_linux_001.py", "deferred_checkpoint_equal_001.py",'
    if source.count(before) != 1:
        raise ValueError("The original complete-pair source changed")
    source = source.replace(before, after)
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_complete_pair_cleanup"}
    exec(compile(source, "<frozen_complete_pair_with_owned_failure_drain>", "exec"), namespace)
    owner = OwnedProcesses()
    namespace["subprocess"] = owner
    try:
        namespace["main"]()
    except BaseException:
        owner.close_failed()
        raise


if __name__ == "__main__":
    main()
