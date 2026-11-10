"""Reuse the original bounded node for only the registered current Adam leaves."""

import json
from pathlib import Path
import runpy
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/fused_adam_current_001"


def linux_identity_check(identity, drain=False):
    payload = json.dumps({"pid": identity["pid"], "creation_time": identity["creation_time"]})
    code = "import json,psutil; i=json.loads(" + repr(payload) + "); result=dict(i,alive=False,cmdline=[])\n"
    code += "try:\n p=psutil.Process(i['pid']); result.update(alive=p.create_time()==i['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE,cmdline=p.cmdline())\nexcept psutil.NoSuchProcess: pass\n"
    if drain:
        entries = ("check_current_fused_adam.py", "train_current_fused_adam.py", "evaluate_pair.py")
        allowed = tuple("/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/" + name for name in entries[:2])
        allowed += ("/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v1_20261005/evaluate_pair.py",)
        code += "if result['alive']:\n allowed=" + repr(allowed) + "\n assert any(x in result['cmdline'] for x in allowed), 'An unrelated Linux PID cannot be drained'\n p.terminate()\n try: p.wait(timeout=10)\n except psutil.TimeoutExpired: p.kill(); p.wait(timeout=10)\n result.update(alive=False,forced_owned_drain=True)\n"
    code += "print(json.dumps(result))"
    return json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=30))


def main():
    import sys
    sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
    from runtime import data_identity
    original = CODE / "streamed_ply_node_002.py"
    before = data_identity(original)
    namespace = runpy.run_path(str(original), run_name="preserved_current_fused_resource_controller")
    entry = namespace["main"]
    entry.__globals__["OUT"] = OUT
    entry.__globals__["linux_identity_check"] = linux_identity_check
    entry()
    if before != data_identity(original):
        raise ValueError("The original protected controller changed")


if __name__ == "__main__":
    main()
