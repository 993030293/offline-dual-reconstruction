"""Reuse existing owned-process protection for an isolated Scene cost diagnostic."""

import difflib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import data_identity

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sc001"


def linux_identity_check(identity, drain=False):
    payload = json.dumps({"pid": identity["pid"], "creation_time": identity["creation_time"]})
    code = "import json,psutil; i=json.loads(" + repr(payload) + "); result=dict(i,alive=False,cmdline=[])\n"
    code += "try:\n p=psutil.Process(i['pid']); result.update(alive=p.create_time()==i['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE,cmdline=p.cmdline())\nexcept psutil.NoSuchProcess: pass\n"
    if drain:
        entry = "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/profile_scene_transform_cost.py"
        code += "if result['alive']:\n assert " + repr(entry) + " in result['cmdline'], 'An unrelated Linux PID cannot be drained'\n p.terminate()\n try: p.wait(timeout=10)\n except psutil.TimeoutExpired: p.kill(); p.wait(timeout=10)\n result.update(alive=False,forced_owned_drain=True)\n"
    code += "print(json.dumps(result))"
    return json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=30))


def main():
    original = CODE / "streamed_ply_node_002.py"
    before = original.read_text(encoding="utf-8")
    source_id = data_identity(original)
    replacements = [("\"1750\"", "\"300\""), (" > 1770:", " > 330:"),
                    (" > 1800:", " > 360:"), ("size > 30*1024**3", "size > 12*1024**3")]
    after = before
    for anchor, replacement in replacements:
        if after.count(anchor) != 1:
            raise ValueError("The existing explicit resource bound changed")
        after = after.replace(anchor, replacement)
    anchor = "    args.node.mkdir(parents=True, exist_ok=False)"
    if after.count(anchor) != 1:
        raise ValueError("The original controller source-capture boundary changed")
    after = after.replace(anchor, anchor + "\n    _cost_snapshot(args.node)")
    namespace = {"__file__": str(original), "__name__": "original_Scene_cost_owned_controller"}
    exec(compile(after, str(original), "exec"), namespace)
    namespace["OUT"] = OUT
    namespace["linux_identity_check"] = linux_identity_check
    def snapshot(node):
        (node / "EXECUTED_CONTROLLER.py").write_text(after, encoding="utf-8")
        (node / "CONTROLLER.diff").write_text("".join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="original", tofile="bounded_private_Scene")), encoding="utf-8")
        (node / "ADAPTER_SOURCE.py").write_bytes(Path(__file__).read_bytes())
        (node / "PROFILE_ADAPTER_SOURCE.py").write_bytes((CODE / "profile_scene_transform_cost.py").read_bytes())
    namespace["_cost_snapshot"] = snapshot
    node = Path(sys.argv[sys.argv.index("--node") + 1]).resolve()
    node.relative_to(OUT.resolve())
    namespace["main"]()
    if source_id != data_identity(original):
        raise AssertionError("The original resource controller changed")


if __name__ == "__main__":
    main()
