"""Apply the existing bounded Scene owner to explicit zero-feature entries."""

import inspect
from pathlib import Path
import sys

import scene_transform_cost_node as controller


def main():
    previous_check, previous_out = controller.linux_identity_check, controller.OUT
    source = inspect.getsource(previous_check)
    anchor = "/profile_scene_transform_cost.py"
    if source.count(anchor) != 1:
        raise ValueError("The original Linux owned process boundary changed")
    allowed = ("check_native_feature_zero.py", "profile_native_feature_zero.py",
               "check_native_zero_transfer_v2.py")
    prefix = "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/"
    entries = [name for name in allowed if prefix + name in sys.argv]
    if len(entries) != 1:
        raise ValueError("The concrete owned Linux entry must be explicit and unique")
    namespace = dict(previous_check.__globals__)
    exec(compile(source.replace(anchor, "/" + entries[0]), "<owned_zero_feature_only>", "exec"), namespace)
    controller.linux_identity_check = namespace["linux_identity_check"]
    controller.OUT = controller.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gzi001"
    node = Path(sys.argv[sys.argv.index("--node") + 1]).resolve()
    node.relative_to(controller.OUT.resolve())
    record = node.with_name(node.name + "_ZERO_ADAPTER.py")
    if record.exists():
        raise FileExistsError("The native zero-feature owner requires a fresh source record")
    record.write_bytes(Path(__file__).read_bytes())
    try:
        controller.main()
    finally:
        controller.linux_identity_check, controller.OUT = previous_check, previous_out


if __name__ == "__main__":
    main()
