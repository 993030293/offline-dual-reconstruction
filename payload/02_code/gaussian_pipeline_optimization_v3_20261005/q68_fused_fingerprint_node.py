"""Reuse the exact complete Scene process owner with a composed Linux entry."""

import inspect
from pathlib import Path
import sys

import scene_transform_cost_node as controller


def main():
    previous_check, previous_out = controller.linux_identity_check, controller.OUT
    source = inspect.getsource(previous_check)
    anchor = "/profile_scene_transform_cost.py"
    if source.count(anchor) != 1:
        raise ValueError("The original owned Linux process boundary changed")
    namespace = dict(previous_check.__globals__)
    exec(compile(source.replace(anchor, "/profile_q68_fused_fingerprint.py"), "<owned_complete_composed_Scene>", "exec"), namespace)
    controller.linux_identity_check = namespace["linux_identity_check"]
    controller.OUT = controller.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scfq001"
    node = Path(sys.argv[sys.argv.index("--node") + 1]).resolve()
    node.relative_to(controller.OUT.resolve())
    record = node.with_name(node.name + "_COMPOSED_ADAPTER.py")
    if record.exists():
        raise FileExistsError("The composed owner requires a fresh source record")
    record.write_bytes(Path(__file__).read_bytes())
    try:
        controller.main()
    finally:
        controller.linux_identity_check, controller.OUT = previous_check, previous_out


if __name__ == "__main__":
    main()
