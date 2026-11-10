"""Keep exact worker ownership for the actual complete consumer error paths."""

import inspect
from pathlib import Path
import sys

import scene_transform_cost_node as controller


def main():
    original, prior_out = controller.linux_identity_check, controller.OUT
    before = inspect.getsource(original)
    anchor = "/profile_scene_transform_cost.py"
    if before.count(anchor) != 1:
        raise ValueError("The original exact Linux worker admission changed")
    namespace = dict(original.__globals__)
    exec(compile(before.replace(anchor, "/profile_faulted_camera_fingerprint.py"), "<owned_full_digest_fault_drain>", "exec"), namespace)
    controller.linux_identity_check = namespace["linux_identity_check"]
    controller.OUT = controller.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001"
    node = Path(sys.argv[sys.argv.index("--node")+1]).resolve()
    node.relative_to(controller.OUT.resolve())
    saved = node.with_name(node.name + "_FAULT_ADAPTER.py")
    if saved.exists():
        raise FileExistsError("The original full error controller record must be fresh")
    saved.write_bytes(Path(__file__).read_bytes())
    try:
        controller.main()
    finally:
        controller.linux_identity_check, controller.OUT = original, prior_out


if __name__ == "__main__":
    main()
