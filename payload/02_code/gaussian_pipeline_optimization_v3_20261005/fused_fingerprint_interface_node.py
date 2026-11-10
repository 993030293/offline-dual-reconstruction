"""Use exact owner admission for the separate immutable digest interface checks."""

import inspect
from pathlib import Path
import sys

import scene_transform_cost_node as controller


def main():
    original, prior_out = controller.linux_identity_check, controller.OUT
    before = inspect.getsource(original)
    anchor = "/profile_scene_transform_cost.py"
    if before.count(anchor) != 1:
        raise ValueError("The exact original Linux admission changed")
    namespace = dict(original.__globals__)
    exec(compile(before.replace(anchor, "/check_fused_fingerprint_interfaces.py"), "<owned_digest_interface_drain>", "exec"), namespace)
    controller.linux_identity_check = namespace["linux_identity_check"]
    controller.OUT = controller.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001"
    node = Path(sys.argv[sys.argv.index("--node")+1]).resolve()
    node.relative_to(controller.OUT.resolve())
    saved = node.with_name(node.name + "_INTERFACE_ADAPTER.py")
    if saved.exists():
        raise FileExistsError("The interface owner record must be fresh")
    saved.write_bytes(Path(__file__).read_bytes())
    try:
        controller.main()
    finally:
        controller.linux_identity_check, controller.OUT = original, prior_out


if __name__ == "__main__":
    main()
