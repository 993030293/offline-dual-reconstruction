"""Bind the original resource and owner guard to the full digest consumer."""

import inspect
from pathlib import Path
import sys

import scene_transform_cost_node as controller


def main():
    original_check, original_out = controller.linux_identity_check, controller.OUT
    source = inspect.getsource(original_check)
    anchor = "/profile_scene_transform_cost.py"
    if source.count(anchor) != 1:
        raise ValueError("The original exact owned Linux admission changed")
    namespace = dict(original_check.__globals__)
    exec(compile(source.replace(anchor, "/profile_fused_camera_fingerprint.py"), "<owned_complete_digest_Scene_drain>", "exec"), namespace)
    controller.linux_identity_check = namespace["linux_identity_check"]
    controller.OUT = controller.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001"
    node = Path(sys.argv[sys.argv.index("--node")+1]).resolve()
    node.relative_to(controller.OUT.resolve())
    saved = node.with_name(node.name + "_FUSED_DIGEST_ADAPTER.py")
    if saved.exists():
        raise FileExistsError("The complete digest adapter record must be fresh")
    saved.write_bytes(Path(__file__).read_bytes())
    try:
        controller.main()
    finally:
        controller.linux_identity_check, controller.OUT = original_check, original_out


if __name__ == "__main__":
    main()
