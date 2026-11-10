"""Bind the original Scene guard and owner admission to the field diagnostic."""

import inspect
from pathlib import Path
import sys

import scene_transform_cost_node as controller


def main():
    before = inspect.getsource(controller.linux_identity_check)
    anchor = "/profile_scene_transform_cost.py"
    if before.count(anchor) != 1:
        raise ValueError("The original exact Linux Scene worker admission changed")
    namespace = dict(controller.linux_identity_check.__globals__)
    exec(compile(before.replace(anchor, "/profile_scene_fingerprint_cost.py"), "<owned_complete_field_Scene_drain>", "exec"), namespace)
    original_check, original_out = controller.linux_identity_check, controller.OUT
    controller.linux_identity_check = namespace["linux_identity_check"]
    controller.OUT = controller.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001"
    node = Path(sys.argv[sys.argv.index("--node") + 1]).resolve()
    node.relative_to(controller.OUT.resolve())
    saved = node.with_name(node.name + "_FIELD_ADAPTER.py")
    if saved.exists():
        raise FileExistsError("The original field controller record must be fresh")
    saved.write_bytes(Path(__file__).read_bytes())
    node.with_name(node.name + "_FIELD_PROFILE.py").write_bytes((controller.CODE / "profile_scene_fingerprint_cost.py").read_bytes())
    try:
        controller.main()
    finally:
        controller.linux_identity_check, controller.OUT = original_check, original_out


if __name__ == "__main__":
    main()
