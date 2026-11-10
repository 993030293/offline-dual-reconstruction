"""Bind the original diagnostic node to the complete batched Scene profile."""

import inspect
from pathlib import Path
import sys

import scene_transform_cost_node as controller


def main():
    before = inspect.getsource(controller.linux_identity_check)
    anchor = "/profile_scene_transform_cost.py"
    if before.count(anchor) != 1:
        raise ValueError("The existing explicit Linux Scene worker admission changed")
    namespace = dict(controller.linux_identity_check.__globals__)
    exec(compile(before.replace(anchor, "/profile_batched_camera_scene.py"), "<owned_complete_batch_Scene_drain>", "exec"), namespace)
    original_check = controller.linux_identity_check
    controller.linux_identity_check = namespace["linux_identity_check"]
    node = Path(sys.argv[sys.argv.index("--node") + 1]).resolve()
    node.relative_to(controller.OUT.resolve())
    saved = node.with_name(node.name + "_BATCH_ADAPTER.py")
    if saved.exists():
        raise FileExistsError("The complete Scene adapter source record must be fresh")
    saved.write_bytes(Path(__file__).read_bytes())
    try:
        controller.main()
    finally:
        controller.linux_identity_check = original_check


if __name__ == "__main__":
    main()
