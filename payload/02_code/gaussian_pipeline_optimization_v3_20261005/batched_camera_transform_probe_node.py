"""Bind the preserved diagnostic controller to the exact batch semantic probe."""

import inspect
from pathlib import Path
import scene_transform_cost_node as controller


def main():
    original_check = controller.linux_identity_check
    before = inspect.getsource(original_check)
    anchor = "/profile_scene_transform_cost.py"
    if before.count(anchor) != 1:
        raise ValueError("The existing explicit Linux worker admission changed")
    namespace = dict(original_check.__globals__)
    exec(compile(before.replace(anchor, "/check_batched_camera_transforms.py"), "<owned_batch_camera_probe_drain>", "exec"), namespace)
    controller.linux_identity_check = namespace["linux_identity_check"]
    controller.OUT.mkdir(parents=True, exist_ok=True)
    saved = controller.OUT / "BATCH_PROBE_CONTROLLER_SOURCE_001.py"
    if saved.exists():
        raise FileExistsError("The batch probe source record must be fresh")
    saved.write_bytes(Path(__file__).read_bytes())
    try:
        controller.main()
    finally:
        controller.linux_identity_check = original_check


if __name__ == "__main__":
    main()
