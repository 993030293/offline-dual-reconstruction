"""Admit only the concrete prefetched-camera regression worker for owned drain."""

import inspect
from pathlib import Path
import sys
import scene_transform_cost_node as controller


def main():
    before = inspect.getsource(controller.linux_identity_check)
    anchor = "/profile_scene_transform_cost.py"
    if before.count(anchor) != 1:
        raise ValueError("The concrete Linux regression worker admission changed")
    namespace = dict(controller.linux_identity_check.__globals__)
    exec(compile(before.replace(anchor, "/check_batch_prefetch_regression.py"), "<owned_prefetch_regression_drain>", "exec"), namespace)
    original = controller.linux_identity_check
    controller.linux_identity_check = namespace["linux_identity_check"]
    node = Path(sys.argv[sys.argv.index("--node") + 1]).resolve()
    node.relative_to(controller.OUT.resolve())
    snapshot = node.with_name(node.name + "_REGRESSION_ADAPTER.py")
    if snapshot.exists():
        raise FileExistsError("The concrete regression source record must be fresh")
    snapshot.write_bytes(Path(__file__).read_bytes())
    try:
        controller.main()
    finally:
        controller.linux_identity_check = original


if __name__ == "__main__":
    main()
