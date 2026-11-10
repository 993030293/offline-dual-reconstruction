"""Keep original resource ownership for the actual complete file identity error."""

import inspect
from pathlib import Path
import sys

import scene_transform_cost_node as controller


def main():
    previous_check, previous_out = controller.linux_identity_check, controller.OUT
    source = inspect.getsource(previous_check)
    if source.count("/profile_scene_transform_cost.py") != 1:
        raise ValueError("The original Linux owner admission boundary changed")
    namespace = dict(previous_check.__globals__)
    exec(compile(source.replace("/profile_scene_transform_cost.py", "/profile_faulted_q68_fingerprint.py"), "<owned_complete_identity_error>", "exec"), namespace)
    controller.linux_identity_check = namespace["linux_identity_check"]
    controller.OUT = controller.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scfq001"
    node = Path(sys.argv[sys.argv.index("--node") + 1]).resolve()
    node.relative_to(controller.OUT.resolve())
    saved = node.with_name(node.name + "_FAULT_ADAPTER.py")
    if saved.exists():
        raise FileExistsError("A complete identity error owner must have a fresh record")
    saved.write_bytes(Path(__file__).read_bytes())
    try:
        controller.main()
    finally:
        controller.linux_identity_check, controller.OUT = previous_check, previous_out


if __name__ == "__main__":
    main()
