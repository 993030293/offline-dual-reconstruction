"""Preserve original guards around the complete original main-call diagnostic."""

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
    exec(compile(source.replace(anchor, "/profile_composed_scene_main_cost.py"), "<owned_original_main_call_cost>", "exec"), namespace)
    controller.linux_identity_check = namespace["linux_identity_check"]
    controller.OUT = controller.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sclm002"
    node = Path(sys.argv[sys.argv.index("--node") + 1]).resolve()
    node.relative_to(controller.OUT.resolve())
    record = node.with_name(node.name + "_MAIN_COST_ADAPTER.py")
    if record.exists():
        raise FileExistsError("The main cost owner requires a fresh source record")
    record.write_bytes(Path(__file__).read_bytes())
    try:
        controller.main()
    finally:
        controller.linux_identity_check, controller.OUT = previous_check, previous_out


if __name__ == "__main__":
    main()
