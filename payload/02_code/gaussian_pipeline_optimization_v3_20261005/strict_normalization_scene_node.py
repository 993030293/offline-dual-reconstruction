"""Retain the existing bounds and drain only the owned arithmetic entry."""

import inspect
from pathlib import Path
import sys

import scene_transform_cost_node as controller


def main():
    previous_check, previous_out, previous_main = controller.linux_identity_check, controller.OUT, controller.main
    source = inspect.getsource(previous_check)
    anchor = "/profile_scene_transform_cost.py"
    if source.count(anchor) != 1:
        raise ValueError("The original Linux owned process boundary changed")
    namespace = dict(previous_check.__globals__)
    exec(compile(source.replace(anchor, "/profile_strict_normalization_scene.py"), "<owned_arithmetic_Scene>", "exec"), namespace)
    controller.linux_identity_check = namespace["linux_identity_check"]
    controller.OUT = controller.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/nn001"
    main_source = inspect.getsource(previous_main)
    boundary = '    exec(compile(after, str(original), "exec"), namespace)'
    addition = "    _guard_exit = '    guard = read(args.node/\"guard.json\")'\n"
    addition += "    if after.count(_guard_exit) != 1:\n        raise ValueError('The original actual guard exit boundary changed')\n"
    addition += "    after = after.replace(_guard_exit, '    atomic_json(args.node/\"ACTUAL_GUARD_EXIT.json\", {\"returncode\": record[\"returncode\"], \"bounded_command\": bounded, \"owner\": owner})\\n' + _guard_exit)\n"
    if main_source.count(boundary) != 1:
        raise ValueError("The original controller compile boundary changed")
    main_namespace = dict(previous_main.__globals__)
    exec(compile(main_source.replace(boundary, addition + boundary), "<original_guard_exit_receipt_only>", "exec"), main_namespace)
    controller.main = main_namespace["main"]
    node = Path(sys.argv[sys.argv.index("--node") + 1]).resolve()
    node.relative_to(controller.OUT.resolve())
    record = node.with_name(node.name + "_ARITHMETIC_ADAPTER.py")
    if record.exists():
        raise FileExistsError("The arithmetic owner requires a fresh source record")
    record.write_bytes(Path(__file__).read_bytes())
    try:
        controller.main()
    finally:
        controller.linux_identity_check, controller.OUT, controller.main = previous_check, previous_out, previous_main


if __name__ == "__main__":
    main()
