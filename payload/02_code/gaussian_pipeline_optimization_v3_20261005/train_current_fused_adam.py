"""Bind the registered fused Adam comparison to the original camera entry."""

import argparse
import inspect
from pathlib import Path
import sys
import textwrap
import types

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/fused_adam_current_001"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
import train_case


def compose_main():
    source = textwrap.dedent(inspect.getsource(train_case.main))
    before = '''    if args.vectorized_ply and (args.compact_image_cache or args.resident_fused_adam
            or args.pyramid_until or args.target_density_roi or args.dense_adam_backend != "native"
            or args.ssim_backend != "reference" or args.optimizer_type not in (None, "default")):
        raise ValueError("The first serialization comparison excludes unaccepted training candidates")'''
    after = before.replace('args.dense_adam_backend != "native"',
                           'args.dense_adam_backend not in ("native", "fused")')
    if source.count(before) != 1:
        raise ValueError("The registered original serialization comparison guard changed")
    return source.replace(before, after)


def argument_prefix(source):
    anchor = "    args.output.mkdir(parents=True, exist_ok=False)"
    if source.count(anchor) != 1:
        raise ValueError("The original argument validation boundary changed")
    return source[:source.index(anchor)] + "    return vars(args)\n"


def bind_original_globals(source):
    namespace = dict(train_case.main.__globals__)
    exec(compile(source, "<registered_original_global_training_entry>", "exec"), namespace)
    compiled = namespace["main"]
    return types.FunctionType(compiled.__code__, train_case.main.__globals__,
                              compiled.__name__, compiled.__defaults__, compiled.__closure__)


def main():
    argv = sys.argv[1:]
    if argv.count("--output") != 1 or argv.count("--dense-adam-backend") != 1:
        raise ValueError("A unique output and explicit dense Adam mode are required")
    output = Path(argv[argv.index("--output") + 1])
    output.resolve().relative_to(OUT.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    composed = compose_main()
    source_path = output.parent / (output.name + "_composed_train_case.py")
    if source_path.exists() or output.exists():
        raise FileExistsError("Never overwrite a current fused comparison")
    source_path.write_text(composed, encoding="utf-8")
    sources = [data_identity(Path(__file__)), data_identity(V1 / "train_case.py"),
               data_identity(V1 / "optimizer_adapter.py"), data_identity(V1 / "ply_serialization.py")]
    contract = output.parent / (output.name + "_fused_entry_contract.json")
    atomic_json(contract, {"status": "registered_before_original_training", "sources": sources,
        "compiled_source": data_identity(source_path), "explicit_mode": argv[argv.index("--dense-adam-backend") + 1],
        "only_changed_argument_guard": "allow registered native or fused with original vectorized byte oracle",
        "original_files_modified": False, "new_optimization_count": 0})
    original = train_case.main
    bound = bind_original_globals(composed)
    if bound.__globals__ is not original.__globals__:
        raise AssertionError("The current entry must observe the original live adapter globals")

    def registered_main():
        if (bound.__globals__["camera_loader_adapter"] is train_case.camera_loader_adapter
                and bound.__globals__["atomic_json"] is train_case.atomic_json):
            return bound()
        raise AssertionError("The original live camera and publication bindings were not inherited")

    train_case.main = registered_main
    import psutil
    process = psutil.Process()
    atomic_json(output.parent / (output.name + "_PROCESS_IDENTITY.json"), {
        "scope": "WSL", "pid": process.pid, "creation_time": process.create_time()})
    import train_camera_execution
    import checkpoint_adapter
    original_json = train_case.atomic_json
    original_recovery = checkpoint_adapter.RecoveryState
    binding = {"entry_source_sha256": [row["sha256"] for row in sources],
               "compiled_source_sha256": data_identity(source_path)["sha256"]}

    class RegisteredRecovery(original_recovery):
        def __init__(self, output, identity, interval, resume=None, crash_after=None):
            if "registered_fused_entry" in identity:
                raise ValueError("A current fused recovery binding already exists")
            super().__init__(output, {**identity, "registered_fused_entry": binding},
                             interval, resume, crash_after)

    def publish(path, value):
        if Path(path) == output / "STATE.json" and not (output / "PROCESS_IDENTITY.json").exists():
            atomic_json(output / "PROCESS_IDENTITY.json", {
                "scope": "WSL", "pid": process.pid, "creation_time": process.create_time()})
        original_json(path, value)

    train_case.atomic_json = publish
    checkpoint_adapter.RecoveryState = RegisteredRecovery
    try:
        train_camera_execution.main()
        if sources != [data_identity(Path(row["path"])) for row in sources]:
            raise ValueError("The registered current fused entry sources changed")
        atomic_json(output / "REGISTERED_FUSED_ENTRY.json", {
            "status": "actual_original_entry_completed_acceptance_pending", "sources": sources,
            "compiled_source": data_identity(source_path), "new_optimization_count": 0})
    finally:
        train_case.main = original
        train_case.atomic_json = original_json
        checkpoint_adapter.RecoveryState = original_recovery


if __name__ == "__main__":
    main()
