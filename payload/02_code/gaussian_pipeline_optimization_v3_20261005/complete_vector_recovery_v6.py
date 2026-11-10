"""Run the unchanged real interruption protocol in a new V6 output directory."""

from pathlib import Path

from profile_vector_full_export import BASE
from runtime import atomic_json, data_identity


def main():
    if not all(__import__("json").loads((BASE / "v6_completion_001/p0/RESULT.json").read_text())["checks"].values()):
        raise AssertionError("Current full evidence identity must pass before real recovery")
    original = Path(__file__).with_name("check_vector_full_export_recovery.py")
    source = original.read_text(encoding="utf-8")
    before, after = 'output = BASE / "v0"', 'output = BASE / "v6_completion_001/recovery"'
    if source.count(before) != 1:
        raise AssertionError("The exact original recovery entry changed")
    executed = source.replace(before, after)
    own = BASE / "v6_completion_001"
    (own / "EXECUTED_RECOVERY_V6.py").write_text(executed, encoding="utf-8")
    atomic_json(own / "RECOVERY_BRIDGE_CONTRACT.json", {"sources": [data_identity(original), data_identity(Path(__file__))],
        "sole_original_text_replacement": {"before": before, "after": after}, "new_optimization_count": 0})
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_V6_recovery_bridge"}
    exec(compile(executed, str(original), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
