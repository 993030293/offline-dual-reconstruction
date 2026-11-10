"""Bind the unchanged artifact worker to the explicitly isolated raw campaign."""

from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    original = root / "02_code/gaussian_pipeline_optimization_v3_20261005/export_qa_artifact_worker.py"
    source = original.read_text(encoding="utf-8")
    before = 'output.resolve().relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/export_qa_dag_001")'
    after = 'output.resolve().relative_to(ROOT / "04_outputs/gaussian_full_raw_ab_20261008")'
    if source.count(before) != 1:
        raise ValueError("The original artifact worker output ownership guard changed")
    namespace = {"__file__": str(original), "__name__": "raw_scope_artifact_worker"}
    exec(compile(source.replace(before, after), "<original_artifact_worker_raw_scope>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
