"""Restrict the original atomic writer to the strict-encoding experiment."""

from pathlib import Path

SOURCE_PATH = Path(__file__).with_name("owned_json_write.py")
source = SOURCE_PATH.read_text(encoding="utf-8")
before = 'ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sp001"'
if source.count(before) != 1:
    raise ValueError("The original owned writer boundary changed")
COMPOSED_SOURCE = source.replace(before,
    'ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup019/R171"')
namespace = {"__file__": str(SOURCE_PATH), "__name__": __name__ + ".private"}
exec(compile(COMPOSED_SOURCE, str(SOURCE_PATH), "exec"), namespace)
owned_atomic_json = namespace["owned_atomic_json"]
