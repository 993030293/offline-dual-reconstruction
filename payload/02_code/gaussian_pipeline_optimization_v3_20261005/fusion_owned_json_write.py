"""Retain the original write semantics with a distinct restricted fusion root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = Path(__file__).with_name("owned_json_write.py")
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
OLD = 'ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sp001"'
NEW = 'ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/fusion_geometry_sift_001"'
if SOURCE.count(OLD) != 1:
    raise ValueError("The original restricted write source changed")
COMPOSED_SOURCE = SOURCE.replace(OLD, NEW)
NAMESPACE = {"__file__":str(SOURCE_PATH),"__name__":__name__+".private"}
exec(compile(COMPOSED_SOURCE,str(SOURCE_PATH),"exec"),NAMESPACE)
OwnedAtomicJSON = NAMESPACE["OwnedAtomicJSON"]
owned_atomic_json = NAMESPACE["owned_atomic_json"]
ORIGINAL = NAMESPACE["ORIGINAL"]
