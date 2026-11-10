"""Bind original resource closure to one private array-matching output tree."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
DOC = ROOT/"research/2026-10-05-gaussian-optimization-round3-50"
OUT = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/fm001"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
