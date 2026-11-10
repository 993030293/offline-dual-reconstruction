"""Inspect native Boolean and owned-index result layouts without changing gates."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    rows = []
    for count in (0, 1, 2, 7, 31):
        for retained in sorted({0, min(1, count), count // 2, count}):
            for tail in ((3,), (1,), (1, 3), (3, 1), (15, 3), (1, 1, 3)):
                for strided in (False, True):
                    source = torch.arange(count * int(torch.tensor(tail).prod()), device="cuda", dtype=torch.float32).reshape(count, *tail)
                    if strided:
                        backing = torch.zeros((*source.shape[:-1], source.shape[-1] * 2), device="cuda")
                        backing[..., ::2] = source
                        source = backing[..., ::2]
                    mask = torch.arange(count, device="cuda") < retained
                    native = source[mask]
                    selected = source.index_select(0, mask.nonzero(as_tuple=True)[0])
                    empty_native = source[:0][torch.empty(0, dtype=torch.bool, device="cuda")]
                    rows.append({"count": count, "retained": retained, "tail": list(tail), "strided": strided,
                        "source_stride": list(source.stride()), "native_stride": list(native.stride()),
                        "index_stride": list(selected.stride()), "empty_native_stride": list(empty_native.stride()),
                        "values_exact": torch.equal(native, selected),
                        "empty_metadata_matches_actual_stride": native.stride() == empty_native.stride()})
    result = {"status": "diagnostic_only", "torch": torch.__version__, "source": data_identity(Path(__file__)),
        "rows": rows, "all_values_exact": all(r["values_exact"] for r in rows),
        "all_empty_metadata_matches_native_stride": all(r["empty_metadata_matches_actual_stride"] for r in rows),
        "counted": False, "original_gate_changed": False}
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps({"rows": len(rows), "all_values_exact": result["all_values_exact"],
          "all_empty_metadata_matches_native_stride": result["all_empty_metadata_matches_native_stride"]}), flush=True)


if __name__ == "__main__":
    main()
