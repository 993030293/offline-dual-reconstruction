"""Expand validated native stereo templates to the complete ordered image list."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/ud001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json


def complete_configuration(dataset, image_names, template_names, proof):
    dataset, proof = Path(dataset).resolve(), Path(proof).resolve()
    dataset.relative_to(OWN.resolve())
    proof.relative_to(OWN.resolve())
    if dataset == proof or dataset in proof.parents:
        raise ValueError("Configuration evidence must be outside the dataset artifact")
    names, templates = tuple(image_names), tuple(template_names)
    for values in (names, templates):
        if not values or len(values) > 100000 or len(set(values)) != len(values):
            raise ValueError("A complete unique bounded image list is required")
        if any(type(name) is not str or not name or len(name.encode("utf-8")) > 4096
               or Path(name).name != name or any(char in name for char in "\r\n\x00") for name in values):
            raise ValueError("A native flat image name contains an unsupported path or separator")
    if not set(templates).issubset(names):
        raise ValueError("Native template images must belong to the complete image list")
    paths = {name: dataset / "stereo" / name for name in ("fusion.cfg", "patch-match.cfg")}
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError("Original owned native configuration files are required")
    before = {name: path.read_bytes() for name, path in paths.items()}
    newline = b"\r\n" if b"\r\n" in before["fusion.cfg"] else b"\n"
    fusion = b"".join(name.encode("utf-8") + newline for name in templates)
    patch = b"".join(name.encode("utf-8") + newline + b"__auto__, 20" + newline for name in templates)
    if before != {"fusion.cfg": fusion, "patch-match.cfg": patch}:
        raise ValueError("The original native default stereo template differs")
    after = {"fusion.cfg": b"".join(name.encode("utf-8") + newline for name in names),
             "patch-match.cfg": b"".join(name.encode("utf-8") + newline + b"__auto__, 20" + newline for name in names)}
    proof.mkdir(parents=True, exist_ok=False)
    for name, content in before.items():
        (proof / (name + ".native_before")).write_bytes(content)
    for name, content in after.items():
        path = paths[name]
        temporary = path.with_name(path.name + ".owned_complete.tmp")
        with temporary.open("xb") as stream:
            stream.write(content)
        os.replace(temporary, path)
        if path.read_bytes() != content:
            raise OSError("The owned complete native configuration did not persist exactly")
    result = {"status": "complete_native_template_image_denominator_preserved",
              "template_image_names": list(templates), "complete_image_names": list(names),
              "complete_image_count": len(names), "native_auto_source_line_preserved": "__auto__, 20",
              "native_newline_hex": newline.hex(), "all_other_dataset_files_untouched": True,
              "new_optimization_count": 0}
    atomic_json(proof / "RESULT.json", result)
    return result
