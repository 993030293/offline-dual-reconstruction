"""Fail native preparation at the seventeenth ordered camera on both routes."""

import difflib
import inspect
import json
from pathlib import Path
import sys

import profile_open_image_handoff as entry
sys.path.insert(0, str(entry.ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"))
import prefetch_camera_images as prefetch_parent
from runtime import atomic_json, data_identity


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    output.relative_to(entry.OUT.resolve())
    reference = json.loads((entry.OUT / "c0/t0/control/RESULT.json").read_text())
    fields = ("original_image", "alpha_mask", "world_view_transform", "projection_matrix", "full_proj_transform", "camera_center")
    expected = [{name: row[name] for name in fields} for row in reference["scientific"]["rows"][:16]]
    target = reference["scientific"]["rows"][16]["name"]
    registered = list((entry.ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003/dataset/images").glob("*.jpg"))
    if sum(path.name == target for path in registered) != 1:
        raise ValueError("The actual fault target must match one complete registered filename")
    prefix = []

    def prepare(native, image, resolution):
        if Path(image.filename).name == target:
            raise OSError("Forced original native image preparation failure at17")
        return native(image, resolution)

    def observe(camera):
        def tensor(value):
            if value is None:
                return None
            import hashlib
            return {"shape": list(value.shape), "stride": list(value.stride()),
                    "storage_offset": value.storage_offset(), "requires_grad": value.requires_grad,
                    "dtype": str(value.dtype), "device": str(value.device),
                    "sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
        prefix.append({name: tensor(getattr(camera, name)) for name in fields})

    original = prefetch_parent.prefetch_camera_images
    before = inspect.getsource(original)
    after = before
    replacements = (
        ("                prepared = native_prepare(image, resolution)", "                prepared = _fault_prepare(native_prepare, image, resolution)"),
        ("                if expected_index != index+1:", "                _fault_observe(cameras[-1])\n                if expected_index != index+1:"),
    )
    for old, new in replacements:
        if after.count(old) != 1:
            raise ValueError("The real native preparation failure boundary changed")
        after = after.replace(old, new)
    archive = output.with_name(output.name + "_prepare_fault_sources")
    archive.mkdir(parents=True, exist_ok=False)
    physical = archive / "EXECUTED_FAULT_PREFETCH_FACTORY.py"
    physical.write_text(after, encoding="utf-8")
    (archive / "FAULT_FACTORY.diff").write_text("".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="original_prefetch", tofile="same_native_prepare17_fault")), encoding="utf-8")
    (archive / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    namespace = dict(inspect.unwrap(original).__globals__)
    namespace.update(_fault_prepare=prepare, _fault_observe=observe)
    exec(compile(after, str(physical), "exec"), namespace)
    prefetch_parent.prefetch_camera_images = namespace["prefetch_camera_images"]
    try:
        try:
            entry.main()
        except BaseException as error:
            if prefix != expected or not isinstance(error, OSError) or str(error) != "Forced original native image preparation failure at17":
                raise AssertionError("The real preparation error or complete camera prefix changed") from error
            atomic_json(output / "FAULT_PREFIX.json", {"status": "actual_native_prepare17_failure_prefix16_exact",
                "attempted_camera": 17, "target_camera_name": target, "first16_complete_camera_fields": prefix,
                "error_type": type(error).__name__, "error_message": str(error), "source": data_identity(Path(__file__)),
                "actual_factory_source": data_identity(physical), "new_optimization_count": 0})
            raise
        raise AssertionError("The explicitly faulted complete entry unexpectedly succeeded")
    finally:
        prefetch_parent.prefetch_camera_images = original


if __name__ == "__main__":
    main()
