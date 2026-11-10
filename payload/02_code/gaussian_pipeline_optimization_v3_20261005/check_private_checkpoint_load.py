"""Check native private-write isolation, aliases, explicit old-format routing and cleanup."""

import argparse
import io
import mmap
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from private_checkpoint_load import admission, private_checkpoint_load
from run_codec_campaign import exact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    paths = [Path(__file__), Path(__file__).with_name("private_checkpoint_load.py")]
    sources = [data_identity(p) for p in paths]
    atomic_json(args.output/"PROTOCOL.json", {"sources": sources, "original_CPU_labels_retained": True,
        "private_mutation_updates_alias_but_not_original_file": True,
        "old_format_stream_and_shared_flags_preselected_without_retry": True,
        "truncated_ZIP_exception_propagates_and_global_loader_restores": True, "new_optimization_count": 0})
    values = torch.arange(11*17, dtype=torch.float32).reshape(17, 11)
    state = {"base": values, "alias": values[:, ::2], "transposed": values.T,
        "empty": torch.empty((0, 19)), "bool": torch.arange(43)%2 == 0,
        "half": values.half(), "bfloat16": values.bfloat16(), "complex": torch.complex(values, values*.25)}
    native = args.output/"native.pth"
    torch.save(state, native)
    before = data_identity(native)
    baseline = torch.load(native, weights_only=False)
    original = torch.load
    with private_checkpoint_load(native) as trace:
        mapped = torch.load(native, weights_only=False)
        checks = {"all_heterogeneous_native_values_stride_and_type_exact": exact(baseline, mapped),
            "CPU_labels_unchanged": all(v.device.type == "cpu" for v in mapped.values()),
            "shared_storage_and_offset_preserved": mapped["base"].untyped_storage().data_ptr() == mapped["alias"].untyped_storage().data_ptr()
                == mapped["transposed"].untyped_storage().data_ptr()}
        mapped["base"][0, 0] = 123456.
        checks["private_alias_mutation_is_visible_in_own_views"] = float(mapped["alias"][0, 0]) == 123456. and float(mapped["transposed"][0, 0]) == 123456.
        checks["original_file_and_native_reload_remain_exact"] = data_identity(native) == before and exact(baseline, original(native, weights_only=False))
    checks["successful_context_restores_original_loader"] = trace["original_global_loader_restored"] and torch.load is original
    legacy = args.output/"legacy.pth"
    torch.save(state, legacy, _use_new_zipfile_serialization=False)
    checks["legacy_and_stream_preselect_original"] = not admission(legacy)[0] and not admission(io.BytesIO(native.read_bytes()))[0]
    with private_checkpoint_load(legacy, False) as legacy_trace:
        checks["explicit_legacy_native_path_preserves_payload"] = exact(baseline, torch.load(legacy, weights_only=False))
    checks["legacy_loader_restored"] = legacy_trace["original_global_loader_restored"]
    with torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
        checks["shared_setting_rejected_before_load"] = not admission(native)[0]
    checks["private_setting_restored"] = torch.serialization.get_default_mmap_options() == mmap.MAP_PRIVATE
    bad = args.output/"truncated.pth"
    bad.write_bytes(native.read_bytes()[:32])
    caught = False
    try:
        with private_checkpoint_load(bad):
            torch.load(bad, weights_only=False)
    except RuntimeError:
        caught = True
    checks["native_error_propagated_no_retry_and_loader_restored"] = caught and torch.load is original
    checks["source_and_original_native_bytes_unchanged"] = sources == [data_identity(p) for p in paths] and data_identity(native) == before
    result = {"status": "private_checkpoint_loader_contract_passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "sources": sources, "trace": trace, "new_optimization_count": 0,
        "new_object_training_quality_claim": False}
    atomic_json(args.output/"RESULT.json", result)
    print({k: result[k] for k in ("status", "checks")}, flush=True)
    if not all(checks.values()):
        raise ValueError("A private complete native state loading contract failed")


if __name__ == "__main__":
    main()
