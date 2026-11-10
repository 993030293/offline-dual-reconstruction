"""Check owned cache lifetimes and existing public image format compatibility."""

from types import SimpleNamespace
import gc
import pickle
from pathlib import Path
import random
import shutil
import sys
import time
import weakref

import numpy as np

from fused_camera_digest import original_digest, ROOT
from fused_camera_fingerprint import ImageFingerprintCache
from profile_scene_transform_cost import BoundedSceneGuard
from runtime import activate_reference, atomic_json, data_identity

CODE = Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001"


def main():
    output = OUT / "v0_interfaces"
    output.mkdir(exist_ok=False)
    sources = [data_identity(path) for path in (Path(__file__), CODE / "fused_camera_fingerprint.py", CODE / "fused_camera_digest.py")]
    with BoundedSceneGuard(output, atomic_json):
        activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import torch
        from PIL import Image
        from utils.general_utils import PILtoTorch
        library = OUT / "g0/build/fused_camera_digest.so"

        def make():
            return torch.arange(3*5*7, dtype=torch.float32).reshape(5, 7, 3).permute(2, 0, 1)

        def cache(values):
            trace = {}
            value = ImageFingerprintCache(library, trace)
            value.prepare([SimpleNamespace(original_image=image) for image in values])
            return value, trace

        def error(function, expected):
            try:
                function()
            except expected:
                return True
            return False

        a, b = make(), make()
        before = (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone())
        owned, trace = cache([a, b, a])
        exact = [owned.take(value) for value in (a, b, a)] == [original_digest(value) for value in (a, b, a)]
        repeated = error(lambda: owned.take(a), RuntimeError)
        owned.close()
        mutated, mutation_trace = cache([a])
        a.add_(1.)
        mutation_error = error(lambda: mutated.take(a), RuntimeError)
        mutated.close(aborted=True)
        fresh, fresh_trace = cache([a])
        fresh_exact = fresh.take(a) == original_digest(a)
        fresh.close()
        partial, partial_trace = cache([a, b])
        partial.take(a)
        partial_error = error(partial.close, RuntimeError)
        partial.close(aborted=True)
        second, second_trace = cache([b])
        twice_error = error(lambda: second.prepare([SimpleNamespace(original_image=b)]), RuntimeError)
        unowned_error = error(lambda: second.take(make()), RuntimeError)
        second.close(aborted=True)
        temporary = make()
        weak = weakref.ref(temporary)
        abandoned, abandoned_trace = cache([temporary])
        del temporary
        gc.collect()
        weak_closed = weak() is None
        abandoned.close(aborted=True)
        unsupported = error(lambda: cache([torch.zeros((3, 5, 7), dtype=torch.float64)]), ValueError)
        root = ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/images"
        public_files = sorted(path for path in root.iterdir() if path.is_file())
        if len(public_files) != 360:
            raise ValueError("The original complete public image pool denominator changed")
        selected = public_files[:32]
        identities = [data_identity(path) for path in selected]
        public = []
        for path in selected:
            with Image.open(path) as image:
                public.append(PILtoTorch(image, image.size))
        external, external_trace = cache(public)
        actual = [external.take(value) for value in public]
        expected = [original_digest(value) for value in public]
        external.close()
        checks = {
            "owned_duplicate_consumption_order_full_digest_exact": exact and trace["prepared"] == trace["consumed"] == 3,
            "already_consumed_binding_explicit_error": repeated,
            "tracked_mutation_never_reuses_stale_digest": mutation_error,
            "fresh_after_mutation_reads_current_full_bytes": fresh_exact,
            "partial_or_repeated_preparation_explicit_errors": partial_error and twice_error,
            "unowned_binding_explicit_error": unowned_error,
            "weak_prepared_bindings_do_not_retain_dead_tensor": weak_closed,
            "unsupported_format_explicit_admission_error": unsupported,
            "all_success_and_abort_caches_closed": all(item["cache_closed"] and item["active_entries"] == 0 for item in (trace, mutation_trace, fresh_trace, partial_trace, second_trace, abandoned_trace, external_trace)),
            "original_public360_pool_and32_full_size_images": len(public_files) == 360 and len(public) == 32,
            "all32_existing_public_complete_pixel_digests_exact": actual == expected and external_trace["prepared"] == external_trace["consumed"] == 32,
            "all_selected_public_inputs_and_own_sources_unchanged": identities == [data_identity(path) for path in selected]
                and sources == [data_identity(row["path"]) for row in sources],
            "Python_numpy_torch_CPU_RNG_unchanged": before[0] == pickle.dumps(random.getstate())
                and before[1] == pickle.dumps(np.random.get_state()) and torch.equal(before[2], torch.get_rng_state()),
        }
        atomic_json(output / "RESULT.json", {"status": "owned_digest_cache_interfaces_and32_public_full_pixel_exact0" if all(checks.values()) else "failed_verification",
            "checks": checks, "traces": {"owned": trace, "mutation": mutation_trace, "fresh": fresh_trace, "partial": partial_trace,
                "second": second_trace, "abandoned": abandoned_trace, "public": external_trace},
            "sources": sources, "public_inputs": identities, "all_public_pool_count": 360, "public_full_size_checked": 32,
            "public_already_used_development_not_GT_blind_or_whole_generalization": True,
            "untracked_external_pointer_writes_outside_owned_immutable_scope": True,
            "full_pipeline_interruption_and_recovery": "pending_separate_task", "new_optimization_count": 0})
        if not all(checks.values()):
            raise AssertionError("An owned cache lifecycle or public format check failed")
        print("Owned cache and existing public image interfaces passed: 13 checks, new 0")


if __name__ == "__main__":
    main()
