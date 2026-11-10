"""Keep the complete evaluator protocol while preregistering memory as primary."""

import time

STARTED = time.perf_counter()

import inspect
import argparse
from pathlib import Path

import run_interior_ssim_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-name", choices=("c0", "c1", "c2"), default="c0")
    parser.add_argument("--generic-name", choices=("g1", "g2"), default="g1")
    args = parser.parse_args()
    source = inspect.getsource(run_interior_ssim_campaign.main)
    replacements = {
        'generic, extra = read(OUT / "g0/RESULT.json"), read(OUT / "g1/RESULT.json")':
            'generic = read(OUT / "g0/RESULT.json")\n    extra = generic',
        'extra["status"] != "generic_extra_passed"': 'extra["status"] != "generic_passed"',
        'CODE / "evaluate_interior_ssim.py"': 'CODE / "evaluate_view_major.py"',
        'CODE / "interior_ssim_reuse.py"': 'CODE / "view_major_evaluation.py"',
        'CODE / "mapped_image_linux_guard.py"]':
            'CODE / "mapped_image_linux_guard.py", CODE / "evaluate_interior_ssim.py", CODE / "run_interior_ssim_campaign.py"]',
        '"V2_complete_phase_reduction_minimum": .05': '"minimum_actual_CUDA_reserved_peak_reduction_primary": .2',
        '"first_pair_stop_on_no_raw_phase_benefit_or_whole_ratio_violation": True':
            '"first_pair_stop_on_less_than20percent_CUDA_peak_or_whole_violation": True',
        '"complete_whole_seconds": time.perf_counter() - before, "setup_seconds": setup_seconds,':
            '"complete_whole_seconds": time.perf_counter() - before, "setup_seconds": setup_seconds, "peak_CUDA_reserved_bytes": measurement["peak_CUDA_reserved_bytes"],',
        'if trial == 0 and trial_rows["interior"]["complete_phase_seconds"] >= trial_rows["native"]["complete_phase_seconds"]:\n                raise RuntimeError("The preregistered first complete pair has no raw phase benefit")':
            'if trial == 0 and trial_rows["interior"]["peak_CUDA_reserved_bytes"] > .8 * trial_rows["native"]["peak_CUDA_reserved_bytes"]:\n                raise RuntimeError("The preregistered first complete pair has insufficient CUDA peak reduction")',
        'for key in ("complete_phase_seconds", "complete_whole_seconds")':
            'for key in ("complete_phase_seconds", "complete_whole_seconds", "peak_CUDA_reserved_bytes")'
    }
    for anchor, replacement in replacements.items():
        expected_count = 2 if anchor == 'CODE / "evaluate_interior_ssim.py"' else 1
        if source.count(anchor) != expected_count:
            raise ValueError("The exact complete evaluation campaign source changed: " + anchor)
        source = source.replace(anchor, replacement)
    source = source.replace('"interior"', '"view"').replace('/interior/', '/view/')
    source = source.replace('output = OUT / "c0"', 'output = OUT / ' + repr(args.campaign_name))
    source = source.replace('generic = read(OUT / "g0/RESULT.json")', 'generic = read(OUT / ' + repr(args.generic_name + "/RESULT.json") + ')')
    source = source.replace('read(OUT / "g0/CONTRACT.json")', 'read(OUT / ' + repr(args.generic_name + "/CONTRACT.json") + ')')
    namespace = dict(vars(run_interior_ssim_campaign))
    output = namespace["ROOT"] / "04_outputs/gaussian_pipeline_optimization_v3_20261005/vm001"
    output.mkdir(parents=True, exist_ok=True)
    (output / "COMPOSED_CAMPAIGN.py").write_text(source, encoding="utf-8")
    namespace.update(__file__=str(Path(__file__)), OUT=output, STARTED=STARTED)
    exec(compile(source, "<private-current-full-view-major-campaign>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
