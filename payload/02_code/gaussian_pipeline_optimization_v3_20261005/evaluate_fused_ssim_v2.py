"""Bind the complete evaluator to the mixed-layout formula without other changes."""

from pathlib import Path
from types import FunctionType

import evaluate_fused_ssim as original

entry=original.entry


def composed_main():
    source=original.composed_main()
    return original.original.replace_once(source,
        'from fused_ssim_formula import FusedInteriorSSIM',
        'from fused_ssim_formula_v2 import FusedInteriorSSIM').replace(
            'CODE / "fused_ssim_formula.py",',
            'CODE / "fused_ssim_formula.py", CODE / "fused_ssim_formula_v2.py",')


namespace=dict(vars(original),composed_main=composed_main,__file__=str(Path(__file__)))
run=FunctionType(original.run.__code__,namespace,'run',original.run.__defaults__)


if __name__=='__main__':run()
