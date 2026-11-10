"""Mark possible CUDA work before dispatch so an early interrupt drains it."""

from pathlib import Path


source = Path(__file__).with_name("deferred_checkpoint_equal_001.py").read_text(encoding="utf-8")
replacements = [
    ('"CUDA_comparison_leaves": 0,', '"CUDA_comparison_leaves": 0, "CUDA_work_may_be_issued": False,'),
    ('    def native_equal(left, right):\n        result = original_tensor_equal(left, right)',
     '    def native_equal(left, right):\n        if isinstance(left, torch.Tensor) and left.is_cuda:\n            trace["CUDA_work_may_be_issued"] = True\n        result = original_tensor_equal(left, right)'),
    ('                pending.append(torch.eq(first, second).all())',
     '                trace["CUDA_work_may_be_issued"] = True\n                pending.append(torch.eq(first, second).all())'),
    ('        if trace["CUDA_comparison_leaves"]:', '        if trace["CUDA_work_may_be_issued"]:'),
]
for before, after in replacements:
    if source.count(before) != 1:
        raise ValueError("The frozen first equality mechanism changed")
    source = source.replace(before, after)
exec(compile(source, "<same_equality_with_pre_dispatch_CUDA_lifecycle_mark>", "exec"), globals())
