"""Mutate only detached private storage without changing saved gradient metadata."""

from pathlib import Path


source = Path(__file__).with_name("fault_deferred_restore_hook_001.py").read_text(encoding="utf-8")
before = 'independent["gaussian_capture"][1].reshape(-1)[0].add_(1.0)'
if source.count(before) != 1:
    raise ValueError("The retained first fault injector source changed")
source = source.replace(before, 'independent["gaussian_capture"][1].detach().reshape(-1)[0].add_(1.0)')
exec(compile(source, "<private_storage_fault_without_gradient_metadata_mutation>", "exec"), globals())
