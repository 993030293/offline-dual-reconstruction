"""Bound the owned SSIM family and verify both operating-system process exits."""

import inspect
from pathlib import Path

import sift_process_node_003 as original

OUT = Path(__file__).resolve().parents[2] / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup020/R172"


def compose():
    source = inspect.getsource(original.main)
    changes = [
        ('"1750"','"350"'),('size > 30*1024**3','size > 12*1024**3'),
        ('time.perf_counter()-started > 1770','time.perf_counter()-started > 360'),
        ('process.wait(timeout=1780)','process.wait(timeout=370)'),
        ('record["complete_node_seconds"] > 1800','record["complete_node_seconds"] > 360'),
        ('    if args.output.exists():\n        raise FileExistsError(args.output)',
         '    if not args.output.is_dir() or (args.output / "measurement").exists() or (args.output / "pair/evaluation").exists():\n'
         '        raise FileExistsError("A complete clean prepared model pair is required")'),
        ('                size = sum(path.stat().st_size',
         '                for resource_path in args.output.rglob("WSL_RESOURCE.json"):\n'
         '                    resource = safe_json(resource_path)\n'
         '                    if resource.get("peak_tree_RSS_bytes", 0) > 8*1024**3:\n'
         '                        raise MemoryError("The actual Linux evaluator exceeded8GiB")\n'
         '                size = sum(path.stat().st_size')]
    for before,after in changes:
        if source.count(before)!=1:raise ValueError("The original controller source boundary changed: "+before)
        source=source.replace(before,after)
    drain = inspect.getsource(original.linux_identity_check)
    before='("parallel_a_003/owned_matching_worker.py",)'
    if drain.count(before)!=1:raise ValueError("The original owned drain contract changed")
    drain=drain.replace(before,'("evaluate_fused_ssim.py", "fault_fused_ssim.py")')
    namespace=dict(original.__dict__,OUT=OUT,__file__=str(Path(__file__).resolve()))
    exec(compile(drain+"\n"+source,str(Path(__file__)),"exec"),namespace)
    return namespace["main"]


if __name__=='__main__':compose()()
