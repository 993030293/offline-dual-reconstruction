"""Restrict the original complete controller to the owned wire-codec family."""

import inspect
from pathlib import Path

import sift_process_node_003 as original

OUT = Path(__file__).resolve().parents[2] / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup018/R170"


def compose():
    source = inspect.getsource(original.main)
    changes = [
        ('"1750"', '"880"'),
        ('size > 30*1024**3', 'size > 12*1024**3'),
        ('time.perf_counter()-started > 1770', 'time.perf_counter()-started > 900'),
        ('process.wait(timeout=1780)', 'process.wait(timeout=910)'),
        ('record["complete_node_seconds"] > 1800', 'record["complete_node_seconds"] > 900'),
        ('                size = sum(path.stat().st_size',
         '                for resource_path in args.output.rglob("WSL_RESOURCE.json"):\n'
         '                    resource = safe_json(resource_path)\n'
         '                    if resource["peak_RSS_bytes"] > 16*1024**3:\n'
         '                        raise MemoryError("The actual Linux worker exceeded16GiB")\n'
         '                size = sum(path.stat().st_size'),
    ]
    for before, after in changes:
        if source.count(before) != 1:
            raise ValueError("The original complete controller boundary changed: " + before)
        source = source.replace(before, after)
    namespace = dict(original.__dict__)
    namespace.update(OUT=OUT, __file__=str(Path(__file__).resolve()))
    drain_source = inspect.getsource(original.linux_identity_check)
    before = '("parallel_a_003/owned_matching_worker.py",)'
    if drain_source.count(before) != 1:
        raise ValueError("The original owned Linux drain boundary changed")
    drain_source = drain_source.replace(before,
        '("parallel_a_003/owned_matching_worker.py", "codec_owned_matching_worker.py")')
    exec(compile(drain_source + "\n" + source, str(Path(__file__)), "exec"), namespace)
    return namespace["main"]


if __name__ == "__main__":
    compose()()
