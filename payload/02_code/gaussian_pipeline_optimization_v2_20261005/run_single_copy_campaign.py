"""Keep the original three-pair complete raw-entry protocol for direct payload stream reads."""

import inspect
from pathlib import Path
import textwrap

import run_raw_runtime_campaign as native


def main():
    source = textwrap.dedent(inspect.getsource(native.main))
    source = source.replace('choices=["columnar", "mapped"]', 'choices=["single-copy"]')
    source = source.replace('Path(__file__).with_name("run_initial_with_runtime_choices.py")', 'Path(__file__).with_name("run_initial_single_copy.py")')
    source = source.replace('Path(__file__).with_name("columnar_quality_scores.py")', 'Path(__file__).with_name("single_copy_raw_reader.py")')
    source = source.replace('Path(__file__).with_name("windowed_raw_reader.py")', 'Path(__file__).with_name("run_raw_runtime_campaign.py")')
    before = '"--reader-mode", "mapped" if mode == "candidate" and args.candidate == "mapped" else "indexed"'
    if source.count(before) != 1:
        raise ValueError("The complete raw campaign source and reader interface changed")
    source = source.replace(before, '"--single-copy", "true" if mode == "candidate" else "false"')
    namespace = dict(native.main.__globals__)
    namespace["__file__"] = str(Path(__file__))
    exec(compile(source, "<three_actual_pairs_of_single_copy_raw_stream_input>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
