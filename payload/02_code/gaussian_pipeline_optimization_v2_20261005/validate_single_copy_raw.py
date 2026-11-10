"""Apply the complete native payload and boundary oracle to single-copy stream reading."""

import inspect
from pathlib import Path
import textwrap

import validate_mapped_raw as native
from single_copy_raw_reader import iter_single_copy_messages


def main():
    source = textwrap.dedent(inspect.getsource(native.main))
    source = source.replace('Path(__file__).with_name("windowed_raw_reader.py")', 'Path(__file__).with_name("single_copy_raw_reader.py")')
    source = source.replace('"window_allocation_max_bytes": 64*1024**2+65536,', '"mapped_input": False, "single_copy_payload_bytes": True,')
    namespace = dict(native.main.__globals__)
    namespace.update(iter_mapped_messages=iter_single_copy_messages, __file__=str(Path(__file__)))
    exec(compile(source, "<native_all7338_payload_oracle_for_single_copy_reader>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
