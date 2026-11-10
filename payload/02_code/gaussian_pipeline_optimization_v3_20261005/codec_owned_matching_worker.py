"""Retain the existing observer for the sole registered lossless input entry."""

import inspect
from pathlib import Path

import fused_owned_matching_worker as preserved


def main():
    source = inspect.getsource(preserved.main)
    edits = [('fused_integer_match_worker.py', 'sift_payload_input_worker.py'),
             ('32*1024**3', '16*1024**3'), ('1750', '900')]
    for before, after in edits:
        if source.count(before) != (2 if before == '1750' else 1):
            raise ValueError("The preserved native observer bound changed: " + before)
        source = source.replace(before, after)
    namespace = dict(preserved.main.__globals__)
    namespace['__file__'] = str(Path(__file__).resolve())
    exec(compile(source, str(Path(__file__)), 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
