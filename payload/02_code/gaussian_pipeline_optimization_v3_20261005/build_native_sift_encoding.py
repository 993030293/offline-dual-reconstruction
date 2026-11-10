"""Compile one owned strict encoder using the verified existing MSVC paths."""

import inspect
from pathlib import Path

import build_native_vector_gray_stats as original


def main():
    source = inspect.getsource(original.main)
    edits = [("vector_gray_stats_001", "oup019"), ("native_vector_gray_stats", "native_sift_encoding"),
             ('"/fp:precise"', '"/fp:strict"')]
    for before, after in edits:
        expected = 1 if before == "vector_gray_stats_001" or before == '"/fp:precise"' else 4
        if source.count(before) != expected:
            raise ValueError("The existing compiler source boundary changed: " + before)
        source = source.replace(before, after)
    namespace = dict(original.main.__globals__)
    namespace.update(__file__=str(Path(__file__).resolve()))
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
