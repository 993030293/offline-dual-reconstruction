"""Keep the first failed text receipt and close the corrected current header."""

import inspect
from pathlib import Path

import close_coordinate_codec_text as preserved


def main():
    source = inspect.getsource(preserved.main)
    before = '"oup018/FINAL_TEXT_STATE_CLOSURE_001.json"'
    if source.count(before) != 1:
        raise ValueError("The preserved text closer output boundary changed")
    namespace = dict(preserved.main.__globals__)
    exec(compile(source.replace(before, '"oup018/FINAL_TEXT_STATE_CLOSURE_002.json"'), str(Path(__file__)), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
