"""Preserve the failed generated tuple and repair its closing syntax only."""

from pathlib import Path


def main():
    source = Path(__file__).with_name("check_deferred_checkpoint_equal_v2.py").read_text(encoding="utf-8")
    before = 'Path(__file__).with_name("check_deferred_checkpoint_equal_001.py")]'
    after = 'Path(__file__).with_name("check_deferred_checkpoint_equal_001.py"), Path(__file__).with_name("check_deferred_checkpoint_equal_v2.py"))]'
    if source.count(before) != 1:
        raise ValueError("The saved first generic wrapper source changed")
    source = source.replace(before, after)
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "owned_generic_v2_tuple_repair"}
    exec(compile(source, "<same_generic_v2_with_closed_source_identity_tuple>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
