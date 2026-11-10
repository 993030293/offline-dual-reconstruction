"""Read the actual coordinate comparison schema while retaining the failed audit."""

import inspect
from pathlib import Path

import review_coordinate_and_codec_negative as preserved


def main():
    source = inspect.getsource(preserved.main)
    edits = [
        ('all(row["checks"].values()) for row in c["pairs"]',
         'all(row["comparison"]["checks"].values()) for row in c["pairs"]'),
        ('path.name == "NEGATIVE_REVIEW_LEASE.json"', 'path.name == "NEGATIVE_REVIEW_V2_LEASE.json"'),
        ('shutil.copyfile(Path(__file__), OUT / "REVIEW_SOURCE.py")',
         'shutil.copyfile(Path(__file__), OUT / "REVIEW_SOURCE.py")\n'
         '    shutil.copyfile(PRESERVED_SOURCE, OUT / "PRESERVED_REVIEW_SOURCE.py")'),
        ('"checks": checks, "review_is_separate_root_self_review_not_independent": True,',
         '"checks": checks, "review_is_separate_root_self_review_not_independent": True,\n'
         '              "reviewer_sources": [data_identity(Path(__file__)), data_identity(PRESERVED_SOURCE)],'),
    ]
    for before, after in edits:
        if source.count(before) != 1:
            raise ValueError("The preserved root reviewer boundary changed")
        source = source.replace(before, after)
    namespace = dict(preserved.main.__globals__)
    namespace.update(OUT=preserved.OUT.with_name("negative_review1"),
                     __file__=str(Path(__file__).resolve()), PRESERVED_SOURCE=Path(preserved.__file__))
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
