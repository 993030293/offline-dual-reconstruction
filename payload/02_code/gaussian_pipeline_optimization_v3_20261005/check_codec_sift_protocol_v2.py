"""Keep the failed checker intact and select complete API-supported members."""

import inspect
from pathlib import Path

import check_codec_sift_protocol as preserved


def main():
    source = inspect.getsource(preserved.main)
    edits = [
        ('WHERE rows >= 2 ORDER BY image_id LIMIT 6', 'WHERE rows >= 2 AND rows <= 12000 ORDER BY image_id LIMIT 6'),
        ('    public = []', '    with sqlite3.connect("file:" + database.as_posix() + "?mode=ro", uri=True) as connection:\n        public_population = connection.execute("SELECT count(*),sum(rows<=12000),sum(rows>12000) FROM descriptors").fetchone()\n    public = []'),
        ('"sources": source_ids, "complete_component_seconds"', '"public_API_population_total_supported_unsupported": public_population, "public_selected_full_rows": [row[1] for row in selected], "sources": source_ids, "complete_component_seconds"'),
        ('"integer_sift_matching.py", "metric_branch_localization_003.py", "fused_owned_matching_worker.py")]',
         '"integer_sift_matching.py", "metric_branch_localization_003.py", "fused_owned_matching_worker.py", "check_codec_sift_protocol.py")]'),
    ]
    for before, after in edits:
        if source.count(before) != 1:
            raise ValueError("The preserved checker source boundary changed: " + before[:80])
        source = source.replace(before, after)
    namespace = dict(preserved.main.__globals__)
    namespace.update(OUT=preserved.OUT.with_name("g1"), __file__=str(Path(__file__).resolve()))
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
