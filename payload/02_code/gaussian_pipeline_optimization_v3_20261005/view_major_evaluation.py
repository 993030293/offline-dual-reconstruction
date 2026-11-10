"""Compose a view-major consumer without changing the original metric operations."""

import inspect


def compose_view_major(original):
    source = inspect.getsource(original)
    start = "    for label in [args.control_label, args.candidate_label]:\n"
    end = "    deltas = {region:"
    row_start = "        rows = []\n"
    view_start = '        for source in localization["rows"]:\n'
    summary_start = "        summary = {}\n"
    if any(source.count(anchor) != 1 for anchor in (start, end, row_start, view_start, summary_start)):
        raise ValueError("The exact two-model complete evaluator loop source changed")
    first, last = source.index(start), source.index(end)
    row_pos, view_pos, summary_pos = source.index(row_start), source.index(view_start), source.index(summary_start)
    if not first < row_pos < view_pos < summary_pos < last:
        raise ValueError("The complete evaluator dependency ordering changed")
    initialization = source[first:row_pos]
    view_body = source[view_pos + len(view_start):summary_pos]
    summary = source[summary_pos:last]
    delete = "        del gaussian\n"
    if summary.count(delete) != 1:
        raise ValueError("The original per-model release source changed")
    summary = summary.replace(delete, "        del models[label]\n" + delete)
    composed = (
        "    models = {}\n"
        "    collected = {label: [] for label in [args.control_label, args.candidate_label]}\n"
        + initialization
        + '        models[label] = {"gaussian": gaussian, "state": state, "model_path": model_path}\n'
        + '    for source in localization["rows"]:\n'
        + "        for label in [args.control_label, args.candidate_label]:\n"
        + '            gaussian = models[label]["gaussian"]\n'
        + "            rows = collected[label]\n"
        + view_body
        + "        _finish_view()\n"
        + start
        + '        gaussian = models[label]["gaussian"]\n'
        + '        state = models[label]["state"]\n'
        + '        model_path = models[label]["model_path"]\n'
        + "        rows = collected[label]\n"
        + summary
    )
    return source[:first] + composed + source[last:]


def finish_reference_view(cache, trace):
    if cache.resident_bytes < 0 or cache.resident_bytes > cache.maximum_bytes:
        raise ValueError("The original reference cache accounting is invalid")
    released = cache.resident_bytes
    entries = len(cache.entries)
    cache.entries.clear()
    cache.resident_bytes = 0
    trace.append({"completed_view_index": len(trace), "released_accounted_reference_bytes": released,
                  "released_reference_entries": entries, "retained_windows": len(cache.windows)})
    if cache.entries or cache.resident_bytes:
        raise AssertionError("The completed reference view remains retained")
