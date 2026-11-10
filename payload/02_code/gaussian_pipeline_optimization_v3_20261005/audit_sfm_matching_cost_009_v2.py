"""Preserve current and historical differences without treating either as a candidate acceptance."""
from pathlib import Path
import sys

SOURCE = Path(__file__).with_name("audit_sfm_matching_cost_009.py")
body = SOURCE.read_text(encoding="utf-8")


def once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError("The diagnostic source boundary changed")
    return text.replace(before, after)


body = once(body, "sources = [data_identity(Path(__file__)),", "sources = [data_identity(Path(__file__)), data_identity(Path(" + repr(str(SOURCE)) + ")),")
body = once(body, '            expected = read(SOURCE / oracle)', '''            historical = read(SOURCE / oracle)
            expected = database_summary(SOURCE / relative, names)
            changed_fields = [key for key in sorted(set(historical) | set(expected)) if historical.get(key) != expected.get(key)]
            save(output, "HISTORICAL_" + ("FEATURES.json" if relative == "features.db" else "MATCHING.json"), historical)
            diff = {"changed_fields": changed_fields, "historical_full_summary_exact": not changed_fields,
                    "origin_of_changes_unconfirmed": bool(changed_fields),
                    "historical_values_not_a_current_acceptance_oracle": True,
                    "historical_counts": {key: historical.get(key) for key in ("image_count", "attempted_matching_pairs", "stored_two_view_rows", "geometric_pairs_at_least_15_inliers", "component_sizes")},
                    "current_counts": {key: expected.get(key) for key in ("image_count", "attempted_matching_pairs", "stored_two_view_rows", "geometric_pairs_at_least_15_inliers", "component_sizes")}}
            save(output, "DIFF_" + ("FEATURES.json" if relative == "features.db" else "MATCHING.json"), diff)''')
body = body.replace('"The current original full database summary differs from its frozen receipt"', '"The current readonly database output changed between fresh calls"')
body = once(body, '"database": relative, "complete_function_seconds_each": timed, "graph": graph,', '"database": relative, "complete_function_seconds_each": timed, "graph": graph, "historical_difference": diff,')
body = once(body, '"full_original_both_databases_and_all_fields_exact"', '"current_original_all_fields_repeat_exact_historical_differences_preserved"')
body = once(body, '"complete_original_readonly_summary_cost_diagnostic0"', '"complete_original_readonly_summary_cost_with_historical_drift_unconfirmed0"')
body = once(body, '    (output / "sources").mkdir()', '    (output / "sources").mkdir()\n    (output / "sources/EXECUTED_DIAGNOSTIC.py").write_text(EXECUTED_BODY, encoding="utf-8")')
namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "__main__", "EXECUTED_BODY": body}
exec(compile(body, str(Path(__file__).resolve()), "exec"), namespace)
