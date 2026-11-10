"""Check bounded ordering, producer fault positions, consumer close and restoration."""

import argparse
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from query_preparation_pipeline import OrderedQueryPreparation, query_preparation_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for count in (0, 1, 2, 3, 79):
        for parallel in (False, True):
            producer = OrderedQueryPreparation(parallel)
            inputs = [{"name": f"arbitrary_{i}", "values": [i, -i, i*i]} for i in range(count)]
            outputs = list(producer.iterate(inputs, lambda source: {**source, "values": source["values"].copy()}))
            rows.append({"count": count, "parallel": parallel, "outputs_exact_and_independent": outputs == inputs
                and all(a is not b and a["values"] is not b["values"] for a, b in zip(inputs, outputs)),
                "all_records_once_and_capacity": producer.trace["submitted"] == producer.trace["completed"]
                    == producer.trace["delivered"] == count and producer.trace["peak_pending"] <= 2,
                "closed": producer.closed,
                "one_producer_thread": len({r["thread"] for r in producer.trace["preparations"]}) <= 1})
    checks = {"all10_empty_single_varied_sequences_exact_once_and_single_producer": len(rows) == 10
        and all(all(row[key] for key in ("outputs_exact_and_independent", "all_records_once_and_capacity", "closed", "one_producer_thread")) for row in rows)}
    producer = OrderedQueryPreparation(True)
    outputs, invoked = [], []
    def fault(source):
        invoked.append(source)
        if source == 2:
            raise ValueError("original_query_preparation_failure_2")
        return source
    try:
        outputs.extend(producer.iterate(range(17), fault))
    except ValueError as error:
        checks["producer_error_at_original_consumption_position_and_no_following_preparation"] = (
            str(error) == "original_query_preparation_failure_2" and outputs == [0, 1] and invoked == [0, 1, 2]
            and producer.closed)
    else:
        checks["producer_error_at_original_consumption_position_and_no_following_preparation"] = False
    for name in ("consumer_fault", "early_close"):
        producer = OrderedQueryPreparation(True)
        iterator = producer.iterate(range(19), lambda value: value)
        first = next(iterator)
        try:
            if name == "consumer_fault":
                raise RuntimeError("owned_consumer_failure")
        except RuntimeError as error:
            checks["consumer_failure_preserved"] = str(error) == "owned_consumer_failure"
        finally:
            iterator.close()
        checks[name+"_cancels_waits_and_closes_before_outer_caches"] = first == 0 and producer.closed and not producer.pending
    import localize_new_map_evaluation as native
    original = native.main
    for mode in ("incumbent", "serial", "parallel"):
        try:
            with query_preparation_pipeline(mode) as trace:
                if native.main is original or trace["composition_seconds"] is None:
                    raise AssertionError("The actual native loop was not composed")
                raise RuntimeError("owned_scope_failure")
        except RuntimeError as error:
            checks[mode+"_native_source_composition_and_error_restoration"] = str(error) == "owned_scope_failure" and native.main is original
    checks["all_owned_producer_threads_closed"] = not any(t.name.startswith("owned-query-reference") for t in threading.enumerate())
    result = {"status": "passed" if all(checks.values()) else "failed_verification", "checks": checks, "rows": rows,
        "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("query_preparation_pipeline.py")),
            data_identity(Path(native.__file__))], "counted": False, "actual_images_and_current_pose_entry_pending": True}
    atomic_json(args.output/"RESULT.json", result)
    print({key: result[key] for key in ("status", "checks")}, flush=True)
    if not all(checks.values()):
        raise AssertionError("An original query preparation ordering or ownership guard failed")


if __name__ == "__main__":
    main()
