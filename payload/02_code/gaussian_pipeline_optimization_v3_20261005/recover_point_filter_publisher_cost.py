"""Recover only historical publisher command/result timing from this chat log."""

import argparse
import datetime
import json
import mmap
from pathlib import Path
import re
import time

import profile_quality_corner_cost as reference


def metadata(value, timings, exits):
    if isinstance(value, dict):
        if isinstance(value.get("wall_time_seconds"), (int, float)):
            timings.append(float(value["wall_time_seconds"]))
        if isinstance(value.get("exit_code"), int):
            exits.append(value["exit_code"])
        for item in value.values():
            if isinstance(item, (dict, list, str)):
                metadata(item, timings, exits)
    elif isinstance(value, list):
        for item in value:
            metadata(item, timings, exits)
    elif isinstance(value, str):
        try:
            nested = json.loads(value)
        except (ValueError, TypeError):
            timings.extend(float(item) for item in re.findall(r'"wall_time_seconds"\s*:\s*([0-9.]+)', value))
            exits.extend(int(item) for item in re.findall(r'"exit_code"\s*:\s*(-?[0-9]+)', value))
        else:
            if nested != value:
                metadata(nested, timings, exits)


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    output = parser.parse_args().output.resolve()
    output.relative_to(reference.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/pf_v5_001")
    output.mkdir(parents=True, exist_ok=False)
    (output / "EXECUTED_SOURCE.py").write_bytes(Path(__file__).read_bytes())
    source = Path("C:/Users/99303/.codex/sessions/2026/09/07/rollout-2026-09-07T10-37-39-01a079ba-7973-7780-bf60-a034f8b50aa3.jsonl")
    size = source.stat().st_size
    closure = reference.load(reference.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/pf_v4_001/V4_NEGATIVE_CLOSURE_001.json")
    instant = datetime.datetime.fromtimestamp(closure["publisher_creation_time"], datetime.timezone.utc)
    earliest = (instant - datetime.timedelta(seconds=60)).isoformat()[:19].encode()
    post_path = reference.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/pf_v4_001/FINAL_POST_EXIT_001.json"
    post_instant = datetime.datetime.fromtimestamp(post_path.stat().st_mtime, datetime.timezone.utc)
    latest = (max(instant + datetime.timedelta(seconds=120), post_instant + datetime.timedelta(seconds=60))).isoformat()[:19].encode()
    with source.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        lower, upper = 0, size
        while upper - lower > 1024 * 1024:
            middle = (lower + upper) // 2
            following = mapped.find(b"\n", middle) + 1
            header = re.search(rb'"timestamp"\s*:\s*"([0-9T:-]{19})', mapped[following:following+160])
            if not following or header is None:
                raise ValueError("The historical metadata timestamp index is unavailable")
            if header.group(1) < earliest:
                lower = following
            else:
                upper = middle
        begin = mapped.rfind(b"\n", 0, lower) + 1
        rows = []
        cursor, end = begin, begin
        while cursor < size:
            end = mapped.find(b"\n", cursor)
            if end < 0:
                end = size
            header = re.search(rb'"timestamp"\s*:\s*"([0-9T:-]{19})', mapped[cursor:cursor+160])
            if header is None:
                raise ValueError("The bounded historical row has no timestamp")
            if header.group(1) > latest:
                break
            if earliest <= header.group(1) <= latest and end-cursor <= 1024*1024:
                line = mapped[cursor:end]
                if b"close_point_filter_v4.py" in line or b"FINAL_POST_EXIT_001.json" in line or b"function_call_output" in line or b"custom_tool_call_output" in line:
                    rows.append(json.loads(line))
            cursor = end + 1
        if end-begin > 32*1024**2:
            raise MemoryError("The historical publisher metadata window exceeded its byte budget")
    calls, post_calls = [], []
    for row in rows:
        payload = row.get("payload", {})
        arguments = payload.get("arguments", payload.get("input", ""))
        commands = [json.loads(piece) for piece in re.findall(r'\bcmd\s*:\s*("(?:\\.|[^"\\])*")', arguments)] if isinstance(arguments, str) else []
        if payload.get("type") in ("function_call", "custom_tool_call") and any("close_point_filter_v4.py" in command and "-X utf8" in command for command in commands):
            calls.append({"call_id": payload["call_id"], "timestamp": row["timestamp"]})
        if payload.get("type") in ("function_call", "custom_tool_call") and isinstance(arguments, str) and "FINAL_POST_EXIT_001.json" in arguments and "pf_v4_001" in arguments and "tools.exec_command" in arguments and ("write_text" in arguments or "atomic_json" in arguments):
            post_calls.append({"call_id": payload["call_id"], "timestamp": row["timestamp"]})
    if len(calls) != 1:
        reference.atomic_json(output / "FAILURE.json", {"reason": "historical_exact_publisher_command_not_unique_in_bounded_window", "matched_command_ids": calls,
            "bounded_bytes": end-begin, "raw_log_not_copied": True, "node_seconds": time.perf_counter()-started})
        raise ValueError("The historical publisher timing could not be uniquely recovered")
    outputs = []
    for row in rows:
        payload = row.get("payload", {})
        if payload.get("call_id") != calls[0]["call_id"] or payload.get("type") not in ("function_call_output", "custom_tool_call_output"):
            continue
        elapsed, exit_codes = [], []
        metadata(payload, elapsed, exit_codes)
        outputs.append({"timestamp": row["timestamp"], "actual_command_wall_seconds": elapsed,
                        "exit_codes": exit_codes, "payload_fields": list(payload)})
    if len(outputs) != 1 or len(outputs[0]["actual_command_wall_seconds"]) != 1 or outputs[0]["exit_codes"] != [0]:
        reference.atomic_json(output / "FAILURE.json", {"reason": "historical_publisher_closed_result_not_unique", "command": calls[0], "outputs": outputs,
            "raw_log_not_copied": True, "node_seconds": time.perf_counter()-started})
        raise ValueError("The historical publisher actual closed command timing is unavailable")
    internal = reference.load(reference.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/pf_v4_001/V4_NEGATIVE_CLOSURE_001.json")["publication_internal_seconds"]
    actual = outputs[0]["actual_command_wall_seconds"][0]
    if actual < internal:
        raise ValueError("The closed publisher command omitted its internal measured time")
    post_outputs = []
    if len(post_calls) != 1:
        reference.atomic_json(output / "FAILURE.json", {"reason": "historical_post_exit_command_not_unique", "post_command_ids": post_calls, "raw_log_not_copied": True})
        raise ValueError("The old post-exit check could not be uniquely recovered")
    for row in rows:
        payload = row.get("payload", {})
        if payload.get("call_id") == post_calls[0]["call_id"] and payload.get("type") in ("function_call_output", "custom_tool_call_output"):
            timings, exits = [], []
            metadata(payload, timings, exits)
            post_outputs.append({"timestamp": row["timestamp"], "actual_command_wall_seconds": timings, "exit_codes": exits})
    if len(post_outputs) != 1 or not post_outputs[0]["actual_command_wall_seconds"] or len(post_outputs[0]["actual_command_wall_seconds"]) != len(post_outputs[0]["exit_codes"]) or any(post_outputs[0]["exit_codes"]):
        reference.atomic_json(output / "FAILURE.json", {"reason": "old_post_exit_closed_result_unavailable", "command": post_calls[0],
            "outputs": post_outputs, "raw_arguments_and_outputs_not_copied": True})
        raise ValueError("The old post-exit actual closed command result is unavailable")
    reference.atomic_json(output / "RESULT.json", {"status": "original_publisher_actual_closed_command_timing_recovered", "command": calls[0], "result": outputs[0],
        "historical_session_path": str(source), "bounded_window_bytes": end-begin, "raw_log_arguments_and_outputs_not_copied": True,
        "original_publisher_internal_seconds_already_in_V4": internal, "actual_publisher_complete_command_seconds": actual,
        "only_uncharged_publisher_envelope_seconds": actual-internal, "scope": "tool_command_wall_includes_host_process_startup_and_exit_not_text_research",
        "old_post_exit_command": post_calls[0], "old_post_exit_result": post_outputs[0],
        "old_post_exit_complete_command_seconds": sum(post_outputs[0]["actual_command_wall_seconds"]),
        "old_post_exit_all_closed_command_walls_summed_not_single_command_or_orchestrator_time": True,
        "source": reference.data_identity(Path(__file__)), "node_seconds": time.perf_counter()-started, "accepted_count_contribution": 0})
    print({"status": "publisher_cost_recovered", "actual_command_wall_seconds": actual, "additional_seconds": actual-internal}, flush=True)


if __name__ == "__main__":
    main()
