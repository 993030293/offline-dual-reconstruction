"""Count successful native convolution outputs without changing their operation."""

from contextlib import contextmanager
import time


@contextmanager
def native_conv_work():
    import torch.nn.functional as functional

    original = functional.conv2d
    trace = {"rows": [], "successful_CUDA_output_elements": 0,
             "successful_CUDA_output_bytes": 0, "failed_calls": 0}

    def measured(*args, **kwargs):
        row = {"ordinal": len(trace["rows"]), "completed": False}
        before = time.perf_counter()
        try:
            output = original(*args, **kwargs)
            row.update(completed=True, shape=list(output.shape), dtype=str(output.dtype),
                       device=str(output.device), output_elements=output.numel(),
                       output_bytes=output.numel() * output.element_size())
            if output.device.type == "cuda":
                trace["successful_CUDA_output_elements"] += row["output_elements"]
                trace["successful_CUDA_output_bytes"] += row["output_bytes"]
            return output
        except BaseException as error:
            trace["failed_calls"] += 1
            row.update(error_type=type(error).__name__, error_message=str(error))
            raise
        finally:
            row["API_seconds_without_added_sync"] = time.perf_counter() - before
            trace["rows"].append(row)

    functional.conv2d = measured
    try:
        yield trace
    finally:
        functional.conv2d = original
        trace["original_binding_restored"] = functional.conv2d is original

