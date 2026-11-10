"""Compare every native Gaussian PLY byte without materializing reference tuples."""

import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np

NAMES = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
MAXIMUM_BYTES = 256 * 1024**2
ROWS_PER_BLOCK = 8192
MISMATCH = "Vectorized serialization differs from the native file bytes"


def admission(model, base_class):
    import torch
    if torch.__version__ != "2.7.1+cu128" or np.__version__ != "1.26.4" or sys.byteorder != "little":
        return False, "original_runtime_before_execution"
    if type(model).construct_list_of_attributes is not base_class.construct_list_of_attributes:
        return False, "custom_attribute_method_before_execution"
    count = len(model._xyz)
    values = [getattr(model, name) for name in NAMES]
    shapes = ((count, 3), (count, 1, 3), (count, model._features_rest.shape[1], 3),
              (count, 1), (count, 3), (count, 4))
    if any(value.dtype != torch.float32 or value.device.type != "cuda" or tuple(value.shape) != shape
           for value, shape in zip(values, shapes)):
        return False, "original_dtype_device_shapes_before_execution"
    columns = 17 + model._features_rest.shape[1] * 3
    if count * columns * 4 > MAXIMUM_BYTES:
        return False, "complete_parameter_budget_before_execution"
    if not all(bool(torch.isfinite(value).all()) for value in values):
        return False, "original_nonfinite_conversion_before_execution"
    return True, "native_finite_float32_gaussian_bytes"


def expected_blocks(model):
    xyz = model._xyz.detach().cpu().numpy()
    blocks = [xyz, np.zeros_like(xyz)]
    for value in (model._features_dc, model._features_rest):
        blocks.append(value.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy())
    blocks.extend(value.detach().cpu().numpy() for value in (model._opacity, model._scaling, model._rotation))
    return blocks


def expected_header(count, names):
    from plyfile import PlyData, PlyElement
    dtype = np.dtype([(name, "f4") for name in names], align=False)
    # describe() consumes array metadata; no full reference body is allocated.
    rows = np.broadcast_to(np.zeros(1, dtype=dtype), (count,))
    return (PlyData([PlyElement.describe(rows, "vertex")]).header + "\n").encode("ascii")


def verify_complete_bytes(model, path):
    started = time.perf_counter()
    blocks = expected_blocks(model)
    names = model.construct_list_of_attributes()
    if sum(value.shape[1] for value in blocks) != len(names):
        raise ValueError("Attribute matrix and names must have identical column counts")
    header = expected_header(len(blocks[0]), names)
    digest = hashlib.sha256(header)
    total, comparisons = len(header), 0
    with Path(path).open("rb") as stream:
        if stream.read(len(header)) != header:
            raise AssertionError(MISMATCH)
        for begin in range(0, len(blocks[0]), ROWS_PER_BLOCK):
            matrix = np.concatenate([value[begin:begin + ROWS_PER_BLOCK] for value in blocks], axis=1)
            expected = matrix.tobytes(order="C")
            if stream.read(len(expected)) != expected:
                raise AssertionError(MISMATCH)
            digest.update(expected)
            total += len(expected)
            comparisons += 1
        if stream.read(1):
            raise AssertionError(MISMATCH)
    return {"all_header_body_bytes_and_EOF_exact": True, "expected_native_sha256": digest.hexdigest(),
        "complete_bytes": total, "all_vertices": len(blocks[0]), "columns": len(names),
        "blocks": comparisons, "verification_seconds": time.perf_counter() - started}


def streamed_ply_model(base_class):
    from ply_serialization import vectorized_ply_model
    from runtime import data_identity
    native_verified = vectorized_ply_model(base_class, verify_native=True)
    writer = vectorized_ply_model(base_class, verify_native=False)

    class StreamedVerifiedModel(writer):
        def save_ply(self, path):
            supported, reason = admission(self, base_class)
            if not supported:
                return native_verified.save_ply(self, path)
            writer.save_ply(self, path)
            oracle = Path(path).with_name("native_oracle.ply")
            if oracle.exists():
                raise FileExistsError("Never overwrite an existing serialization oracle")
            try:
                proof = verify_complete_bytes(self, path)
            except BaseException:
                # Preserve the original reference artifact and verdict on a failed check.
                base_class.save_ply(self, str(oracle))
                actual, expected = data_identity(path), data_identity(oracle)
                if actual["sha256"] != expected["sha256"]:
                    raise AssertionError(MISMATCH)
                raise
            shutil.copyfile(path, oracle)
            actual, expected = data_identity(path), data_identity(oracle)
            if not (actual["sha256"] == expected["sha256"] == proof["expected_native_sha256"]):
                raise AssertionError(MISMATCH)
            event = self.ply_serialization_events[-1]
            event["native_byte_oracle"] = {"candidate": actual, "reference": expected, "identical": True}
            event["streamed_native_verification"] = {**proof, "route": reason,
                "independent_reference_copy": True, "original_writer_and_model_bytes_unchanged": True}

    return StreamedVerifiedModel
