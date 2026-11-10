"""Serialize full float32 Gaussian attributes without per-vertex Python tuples."""

from pathlib import Path
import time

import numpy as np


def validate_save_schedule(events, schedule, resumed_iteration):
    """Require exactly the saves owned by this continuation process."""
    if resumed_iteration < 0 or len(set(schedule)) != len(schedule):
        raise ValueError("Invalid resume iteration or duplicate save schedule")
    expected = sorted(step for step in schedule if step > resumed_iteration)
    observed = []
    for event in events or []:
        folder = Path(event["path"]).parent.name
        if not folder.startswith("iteration_") or not folder[10:].isdigit():
            raise ValueError("Serialization event has no explicit iteration directory")
        observed.append(int(folder[10:]))
    if observed != expected:
        raise ValueError("Actual serialization events differ from this process's remaining save schedule")
    return {"resumed_iteration": resumed_iteration, "full_schedule": list(schedule),
            "expected_process_iterations": expected, "observed_process_iterations": observed}


def structured_vertices(attributes, names):
    attributes = np.asarray(attributes)
    if attributes.ndim != 2 or attributes.shape[1] != len(names):
        raise ValueError("Attribute matrix and names must have identical column counts")
    if attributes.dtype != np.dtype(np.float32) or not attributes.dtype.isnative:
        raise ValueError("This adapter supports native float32 Gaussian attributes only")
    if len(set(names)) != len(names) or not all(isinstance(name, str) for name in names):
        raise ValueError("PLY attributes require unique string names")
    dtype = np.dtype([(name, np.float32) for name in names], align=False)
    if dtype.itemsize != len(names)*4:
        raise ValueError("Packed PLY attributes cannot contain alignment padding")
    matrix = np.ascontiguousarray(attributes)
    return matrix.view(dtype).reshape(-1)


def vectorized_ply_model(base_class, verify_native=False):
    class VectorizedPlyModel(base_class):
        def save_ply(self, path):
            from plyfile import PlyData, PlyElement
            import torch

            started = time.monotonic()
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            xyz = self._xyz.detach().cpu().numpy()
            blocks = [xyz, np.zeros_like(xyz)]
            for value in (self._features_dc, self._features_rest):
                blocks.append(value.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy())
            blocks.extend(value.detach().cpu().numpy() for value in (self._opacity, self._scaling, self._rotation))
            matrix = np.concatenate(blocks, axis=1)
            elements = structured_vertices(matrix, self.construct_list_of_attributes())
            PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))
            torch.cuda.synchronize()
            event = {"path": str(path), "vertices": len(elements), "columns": matrix.shape[1],
                     "serialization_seconds": time.monotonic()-started, "native_byte_oracle": None}
            if verify_native:
                from runtime import data_identity
                oracle = path.with_name("native_oracle.ply")
                if oracle.exists():
                    raise FileExistsError("Never overwrite an existing serialization oracle")
                base_class.save_ply(self, str(oracle))
                actual, expected = data_identity(path), data_identity(oracle)
                event["native_byte_oracle"] = {"candidate": actual, "reference": expected,
                                               "identical": actual["sha256"] == expected["sha256"]}
                if not event["native_byte_oracle"]["identical"]:
                    raise AssertionError("Vectorized serialization differs from the native file bytes")
            if not hasattr(self, "ply_serialization_events"):
                self.ply_serialization_events = []
            self.ply_serialization_events.append(event)

    return VectorizedPlyModel
