"""E26 joint-pose, masked-MVS, and thin-surface reconstruction primitives.

The package is intentionally independent from files in ``outputs``.  Runtime
entry points must bind every external input through an immutable receipt.
"""

from .contracts import RigidTransformRecord
from .trajectory_io import Trajectory

__all__ = ["RigidTransformRecord", "Trajectory"]
