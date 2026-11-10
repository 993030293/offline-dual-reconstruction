# S00-S08 runtime source excerpts

This directory contains only the source excerpts used by the two installed runtime profiles: the visual frontend, metric trajectory binding, and the raw LiDAR/IMU fallback.

The legacy S02 static visual review source is excluded. The legacy S07 admission that consumed the S02 agent receipt is also excluded and replaced by `code/runtime/admit_bootstrap_scene_machine.py`.

No S09 or later source snapshot belongs to this package. Duplicate visual-only geometry trees, unused bootstrap copies, and Python bytecode caches are excluded from the installer. The parameterized package entry points are in `code/runtime/`, `code/validated_fresh_runtime/`, and the root PowerShell scripts.

PGSR is an external WSL runtime dependency and is not vendored here. Its retained notice is `THIRD_PARTY_NOTICES/PGSR_LICENSE.md`.
