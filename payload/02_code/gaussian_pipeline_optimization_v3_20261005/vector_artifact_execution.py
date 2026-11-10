"""Adapt owned R140 workers without changing their original parent scheduling."""

from contextlib import contextmanager
import difflib
import inspect
from pathlib import Path
import textwrap

import export_qa_dag
from profile_vector_full_export import BASE, LIBRARY


@contextmanager
def vector_artifact_execution(enabled, proof):
    proof = Path(proof).resolve()
    proof.relative_to(BASE)
    proof.mkdir(parents=True, exist_ok=False)
    original = export_qa_dag.OwnedArtifacts
    before = textwrap.dedent(inspect.getsource(original.__init__))
    source = before
    if source.count('code / "export_qa_artifact_worker.py"') != 2 or source.count('contract = {') != 1:
        raise ValueError("The exact original worker source and command anchors changed")
    source = source.replace('code / "export_qa_artifact_worker.py"', 'code / "vector_export_qa_artifact_worker.py"')
    anchor = 'sources = [code / "vector_export_qa_artifact_worker.py",'
    if source.count(anchor) != 1: raise ValueError("The original worker source list changed")
    source = source.replace(anchor, 'sources = [code / "vector_export_qa_artifact_worker.py", code / "export_qa_artifact_worker.py", '
        'code / "vector_artifact_execution.py", code / "profile_vector_full_export.py", code / "dataset_sampling_shared_carrier.py", code / "dataset_sampling_carrier.py", '
        'code / "complete_carrier_configuration.py", code / "vector_camera_sampling.py", code / "native_vector_camera_sampling.cpp", '
        'code / "native_camera_sampling.py", code / "camera_pixel_remap.py", _r123_library,')
    source = source.replace('contract = {', 'contract = {"r123_sampling_enabled": _r123_enabled, "r123_library": str(_r123_library),')
    (proof / "COMPLETE_EXECUTED_OWNED_INIT.py").write_text(source, encoding="utf-8")
    (proof / "OWNED_INIT_ADAPTER.diff").write_text("".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True),
        fromfile="original_OwnedArtifacts_init", tofile="owned_vector_sampling_contract")), encoding="utf-8")
    namespace = dict(original.__init__.__globals__)
    namespace.update(_r123_enabled=bool(enabled), _r123_library=LIBRARY)
    exec(compile(source, "<original_R140_owned_worker_vector_contract>", "exec"), namespace)

    class OwnedVectorArtifacts(original):
        __init__ = namespace["__init__"]

    export_qa_dag.OwnedArtifacts = OwnedVectorArtifacts
    try:
        yield
    finally:
        export_qa_dag.OwnedArtifacts = original
        if export_qa_dag.OwnedArtifacts is not original:
            raise AssertionError("The original owned-worker class was not restored")
