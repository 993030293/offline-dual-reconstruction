"""Read camera metadata without allocating unused COLMAP observation arrays for training."""

from contextlib import contextmanager
import inspect
from pathlib import Path
import struct
import time

import numpy as np

IMAGE_HEADER = struct.Struct("<idddddddi")
COUNT = struct.Struct("<Q")


def metadata_images(path, image_class):
    path = Path(path)
    size = path.stat().st_size
    if size < 8 or size > 512*1024**2:
        raise ValueError("Training metadata file exceeds its declared safe size")
    images, observation_count = {}, 0
    with path.open("rb") as stream:
        count = COUNT.unpack(stream.read(COUNT.size))[0]
        if count == 0 or count > 1000000 or count*73+8 > size:
            raise ValueError("The binary camera count cannot fit its file")
        for _ in range(count):
            raw = stream.read(IMAGE_HEADER.size)
            if len(raw) != IMAGE_HEADER.size:
                raise ValueError("A camera metadata header is truncated")
            values = IMAGE_HEADER.unpack(raw)
            image_id, camera_id = values[0], values[8]
            if image_id <= 0 or camera_id <= 0 or image_id in images:
                raise ValueError("A camera metadata identity is invalid or duplicated")
            filename = bytearray()
            while True:
                char = stream.read(1)
                if char == b"\0":
                    break
                if not char or len(filename) >= 4096:
                    raise ValueError("The binary camera filename is missing its terminator")
                filename.extend(char)
            name = filename.decode("utf-8")
            if not name or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("The binary camera filename is outside its dataset")
            raw_count = stream.read(COUNT.size)
            if len(raw_count) != COUNT.size:
                raise ValueError("The binary observation count is truncated")
            rows = COUNT.unpack(raw_count)[0]
            byte_count = 24*rows
            if byte_count > size-stream.tell():
                raise ValueError("The declared observation records are truncated")
            stream.seek(byte_count, 1)
            observation_count += rows
            # These arrays are deliberately absent in this training-only view.
            images[image_id] = image_class(id=image_id, qvec=np.asarray(values[1:5]),
                tvec=np.asarray(values[5:8]), camera_id=camera_id, name=name, xys=None, point3D_ids=None)
        if stream.tell() != size:
            raise ValueError("Trailing binary image data has no declared metadata record")
    return images, {"image_count": len(images), "observation_rows_not_allocated": observation_count,
        "source_bytes": size, "observation_bytes_skipped": 24*observation_count,
        "observations_deleted_from_source": False, "only_training_metadata_view": True}


@contextmanager
def training_metadata_reader(enabled=True):
    import scene.dataset_readers as readers
    import scene.colmap_loader as colmap

    native = readers.read_extrinsics_binary
    native_text = readers.read_extrinsics_text
    consumer = inspect.getsource(readers.readColmapCameras)
    if any(token in consumer for token in (".xys", ".point3D_ids", ".points2D")):
        raise ValueError("The camera consumer now requires observations; metadata omission is unsafe")
    records = []

    def read(path):
        started = time.perf_counter()
        if enabled:
            images, receipt = metadata_images(path, colmap.Image)
        else:
            images = native(path)
            receipt = {"image_count": len(images), "observation_rows_not_allocated": 0,
                "observation_array_bytes": sum(r.xys.nbytes+r.point3D_ids.nbytes for r in images.values()),
                "observation_rows_loaded": sum(len(r.xys) for r in images.values()),
                "source_bytes": Path(path).stat().st_size, "only_training_metadata_view": False}
        records.append({**receipt, "seconds": time.perf_counter()-started})
        return images

    def reject_text(path):
        raise RuntimeError("The explicit binary metadata reader failed; a text fallback is forbidden")

    readers.read_extrinsics_binary = read
    if enabled:
        readers.read_extrinsics_text = reject_text
    try:
        yield records
    finally:
        readers.read_extrinsics_binary = native
        readers.read_extrinsics_text = native_text
