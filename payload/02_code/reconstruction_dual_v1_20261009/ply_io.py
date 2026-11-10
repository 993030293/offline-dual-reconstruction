"""Read the bounded numeric PLY layouts produced by native Gaussian and Open3D writers."""

from pathlib import Path
import numpy as np

TYPES = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
         "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1", "short": "i2",
         "ushort": "u2", "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4"}


def read_ply(path, memory_map=False):
    with Path(path).open("rb") as stream:
        lines = []
        for _ in range(512):
            line = stream.readline(512)
            if not line.endswith(b"\n"):
                raise ValueError("Incomplete or unbounded PLY header")
            lines.append(line.decode("ascii").strip())
            if lines[-1] == "end_header":
                break
        if lines[:2] != ["ply", "format binary_little_endian 1.0"] or lines[-1] != "end_header":
            raise ValueError("Only the declared binary native output layouts are supported")
        elements = []
        for line in lines[2:-1]:
            parts = line.split()
            if not parts or parts[0] in ("comment", "obj_info"):
                continue
            if parts[0] == "element":
                elements.append({"name": parts[1], "count": int(parts[2]), "properties": []})
            elif parts[0] == "property" and elements:
                elements[-1]["properties"].append(parts[1:])
            else:
                raise ValueError("Unsupported PLY header statement")
        if not elements or elements[0]["name"] != "vertex" or len(elements) > 2:
            raise ValueError("Unsupported native PLY element order")
        properties = elements[0]["properties"]
        if any(len(row) != 2 or row[0] not in TYPES for row in properties):
            raise ValueError("Unsupported native vertex property")
        dtype = np.dtype([(row[1], "<" + TYPES[row[0]]) for row in properties])
        def payload(layout, count):
            if count < 0:
                raise ValueError("Negative PLY element count")
            if not memory_map or not count:
                return np.fromfile(stream, dtype=layout, count=count)
            offset = stream.tell()
            if offset + count * layout.itemsize > Path(path).stat().st_size:
                raise ValueError("Truncated mapped PLY payload")
            result = np.memmap(path, dtype=layout, mode="r", offset=offset, shape=(count,))
            stream.seek(count * layout.itemsize, 1)
            return result

        vertices = payload(dtype, elements[0]["count"])
        if len(vertices) != elements[0]["count"]:
            raise ValueError("Truncated native vertex payload")
        faces = None
        if len(elements) == 2:
            face = elements[1]
            if face["name"] != "face" or len(face["properties"]) != 1:
                raise ValueError("Unsupported mesh face layout")
            property = face["properties"][0]
            if len(property) != 4 or property[:2] != ["list", "uchar"] or property[2] not in ("int", "uint"):
                raise ValueError("Unsupported triangle index layout")
            face_dtype = np.dtype([("count", "u1"), ("indices", "<"+TYPES[property[2]], (3,))])
            faces = payload(face_dtype, face["count"])
            if len(faces) != face["count"]:
                raise ValueError("Incomplete triangle mesh payload")
            for start in range(0, len(faces), 65536):
                chunk = faces[start:start + 65536]
                if np.any(chunk["count"] != 3):
                    raise ValueError("Incomplete triangle mesh payload")
                if chunk["indices"].min() < 0 or chunk["indices"].max() >= len(vertices):
                    raise ValueError("Out-of-range triangle indices")
        if stream.read(1):
            raise ValueError("Unexpected trailing native PLY payload")
    return vertices, faces


def write_vertices(path, vertices):
    reverse = {"f4": "float", "f8": "double", "u1": "uchar", "i4": "int", "u4": "uint"}
    header = ["ply", "format binary_little_endian 1.0", "element vertex " + str(len(vertices))]
    for name in vertices.dtype.names:
        field = vertices.dtype[name]
        kind = field.kind + str(field.itemsize)
        if kind not in reverse or field.subdtype:
            raise ValueError("Unsupported initial cloud property")
        header.append("property " + reverse[kind] + " " + name)
    header.append("end_header")
    with Path(path).open("xb") as stream:
        stream.write(("\n".join(header) + "\n").encode("ascii"))
        vertices.astype(vertices.dtype.newbyteorder("<"), copy=False).tofile(stream)
