"""Verify mapped native PLY readback, chunk boundaries, and malformed payload rejection."""

import argparse
from pathlib import Path

import numpy as np

from common import atomic_json
from package_delivery import validate_cloud
from ply_io import read_ply, write_vertices
from surface_tiled import header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    vertices, faces = read_ply(args.fixture)
    mapped_vertices, mapped_faces = read_ply(args.fixture, memory_map=True)
    if not np.array_equal(vertices, mapped_vertices) or not np.array_equal(faces, mapped_faces):
        raise ValueError('Mapped native fixture changed its vertex or face payload')
    repeated = np.tile(faces, 17)
    path = args.output / 'more_than_one_face_chunk.ply'
    with path.open('xb') as stream:
        header(stream, len(vertices), len(repeated))
        offset = stream.tell() + vertices.nbytes
        vertices.tofile(stream)
        repeated.tofile(stream)
    actual_vertices, actual_faces = read_ply(path, memory_map=True)
    if not np.array_equal(actual_faces, repeated):
        raise ValueError('Mapped mesh changed after the first validation chunk')
    with path.open('r+b') as stream:
        stream.seek(offset + 70000 * repeated.dtype.itemsize + 1)
        stream.write(np.asarray([len(vertices)], dtype='<i4').tobytes())
    try:
        read_ply(path, memory_map=True)
    except ValueError as error:
        if 'Out-of-range triangle indices' not in str(error):
            raise
    else:
        raise ValueError('A corrupt triangle in the second chunk was accepted')
    cloud = np.tile(vertices, 28)
    cloud['z'][70000] = np.nan
    cloud_path = args.output / 'nonfinite_second_vertex_chunk.ply'
    write_vertices(cloud_path, cloud)
    try:
        validate_cloud(cloud_path, len(cloud), memory_map=True)
    except ValueError as error:
        if 'Nonfinite point property: z' not in str(error):
            raise
    else:
        raise ValueError('A nonfinite coordinate in the second chunk was accepted')
    atomic_json(args.output / 'RESULT.json', {'status': 'mapped_native_payloads_and_cross_chunk_guards_verified',
        'fixture_vertices': len(vertices), 'fixture_triangles': len(faces),
        'corrupt_second_chunk_face_rejected': True, 'nonfinite_second_chunk_vertex_rejected': True})
    print('mapped_native_payloads_and_cross_chunk_guards_verified')


if __name__ == '__main__':
    main()
