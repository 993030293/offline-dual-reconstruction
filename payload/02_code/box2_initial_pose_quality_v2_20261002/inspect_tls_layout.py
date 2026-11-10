"""Diagnose the exact binary PCD data boundary without loading the whole map."""

import json
import struct
from public_gt_audit import TLS

with TLS.open("rb") as stream:
    lines=[]
    while True:
        line=stream.readline(4096).decode("ascii").strip()
        lines.append(line)
        if line.startswith("DATA "):
            offset=stream.tell()
            break
    first=stream.read(64)
    count=int(next(line.split()[1] for line in lines if line.startswith("POINTS ")))
    expected_end=offset+count*16
    stream.seek(expected_end)
    tail=stream.read()
print(json.dumps({"path":str(TLS),"header":lines,"data_offset":offset,"file_bytes":TLS.stat().st_size,
    "declared_payload_end":expected_end,"trailing_bytes":len(tail),"trailing_all_zero":not any(tail),
    "first_four_points_xyz":[list(struct.unpack_from("<fff",first,i*16)) for i in range(4)]},indent=2))
