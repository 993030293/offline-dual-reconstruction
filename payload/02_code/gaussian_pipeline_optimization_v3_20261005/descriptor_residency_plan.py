"""Plan exact bounded descriptor residency independently on both pipe endpoints."""

from collections import OrderedDict
import hashlib
import json

MAXIMUM_CACHE_BYTES = 64*1024**2
BYTES_PER_ROW = 132
MAXIMUM_CACHE_ENTRIES = 4096
INPUT_BYTES = 32*1024**2
REPLY_BYTES = 12*12000*20
MATRIX_ELEMENTS = 144_000_000


def shape_admitted(nq, counts):
    if type(nq) is not int or nq < 1 or not isinstance(counts, list) or not 1 <= len(counts) <= 12:
        return False
    if any(type(nt) is not int or nt < 2 for nt in counts):
        return False
    pq = max(32, (nq+15)//16*16)
    return ((nq+sum(counts))*128 <= INPUT_BYTES
        and nq*16*len(counts)+sum(counts)*4 <= REPLY_BYTES
        and all(pq*max(8, (nt+7)//8*8) <= MATRIX_ELEMENTS for nt in counts))


def compact_plan(plan):
    encoded = json.dumps(plan, separators=(",", ":")).encode("ascii")
    return {"state_sha256": hashlib.sha256(encoded).hexdigest(), "missing_count": len(plan["missing"]),
        "evicted_count": len(plan["evicted"]), "resident_bytes": plan["resident_bytes"],
        "resident_entries": len(plan["resident_keys"])}


class DescriptorResidencyPlan:
    def __init__(self, limit=MAXIMUM_CACHE_BYTES):
        if type(limit) is not int or not 264 <= limit <= MAXIMUM_CACHE_BYTES:
            raise ValueError("The private descriptor cache byte budget is unsupported")
        self.limit, self.rows = limit, OrderedDict()

    def plan(self, items):
        unique = OrderedDict()
        for key, count in items:
            if (not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key)
                    or type(count) is not int or count < 2):
                raise ValueError("The complete descriptor content identity or row count is invalid")
            if key in unique and unique[key] != count or key in self.rows and self.rows[key] != count:
                raise ValueError("A complete descriptor identity changed its shape")
            unique[key] = count
        if not unique or len(unique) > 12 or sum(unique.values())*BYTES_PER_ROW > self.limit:
            raise ValueError("All current descriptors must fit without truncation")
        rows = self.rows.copy()
        missing, evicted = [], []
        for key, count in unique.items():
            if key not in rows:
                while (sum(rows.values())+count)*BYTES_PER_ROW > self.limit or len(rows) >= MAXIMUM_CACHE_ENTRIES:
                    victim = next((name for name in rows if name not in unique), None)
                    if victim is None:
                        raise ValueError("Pinned descriptor residency cannot satisfy its byte budget")
                    del rows[victim]
                    evicted.append(victim)
                rows[key] = count
                missing.append([key, count])
            rows.move_to_end(key)
        self.rows = rows
        return {"missing": missing, "evicted": evicted, "resident_bytes": sum(rows.values())*BYTES_PER_ROW,
            "unique_current_keys": list(unique), "resident_keys": list(rows)}

    def clear(self):
        self.rows.clear()
