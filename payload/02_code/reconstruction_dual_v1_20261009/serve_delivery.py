"""Preserve validated native asset publication while serving a dual-output landing page."""

import importlib.util
import json
from pathlib import Path
import re
import sys


def main():
    old = Path(__file__).resolve().parents[1] / "gaussian_full_raw_ab_20261008"
    sys.path.insert(0, str(old))
    spec = importlib.util.spec_from_file_location("dual_native_asset_server", old / "serve_raw_viewer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if "--root" not in sys.argv:
        raise ValueError("An explicit local delivery root is required")
    root = Path(sys.argv[sys.argv.index("--root") + 1])
    case = json.loads((root / "DELIVERY_IDENTITY.json").read_text())["case"]
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", case):
        raise ValueError("Unsafe case identifier")
    module.BRANCHES.add(case + "-optimized")
    native_server = module.ThreadingHTTPServer

    def create_server(address, handler):
        class DeliveryHandler(handler):
            def do_GET(self):
                if self.path.split("?", 1)[0] == "/":
                    self.path = "/index.html"
                return super().do_GET()
        return native_server(address, DeliveryHandler)

    module.ThreadingHTTPServer = create_server
    module.main()


if __name__ == "__main__":
    main()
