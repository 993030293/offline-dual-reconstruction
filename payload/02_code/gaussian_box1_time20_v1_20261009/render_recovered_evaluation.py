"""Use the original evaluator with an explicitly recovered localization file."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "02_code/gaussian_full_raw_ab_20261008"
sys.path.insert(0, str(OLD))
import render_evaluation as entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--localization", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    source = Path(entry.__file__).read_text(encoding="utf-8")
    anchor = 'localization_path = args.side / "localization/LOCALIZATION.json"'
    if source.count(anchor) != 1:
        raise ValueError("The original evaluator input-path anchor changed")
    source = source.replace(anchor, "localization_path = RECOVERED_LOCALIZATION")
    namespace = dict(vars(entry))
    exec(compile(source, entry.__file__, "exec"), namespace)
    namespace["RECOVERED_LOCALIZATION"] = args.localization
    original = sys.argv
    sys.argv = [entry.__file__, *remaining]
    try:
        namespace["main"]()
        result = namespace["json"].loads((Path(remaining[remaining.index("--output")+1]) / "RESULT.json").read_text())
        if result["summary"]["full_image"]["rendered_count"] != 24:
            raise ValueError("All24 original native renders remain required")
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
