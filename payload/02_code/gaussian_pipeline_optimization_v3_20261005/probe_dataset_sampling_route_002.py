"""Recheck the original route after preserving the complete native configuration."""

import argparse
from pathlib import Path
import sys

from complete_carrier_configuration import complete_configuration


def main():
    original = Path(__file__).with_name("probe_dataset_sampling_route.py")
    source = original.read_text(encoding="utf-8")
    replacements = {
        '        structure = [path.relative_to(dataset).as_posix() for path in dataset.rglob("*")':
            '        _complete_configuration(carrier, names, selected[:1], output / "configuration_proof")\n'
            '        structure = [path.relative_to(dataset).as_posix() for path in dataset.rglob("*")',
        '                   CODE / "camera_pixel_remap.py", V1 / "runtime.py", LIBRARY]':
            '                   CODE / "camera_pixel_remap.py", V1 / "runtime.py", LIBRARY, '
            'CODE / "complete_carrier_configuration.py", CODE / "probe_dataset_sampling_route_002.py"]',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError("An original unchanged dataset probe anchor differs")
        source = source.replace(before, after)
    namespace = {"__name__": "private_complete_configuration_probe", "__file__": str(original),
                 "_complete_configuration": complete_configuration}
    exec(compile(source, str(original) + "#complete_configuration", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
