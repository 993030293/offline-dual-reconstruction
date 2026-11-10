"""Run the original complete dataset screen with the registered shared carrier."""

from pathlib import Path


def main():
    path = Path(__file__).with_name("screen_complete_dataset_sampling.py")
    source = path.read_text(encoding="utf-8")
    anchors = {
        "from dataset_sampling_carrier import cached_dataset, closed, ROOT, OWN":
            "from dataset_sampling_shared_carrier import cached_dataset, closed, ROOT, OWN",
        'sources = [Path(__file__), Path(__file__).with_name("dataset_sampling_carrier.py"),':
            'sources = [Path(__file__), Path(__file__).with_name("screen_complete_dataset_sampling.py"), '
            'Path(__file__).with_name("dataset_sampling_shared_carrier.py"), Path(__file__).with_name("dataset_sampling_carrier.py"),',
        '"native_full_model_carrier_cached_four_process_RGB_sampling"':
            '"native_full_model_carrier_shared_eight_process_RGB_sampling"',
        "snapshots.mkdir()":
            'snapshots.mkdir()\n    (snapshots / "COMPLETE_EXECUTED_DRIVER.py").write_text(_driver_text, encoding="utf-8")',
    }
    for anchor, replacement in anchors.items():
        if source.count(anchor) != 1:
            raise ValueError("The complete original dataset screen contract changed")
        source = source.replace(anchor, replacement)
    namespace = {"__file__": str(Path(__file__).resolve()), "__name__": "__main__", "_driver_text": source}
    exec(compile(source, "<complete_original_dataset_screen_with_shared_map>", "exec"), namespace)


if __name__ == "__main__":
    main()
