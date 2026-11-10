"""Check the live original environment and the installed Pillow lifecycle."""

import json
import subprocess

import scene_fingerprint_preflight as parent
from runtime import atomic_json


def main():
    parent.OUT = parent.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/shf001"
    parent.main()
    code = '''import inspect,json,pathlib,PIL
from PIL import Image
root=pathlib.Path("/mnt/c/lidar-camera-fusion-mask-robust-downsampling")
path=next(iter(sorted((root/"04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003/dataset/images").glob("*.jpg"))))
with Image.open(path) as image:
 image.resize((31,29))
 pixel_inside=image.getpixel((0,0))
actual_type=type(image).__module__+"."+type(image).__name__
after_context=image.getpixel((0,0))
file_closed=image.fp is None
image.close()
try:
 image.getpixel((0,0))
except ValueError:
 core_closed=True
else:
 core_closed=False
print(json.dumps({"Pillow_version":PIL.__version__,"Pillow_file":PIL.__file__,"actual_image_class":actual_type,
 "actual_source_path":str(path),"context_file_closed_core_kept":file_closed and pixel_inside==after_context,
 "explicit_close_destroys_core":core_closed,"native_sources":{name:inspect.getsource(value) for name,value in
 (("Image.open",Image.open),("Image.Image.__exit__",Image.Image.__exit__),("Image.Image.close",Image.Image.close),("Image.Image.resize",Image.Image.resize))}}))
'''
    result = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=60))
    atomic_json(parent.OUT / "PILLOW_LIFECYCLE_001.json", result)
    if not result["context_file_closed_core_kept"] or not result["explicit_close_destroys_core"]:
        raise AssertionError("The installed native Pillow lifecycle does not support owned handoff")
    print(json.dumps({key: result[key] for key in ("Pillow_version", "actual_image_class", "context_file_closed_core_kept", "explicit_close_destroys_core")}))


if __name__ == "__main__":
    main()
