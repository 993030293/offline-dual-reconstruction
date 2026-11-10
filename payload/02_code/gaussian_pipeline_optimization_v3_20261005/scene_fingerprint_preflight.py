"""Check the current complete dataset, runtime and bounded existing environment."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def main():
    if OUT.exists():
        raise FileExistsError("A fresh field-cost scope is required")
    OUT.mkdir()
    code = '''import json,os,pathlib,shutil,sys,subprocess,psutil
root=pathlib.Path("/mnt/c/lidar-camera-fusion-mask-robust-downsampling")
sys.path.insert(0,str(root/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import activate_reference
meta=activate_reference(root/"04_outputs/box2_3dgs_display_v1_20260930")
import torch
from scene.colmap_loader import read_extrinsics_binary
dataset=root/"04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003/dataset"
rows=read_extrinsics_binary(dataset/"sparse/0/images.bin")
names={row.name for row in rows.values()}
actual={str(p.relative_to(dataset/"images")) for p in (dataset/"images").rglob("*") if p.is_file()}
others=[]
for p in psutil.process_iter(["pid","cmdline"]):
 if p.pid != os.getpid() and any("python" in x.lower() for x in (p.info["cmdline"] or [])): others.append(p.info)
free,total=torch.cuda.mem_get_info()
compiler=shutil.which("cc")
openssl=shutil.which("openssl")
print(json.dumps({"runtime":meta,"python":sys.executable,"uid":os.getuid(),"torch":torch.__version__,"CUDA":torch.version.cuda,
 "GPU":torch.cuda.get_device_name(0),"GPU_free_bytes":free,"GPU_total_bytes":total,"host_available_bytes":psutil.virtual_memory().available,
 "registered_images":len(rows),"all_registered_names_exact":len(rows)==len(names)==1016 and names==actual,"other_Linux_python_processes":others,
 "Linux_boot_id":pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
 "own_process":{"pid":os.getpid(),"creation_time":psutil.Process().create_time()},
 "existing_cc":compiler,"existing_cc_version":subprocess.check_output([compiler,"--version"],text=True).splitlines()[0] if compiler else None,
 "existing_openssl_version":subprocess.check_output([openssl,"version"],text=True).strip() if openssl else None,
 "existing_evp_header":pathlib.Path("/usr/include/openssl/evp.h").is_file(),"existing_sha_header":pathlib.Path("/usr/include/openssl/sha.h").is_file()}))
'''
    linux = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=60))
    checks = {"current_full1016_registered_names_exact": linux["all_registered_names_exact"],
              "enough_original_GPU_and_host_headroom": linux["GPU_free_bytes"] >= 2 * 1024**3 and linux["host_available_bytes"] >= 16 * 1024**3,
              "no_other_Linux_python_worker": not linux["other_Linux_python_processes"],
              "original_reference_commit": linux["runtime"]["reference_commit"] == "54c035f7834b564019656c3e3fcc3646292f727d",
              "original_disk_headroom": shutil.disk_usage(ROOT).free >= 20 * 1024**3}
    paths = [Path(__file__), CODE / "profile_scene_fingerprint_cost.py", CODE / "fingerprint_scene_node.py", CODE / "profile_mapped_scene.py"]
    atomic_json(OUT / "PREFLIGHT_001.json", {"status": "complete_field_cost_preflight0" if all(checks.values()) else "failed_verification",
        "checks": checks, "Linux": linux, "Windows_memory_total_bytes": psutil.virtual_memory().total,
        "Windows_disk_free_bytes": shutil.disk_usage(ROOT).free, "sources": [data_identity(path) for path in paths],
        "no_environment_install_or_migration": True, "new_optimization_count": 0})
    if not all(checks.values()):
        raise AssertionError("The original bounded field diagnostic preflight failed")
    print(json.dumps({"checks": checks, "cc": linux["existing_cc_version"], "openssl": linux["existing_openssl_version"],
                      "headers": [linux["existing_evp_header"], linux["existing_sha_header"]], "registered": linux["registered_images"]}))


if __name__ == "__main__":
    main()
