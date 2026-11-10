"""Verify installed offline components before a new raw reconstruction stage."""

import argparse
import importlib
from pathlib import Path
import shutil
import subprocess

from common import HERE, OLD, ROOT, SFM_PYTHON, COLMAP, DISTRO, WSL_PYTHON, atomic_json, linux_path


def decode_process_output(value):
    """Decode subprocess bytes while preserving UTF-8 and UTF-16 diagnostics."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if value.startswith((b"\xff\xfe", b"\xfe\xff")):
        return value.decode("utf-16", errors="replace")
    sample = value[:256]
    nul_count = sample.count(0)
    if nul_count >= max(2, len(sample) // 8):
        even_nuls = sample[0::2].count(0)
        odd_nuls = sample[1::2].count(0)
        encoding = "utf-16-le" if odd_nuls >= even_nuls else "utf-16-be"
        return value.decode(encoding, errors="replace")
    return value.decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    modules = {}
    for name in ("numpy", "cv2", "PIL", "pycolmap", "pyceres", "scipy", "psutil"):
        module = importlib.import_module(name)
        modules[name] = {"path": module.__file__, "version": getattr(module, "__version__", None)}
    from package_delivery import browser_dependencies
    from native_quality_runtime import verify_native_quality_runtime
    from run_supported_sparse import bind_reference_camera_tools
    from run_initial_safe import load_native
    node, browser = browser_dependencies()
    initial_frontend = load_native()
    sparse_frontend = bind_reference_camera_tools()
    native_quality = verify_native_quality_runtime(initial_frontend.demosaic_rggb8_to_bgr)
    browser_probe = ('const p=require(process.argv[1]); (async()=>{const b=await p.chromium.launch({headless:true}); '
        'const page=await b.newPage(); await page.setContent("<title>offline-preview-probe</title>"); '
        'if(await page.title()!=="offline-preview-probe")process.exit(2); await b.close(); '
        'console.log("local Playwright browser launched")})().catch(error=>{console.error(error);process.exit(1)})')
    node_check = subprocess.run([str(node), "-e", browser_probe, str(browser)],
        capture_output=True, text=True, check=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
    required = [SFM_PYTHON, COLMAP, HERE / "run_delivery.py", HERE / "pgsr_branch.py",
                OLD / "run_side.py", ROOT / "04_outputs/box2_3dgs_display_v1_20260930/MAIN_CONFIG.json"]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("An explicitly required offline component is absent")
    from target_prepare import PACKAGE
    from common import read_json
    target_runtime = read_json(PACKAGE / 'config/local_runtime.json')
    target_required = [PACKAGE / 'code/derive_vehicle_roi.py', PACKAGE / 'RUN_FROM_S07_M3.ps1',
        Path(target_runtime['generic_python']), Path(target_runtime['rosbags_vendor']) / 'rosbags']
    if any(not path.exists() for path in target_required) or not shutil.which('pwsh'):
        raise FileNotFoundError('An installed offline target geometry dependency is absent')
    generic_check = subprocess.run([target_runtime['generic_python'], '-c',
        'import open3d, scipy, plotly; print(open3d.__version__, scipy.__version__, plotly.__version__)'],
        capture_output=True, text=True, check=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
    free = shutil.disk_usage(args.run_root).free
    if free < 32 * 1024**3:
        raise RuntimeError("Fresh dual outputs require at least 32 GiB of workspace free space")
    code = 'import importlib.util,json,torch; names=["open3d","PIL","numpy","psutil","plyfile"]; missing=[n for n in names if importlib.util.find_spec(n) is None]; assert not missing, missing; assert torch.cuda.is_available(); print(json.dumps({"torch":torch.__version__,"GPU":torch.cuda.get_device_name(0),"cuda_available":True,"modules":names}))'
    command = ["wsl.exe", "-d", DISTRO, "--exec", "/usr/bin/env",
        "PYTHONPATH="+linux_path(HERE / "offline_guard"), "PIP_NO_INDEX=1", "HF_HUB_OFFLINE=1",
        WSL_PYTHON, "-c", code]
    linux = subprocess.run(command, capture_output=True, check=True, timeout=60)
    linux_stdout = decode_process_output(linux.stdout).strip()
    linux_stderr = decode_process_output(linux.stderr).strip()
    atomic_json(args.output, {"status": "installed_offline_components_verified", "Windows_modules": modules,
        "WSL_stdout": linux_stdout, "WSL_stderr": linux_stderr, "node": str(node),
        "browser_package": str(browser), "workspace_free_bytes": free,
        "actual_node_module_import": node_check.stdout.strip(),
        "actual_initial_frontend_import": initial_frontend.decode_bayer.__code__.co_filename,
        "actual_sparse_frontend_import": sparse_frontend.decode_bayer.__code__.co_filename,
        "actual_native_quality_runtime": native_quality,
        "actual_target_dependency_import": generic_check.stdout.strip(),
        "target_package": str(PACKAGE), "PowerShell7": shutil.which('pwsh'),
        "dependency_install_or_download_executed": False, "LLM_or_cloud_API_required": False,
        "same_machine_environment_verified_portable_deployment_not_verified": True})


if __name__ == "__main__":
    main()
