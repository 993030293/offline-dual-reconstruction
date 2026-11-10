"""Update reversible target display controls without changing the reconstructed data."""

import argparse
from pathlib import Path
import shutil

from common import HERE, atomic_json, read_json


def refresh(root, target):
    result = read_json(target / 'DELIVERY_RESULT.json')
    folder = root / 'viewer/target'
    backup = target / 'prior_target_view_template'
    backup.mkdir(exist_ok=False)
    for name in ('index.html', 'point_viewer.js', 'manifest.json'):
        shutil.copyfile(folder / name, backup / name)
    page = (HERE / 'point_viewer.html').read_text(encoding='utf-8')
    case = read_json(root / 'CONFIG.json')['case']
    page = page.replace('box1-optimized', case + '-optimized').replace('PGSR 表面点云', 'PGSR 目标表面 · 已去地面')
    page = page.replace('完整表面', '全部目标表面').replace('自动聚焦', '目标聚焦')
    page = page.replace('<a href="../">双输出交付</a>', '<a href="../">双输出交付</a><a href="./reference_style.html">原版点云预览</a>')
    if not result['strict_ReliableLidar_profile_completed']:
        page = page.replace('坐标单位 m；', '视觉目标诊断；严格雷达支撑未通过。坐标单位 m；')
    (folder / 'index.html').write_text(page, encoding='utf-8')
    shutil.copyfile(HERE / 'point_viewer.js', folder / 'point_viewer.js')
    manifest = read_json(folder / 'manifest.json')
    manifest['defaultNavigation'] = 'overview'
    atomic_json(folder / 'manifest.json', manifest)
    old_surface = root / 'viewer/surface/index.html'
    if old_surface.exists():
        shutil.copyfile(old_surface, target / 'REJECTED_ALL_SCENE_VIEWER.html')
        shutil.copyfile(old_surface, old_surface.with_name('rejected_all_scene.html'))
        old_surface.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta http-equiv="refresh" content="0;url=../target/index.html"><title>目标点云已纠正</title>'
            '<p>当前交付已改为去地面的目标表面。<a href="../target/index.html">打开目标点云</a>。'
            '<a href="./rejected_all_scene.html">查看已撤回的全场景诊断</a>。</p></html>', encoding='utf-8')
    atomic_json(target / 'DISPLAY_REFRESH.json', {'status': 'target_overview_and_legacy_url_redirect_updated',
        'geometry_or_point_payload_changed': False, 'previous_templates': str(backup),
        'old_surface_url_is_now_redirect': old_surface.exists(), 'actual_browser_recheck_pending': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--target-root', type=Path, required=True)
    args = parser.parse_args()
    refresh(args.run_root.resolve(strict=True), args.target_root.resolve(strict=True))
