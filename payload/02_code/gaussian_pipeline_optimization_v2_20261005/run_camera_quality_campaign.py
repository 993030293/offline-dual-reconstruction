"""Reuse the complete native paired campaign with an independently verified common-prefix recovery."""

import inspect
from pathlib import Path
import textwrap

import run_camera_execution_campaign as original


def main():
    source = textwrap.dedent(inspect.getsource(original.main))
    parser_anchor = "    args = parser.parse_args()"
    if source.count(parser_anchor) != 1:
        raise ValueError("The pinned native paired camera parser changed")
    source = source.replace(parser_anchor,
        '    parser.add_argument("--shared-prefix-recovery", type=Path, required=True)\n'+parser_anchor)
    begin = source.index('        recovery = args.output/"recovery"')
    end = source.index("        for seed in range(3):", begin)
    replacement = '''        shared = read(args.shared_prefix_recovery/"RESULT.json")
        if shared["status"] != "same_prefix_recovery_passed_visual_pending" or not all(shared["checks"].values()):
            raise ValueError("The common-prefix native recovery did not pass its unchanged all24 quality gate")
        current_sources = {Path(item["path"]).name: item["sha256"] for item in sources}
        for item in shared["sources"]:
            name = Path(item["path"]).name
            if name in current_sources and current_sources[name] != item["sha256"]:
                raise ValueError("A shared-prefix recovery adapter differs from the frozen training group")
        state["same_prefix_recovery"] = {"receipt": str(args.shared_prefix_recovery/"RESULT.json"),
            "checks": shared["checks"], "candidate_minus_control": shared["candidate_minus_control"],
            "new_optimization_count": 0}
        atomic_json(args.output/"STATE.json", state)
'''
    source = source[:begin]+replacement+source[end:]
    path_anchor = "    paths = [Path(__file__), args.baseline_config, args.candidate_config]"
    if source.count(path_anchor) != 1:
        raise ValueError("The full camera campaign source binding changed")
    source = source.replace(path_anchor,
        '    paths = [Path(__file__), Path(__file__).with_name("run_camera_execution_campaign.py"), args.baseline_config, args.candidate_config]')
    source = source.replace('"recovery": "1100 continuous, actual exit86 at600, resume complete500 checkpoint"',
        '"recovery": "verified actual exit86 and native500 restore with the same saved prefix as the uninterrupted reference"')
    namespace = dict(original.main.__globals__)
    namespace["__file__"] = str(Path(__file__))
    exec(compile(source, "<original_three_seed_camera_campaign_with_verified_shared_prefix>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
