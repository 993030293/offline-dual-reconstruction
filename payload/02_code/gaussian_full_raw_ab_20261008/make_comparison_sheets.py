"""Create six-view visual evidence from paired successful raw evaluation renders."""

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = {mode: json.loads((args.case_root / mode / "evaluation/RESULT.json").read_text())
            for mode in ("baseline", "optimized")
            if (args.case_root / mode / "evaluation/RESULT.json").is_file()}
    if not data or (not args.allow_incomplete and len(data) != 2):
        raise ValueError("The visual protocol requires actual evaluation results")
    rows = {mode: {row["image_name"]: row for row in item["rows"]} for mode, item in data.items()}
    if args.allow_incomplete:
        frozen = json.loads((args.case_root / "optimized/initial/EVALUATION_FRAMES.json").read_text())["rows"]
        names = [row["image_name"] for row in frozen]
        if len(names) != 24 or any([row["image_name"] for row in item["rows"]] != names for item in data.values()):
            raise ValueError("Incomplete comparisons must preserve all frozen evaluation rows")
        selected = [names[int((number+.5)*24/6)] for number in range(6)]
    else:
        names = [row["image_name"] for row in data["baseline"]["rows"]
                 if row.get("localized") and rows["optimized"][row["image_name"]].get("localized")]
        if len(names) < 6:
            raise ValueError("Six shared successful views are required for this visual protocol")
        selected = [names[int((number+.5)*len(names)/6)] for number in range(6)]
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    sheets = []
    paired_reference_equal = True
    for part in range(2):
        sheet = Image.new("RGB", (1218, 1064), "#20242c")
        draw = ImageDraw.Draw(sheet)
        for row_index, name in enumerate(selected[part*3:part*3+3]):
            stem = Path(name).stem
            references = [args.case_root / mode / "localization/reference_images" / (stem+".png")
                          for mode in ("baseline", "optimized")
                          if (args.case_root / mode / "localization/reference_images" / (stem+".png")).is_file()]
            if not references or (len(references) == 2 and references[0].read_bytes() != references[1].read_bytes()):
                raise ValueError("The paired native reference PNG bytes differ")
            paired_reference_equal = paired_reference_equal and len(references) == 2
            paths = [references[0]]+[args.case_root / mode / "evaluation" / (stem+".png") for mode in ("baseline", "optimized")]
            for column, (label, path) in enumerate(zip(("Raw reference", "Native baseline", "Optimized"), paths)):
                x, y = column*406, row_index*352
                draw.text((x+8, y+6), name.split("_")[1]+"  "+label, font=font, fill="white")
                if path.is_file():
                    if column and not rows.get(("baseline", "optimized")[column-1], {}).get(name, {}).get("localized"):
                        raise ValueError("An unlocalized evaluation row has an unexpected rendered image")
                    with Image.open(path) as image:
                        preview = image.convert("RGB").resize((406, 310), Image.Resampling.LANCZOS)
                    sheet.paste(preview, (x, y+34))
                elif args.allow_incomplete:
                    draw.text((x+12, y+150), "UNAVAILABLE: failure retained", font=font, fill="#e5a5a5")
                else:
                    raise FileNotFoundError(path)
        target = args.output / ("six_views_part"+str(part+1)+".png")
        sheet.save(target)
        sheets.append(str(target.resolve()))
    atomic_json(args.output / "VISUAL_PROTOCOL.json", {"status": "render_sheets_created_human_or_agent_visual_inspection_pending",
        "case": args.case_root.name, "complete_raw_evaluation_denominator": 24,
        "incomplete_comparison_explicit": args.allow_incomplete,
        "evaluation_branches_present": sorted(data),
        "shared_localized_count": sum(all(name in rows.get(mode, {}) and rows[mode][name].get("localized")
            for mode in ("baseline", "optimized")) for name in names),
        "six_selected_image_names": selected, "sheets": sheets, "same_reference_PNG_bytes": paired_reference_equal,
        "selection_rule": "six fixed midpoints among all 24 frozen views; unavailable results left empty" if args.allow_incomplete else
            "six equally spaced stratum midpoints among shared localized raw views",
        "view_inspection_does_not_establish_GT_geometry": True})
    print(json.dumps({"sheets": sheets, "names": selected}))


if __name__ == "__main__":
    main()
