import pandas as pd
import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/LLAVA/playground/data/ScienceQA")
DATA_DIR = ROOT / "data"
IMAGE_DIR = ROOT / "images"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def convert_split(split):
    parquet_file = next(DATA_DIR.glob(f"{split}-*.parquet"))
    output_file = ROOT / f"scienceqa_{split}_llava.json"

    df = pd.read_parquet(parquet_file)

    converted = []
    skipped = 0

    for idx, row in df.iterrows():

        # ---------- 图片 ----------
        image = row["image"]

        # 没有图片，跳过
        if image is None:
            skipped += 1
            continue

        image_bytes = image["bytes"]
        original_path = image.get("path", "image.png")

        suffix = Path(original_path).suffix or ".png"
        image_name = f"{split}_{idx:06d}{suffix}"
        image_path = IMAGE_DIR / image_name

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # ---------- 问题 ----------
        question = str(row["question"])

        # ---------- 选项 ----------
        choices = list(row["choices"])

        choice_text = ""
        for i, choice in enumerate(choices):
            letter = chr(ord("A") + i)
            choice_text += f"{letter}. {choice}\n"

        # ---------- 正确答案 ----------
        answer_idx = int(row["answer"])
        answer_letter = chr(ord("A") + answer_idx)
        answer_text = choices[answer_idx]

        # ---------- LLaVA 格式 ----------
        entry = {
            "id": f"{split}_{idx:06d}",
            "image": image_name,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{question}\n{choice_text}"
                },
                {
                    "from": "gpt",
                    "value": f"{answer_letter}. {answer_text}"
                }
            ]
        }

        converted.append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            converted,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"{split}:")
    print(f"  Original samples: {len(df)}")
    print(f"  Converted samples: {len(converted)}")
    print(f"  Skipped no-image: {skipped}")
    print(f"  Saved to: {output_file}")


convert_split("train")
convert_split("test")