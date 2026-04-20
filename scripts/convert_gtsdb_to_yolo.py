#!/usr/bin/env python3
"""Convert GTSDB-style annotations to YOLO format with train/val/test split.

Expected GT format (semicolon-separated):
Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId

Example:
00000.ppm;1360;800;774;411;815;446;11
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GTSDB annotations to YOLO dataset layout")
    parser.add_argument("--src-images", type=Path, required=True, help="Directory of source images")
    parser.add_argument("--gt-file", type=Path, required=True, help="Path to GT annotation file")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root like data/gtsdb")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into output folders (default). If omitted, still copies (kept for explicitness).",
    )
    return parser.parse_args()


def ensure_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def to_yolo(width: float, height: float, box: Box) -> Tuple[float, float, float, float]:
    cx = ((box.x1 + box.x2) / 2.0) / width
    cy = ((box.y1 + box.y2) / 2.0) / height
    w = (box.x2 - box.x1) / width
    h = (box.y2 - box.y1) / height
    return cx, cy, w, h


def read_annotations(
    gt_file: Path, src_images: Path
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, List[Box]], List[int]]:
    image_size: Dict[str, Tuple[int, int]] = {}
    anns: Dict[str, List[Box]] = defaultdict(list)
    class_ids = set()

    with gt_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if not row:
                continue
            if row[0].lower() in {"filename", "path"}:
                continue
            filename = row[0].strip()

            # Support both variants:
            # 1) filename;width;height;x1;y1;x2;y2;classid
            # 2) filename;x1;y1;x2;y2;classid
            if len(row) >= 8:
                width = int(float(row[1]))
                height = int(float(row[2]))
                x1 = float(row[3])
                y1 = float(row[4])
                x2 = float(row[5])
                y2 = float(row[6])
                class_id = int(float(row[7]))
            elif len(row) >= 6:
                img_path = src_images / filename
                if not img_path.exists():
                    raise FileNotFoundError(f"Image listed in GT not found: {img_path}")
                with Image.open(img_path) as im:
                    width, height = im.size
                x1 = float(row[1])
                y1 = float(row[2])
                x2 = float(row[3])
                y2 = float(row[4])
                class_id = int(float(row[5]))
            else:
                raise ValueError(f"Invalid row (expected 6 or 8 columns): {row}")

            image_size[filename] = (width, height)
            anns[filename].append(Box(x1=x1, y1=y1, x2=x2, y2=y2, class_id=class_id))
            class_ids.add(class_id)

    return image_size, anns, sorted(class_ids)


def build_class_map(class_ids: List[int]) -> Dict[int, int]:
    return {orig: idx for idx, orig in enumerate(class_ids)}


def choose_split(names: List[str], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, str]:
    rng = random.Random(seed)
    shuffled = names[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    split_map: Dict[str, str] = {}
    for name in shuffled[:n_train]:
        split_map[name] = "train"
    for name in shuffled[n_train : n_train + n_val]:
        split_map[name] = "val"
    for name in shuffled[n_train + n_val : n_train + n_val + n_test]:
        split_map[name] = "test"
    return split_map


def write_outputs(
    src_images: Path,
    out_root: Path,
    image_size: Dict[str, Tuple[int, int]],
    anns: Dict[str, List[Box]],
    class_map: Dict[int, int],
    split_map: Dict[str, str],
) -> Tuple[int, int]:
    copied = 0
    missing = 0

    for filename, boxes in anns.items():
        src_img = src_images / filename
        if not src_img.exists():
            missing += 1
            continue

        split = split_map[filename]
        out_img_name = Path(filename).stem + ".jpg"
        dst_img = out_root / "images" / split / out_img_name
        dst_label = out_root / "labels" / split / (Path(filename).stem + ".txt")

        with Image.open(src_img) as im:
            im.convert("RGB").save(dst_img, format="JPEG", quality=95)
        copied += 1

        width, height = image_size[filename]
        lines = []
        for b in boxes:
            cid = class_map[b.class_id]
            cx, cy, w, h = to_yolo(width, height, b)
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            w = min(max(w, 1e-9), 1.0)
            h = min(max(h, 1e-9), 1.0)
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return copied, missing


def write_metadata(out_root: Path, class_map: Dict[int, int], split_map: Dict[str, str]) -> None:
    rev_map = {new: old for old, new in class_map.items()}

    classes_file = out_root / "classes.txt"
    class_lines = [f"{idx},class_{orig_id}" for idx, orig_id in sorted(rev_map.items())]
    classes_file.write_text("\n".join(class_lines) + "\n", encoding="utf-8")

    mapping_file = out_root / "class_mapping.json"
    mapping_file.write_text(json.dumps({str(k): v for k, v in class_map.items()}, indent=2), encoding="utf-8")

    manifest = out_root / "split_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "split"])
        for name in sorted(split_map):
            writer.writerow([name, split_map[name]])


def validate_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    ensure_dirs(args.output_root)
    image_size, anns, class_ids = read_annotations(args.gt_file, args.src_images)

    if not anns:
        raise ValueError("No annotations found in gt file")

    class_map = build_class_map(class_ids)
    split_map = choose_split(sorted(anns.keys()), args.train_ratio, args.val_ratio, args.seed)

    copied, missing = write_outputs(
        src_images=args.src_images,
        out_root=args.output_root,
        image_size=image_size,
        anns=anns,
        class_map=class_map,
        split_map=split_map,
    )
    write_metadata(args.output_root, class_map, split_map)

    print("Conversion complete")
    print(f"Annotated images: {len(anns)}")
    print(f"Copied images: {copied}")
    print(f"Missing images: {missing}")
    print(f"Classes: {len(class_ids)}")
    print(f"Class mapping saved to: {args.output_root / 'class_mapping.json'}")
    print(f"Class names template saved to: {args.output_root / 'classes.txt'}")
    print("Next step: update configs/dataset.yaml with nc and names from classes.txt")


if __name__ == "__main__":
    main()
