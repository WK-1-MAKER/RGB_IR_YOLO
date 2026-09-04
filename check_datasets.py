#!/usr/bin/env python3
"""Interactively inspect YOLO bounding-box annotations with OpenCV."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import cv2


DEFAULT_IMAGE_DIR = Path(
    "/home/SENSETIME/wenkai/slam_datasets/FLIR/FLIR/rgb/train/images"
)
DEFAULT_LABEL_DIR = Path(
    "/home/SENSETIME/wenkai/slam_datasets/FLIR/FLIR/rgb/train/labels"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_label_line(line: str, image_width: int, image_height: int) -> tuple[int, int, int, int, int]:
    """Parse one normalized YOLO row and return ``class_id, x1, y1, x2, y2``."""
    fields = line.split()
    if len(fields) != 5:
        raise ValueError("a label row must contain exactly five fields")

    try:
        class_id = int(fields[0])
        values = [float(value) for value in fields[1:]]
    except (TypeError, ValueError) as exc:
        raise ValueError("label fields must be numeric") from exc

    if class_id < 0 or not all(math.isfinite(value) for value in values):
        raise ValueError("class_id and coordinates must be finite and non-negative")
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError("normalized coordinates must be in the range [0, 1]")
    x_center, y_center, width, height = values
    if width <= 0.0 or height <= 0.0:
        raise ValueError("box width and height must be positive")

    x1 = round((x_center - width / 2.0) * image_width)
    y1 = round((y_center - height / 2.0) * image_height)
    x2 = round((x_center + width / 2.0) * image_width)
    y2 = round((y_center + height / 2.0) * image_height)
    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(0, min(image_width - 1, x2))
    y2 = max(0, min(image_height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box is too small for the image resolution")
    return class_id, x1, y1, x2, y2


def read_labels(label_path: Path, image_width: int, image_height: int) -> list[tuple[int, int, int, int, int]]:
    """Read all rows from a label file, rejecting the whole file if one row is invalid."""
    with label_path.open("r", encoding="utf-8") as label_file:
        return [
            parse_label_line(line, image_width, image_height)
            for line in label_file
            if line.strip()
        ]


def image_paths(image_dir: Path) -> Iterable[Path]:
    """Return supported image files in deterministic filename order."""
    return sorted(
        (path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name,
    )


def inspect_dataset(image_dir: Path, label_dir: Path) -> None:
    """Display each valid image and its annotations until the user quits."""
    cv2.namedWindow("Dataset annotation checker", cv2.WINDOW_NORMAL)
    try:
        for image_path in image_paths(image_dir):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"跳过无法读取的图片: {image_path}")
                continue

            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                print(f"跳过缺少标注的图片: {image_path}")
                continue

            try:
                boxes = read_labels(label_path, image.shape[1], image.shape[0])
            except (OSError, UnicodeError, ValueError) as exc:
                print(f"跳过损坏的标注 {label_path}: {exc}")
                continue

            for class_id, x1, y1, x2, y2 in boxes:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    str(class_id),
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Dataset annotation checker", image)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != ord("d"):
                # Keep the current image visible until d, q, or Esc is pressed.
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key in (ord("d"), ord("q"), 27):
                        break
                if key in (ord("q"), 27):
                    break
    finally:
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", nargs="?", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("label_dir", nargs="?", type=Path, default=DEFAULT_LABEL_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.image_dir.is_dir():
        raise SystemExit(f"图片目录不存在: {args.image_dir}")
    if not args.label_dir.is_dir():
        raise SystemExit(f"标注目录不存在: {args.label_dir}")
    inspect_dataset(args.image_dir, args.label_dir)


if __name__ == "__main__":
    main()
