#!/usr/bin/env python3
import cv2
import math
import os
import sys
from typing import Dict, List, Tuple

DATASET_DIR_DEFAULT = "../dataset3"
WINDOW_NAME = "coin-annotator"


def list_images(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png"}
    files = []
    for name in os.listdir(folder):
        if os.path.splitext(name.lower())[1] in exts:
            files.append(name)
    return sorted(files)


def read_csv(path: str) -> List[Tuple[int, int, int]]:
    circles: List[Tuple[int, int, int]] = []
    if not os.path.exists(path):
        return circles
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()
            if len(parts) != 3:
                continue
            x, y, r = map(int, parts)
            circles.append((x, y, r))
    return circles


def write_csv(path: str, circles: List[Tuple[int, int, int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for x, y, r in circles:
            f.write(f"{x},{y},{r}\n")


def annotate_image(img_path: str, initial: List[Tuple[int, int, int]]) -> Tuple[List[Tuple[int, int, int]], bool]:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Cannot open image: {img_path}")
        return [], False

    saved: List[Tuple[int, int, int]] = list(initial)
    center: Tuple[int, int] = (-1, -1)
    drawing = False
    preview_radius = 0
    quit_all = False

    def on_mouse(event, x, y, flags, _userdata):
        nonlocal center, drawing, preview_radius, saved
        if event == cv2.EVENT_LBUTTONDOWN:
            if not drawing:
                center = (x, y)
                preview_radius = 0
                drawing = True
            else:
                radius = int(round(math.hypot(x - center[0], y - center[1])))
                saved.append((center[0], center[1], radius))
                drawing = False
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            preview_radius = int(
                round(math.hypot(x - center[0], y - center[1])))

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        vis = img.copy()
        for x, y, r in saved:
            cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
            cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)
        if drawing and preview_radius > 0:
            cv2.circle(vis, center, preview_radius, (0, 255, 255), 1)
            cv2.circle(vis, center, 2, (0, 255, 255), -1)

        cv2.imshow(WINDOW_NAME, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            if drawing:
                drawing = False
                continue
            quit_all = True
            break
        if key == ord("u") and saved:
            saved.pop()
        if key == 32:  # space
            break

    return saved, quit_all


def print_help(prog: str):
    print(
        f"Usage: {prog} [--dataset <path>] [--single <imgname>] [--help]\n"
        "Controls:\n"
        "  Left click: set center / confirm radius\n"
        "  Move mouse: adjust radius when in drawing mode\n"
        "  u: undo last circle\n"
        "  Space: save current image and advance\n"
        "  ESC: if drawing, cancel the circle; otherwise quit\n"
    )


def parse_args(argv: List[str]) -> Tuple[str, str, bool]:
    dataset = DATASET_DIR_DEFAULT
    single = None
    show_help = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            show_help = True
            i += 1
        elif arg == "--dataset" and i + 1 < len(argv):
            dataset = argv[i + 1]
            i += 2
        elif arg == "--single" and i + 1 < len(argv):
            single = argv[i + 1]
            i += 2
        else:
            print(f"Unknown argument: {arg}")
            show_help = True
            i += 1
    return dataset, single, show_help


def main():
    dataset_dir, target, want_help = parse_args(sys.argv[1:])
    if want_help:
        print_help(sys.argv[0])
        return

    if not os.path.isdir(dataset_dir):
        print(f"Dataset folder not found: {dataset_dir}")
        return

    images = list_images(dataset_dir)
    if not images:
        print(f"No images found in {dataset_dir}")
        return

    if target and target not in images:
        print(f"{target} not found in {dataset_dir}")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def annotate_and_save(img_name: str) -> bool:
        stem, _ = os.path.splitext(img_name)
        csv_path = os.path.join(dataset_dir, f"{stem}.csv")
        existing = read_csv(csv_path)
        print(
            f"Annotating {img_name} (click center, drag radius, click again; space=next, ESC=quit/cancel, u=undo)"
        )
        circles, quit_all = annotate_image(
            os.path.join(dataset_dir, img_name), existing)
        write_csv(csv_path, circles)
        print(f"Saved {len(circles)} circles to {csv_path}")
        return quit_all

    if target:
        annotate_and_save(target)
        cv2.destroyAllWindows()
        return

    for name in images:
        quit_all = annotate_and_save(name)
        if quit_all:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
