# Traffic Management System

An experimental project exploring an Adaptive Traffic Control System (ATCS) and object detection using YOLOv7-tiny for traffic videos.

## What’s inside

- `atcs.py`: a simple adaptive signal controller that rotates lanes based on counted vehicles.
- `input_retrieval.py`: argument parser helper used by YOLO detection workflows.
- `runDetections.py`: batch helper to run detection over a folder of videos.
- `yolo-coco/`: model config, labels, and weights (note: large files; consider Git LFS).
- `input_videos/`: sample input videos (ignored by git via .gitignore).
- `example_gif/`: images for documentation.

## Quick start

1) Create and activate a virtual environment (optional but recommended)
	- macOS/Linux: python3 -m venv .venv && source .venv/bin/activate

2) Install dependencies
	- pip install -r requirements.txt

3) Prepare models and data
	- Place `coco.names`, `yolov7-tiny.cfg`, and `yolov7-tiny.weights` under `yolo-coco/`.
	- Put input videos under `input_videos/` (e.g., `.mp4`, `.avi`).

4) Run batch detections (if `yolo_video.py` is available)
	- python3 runDetections.py

Notes:
- This repo currently doesn’t include `yolo_video.py` (the main detection script). If you have it, drop it in the project root and ensure it supports flags like `--input`, `--output`, `--yolo`, and `--use-gpu`.
- `runDetections.py` will skip gracefully if `yolo_video.py` is missing.

## ATCS module

The `atcs.py` module defines a function `traffic_control(singleton)` that simulates signal phasing. It expects a `singleton` object providing:
- `get_count(lane_index: int) -> int`
- `reset_count(lane_index: int) -> None`

These methods are meant to be fed by an upstream detector (e.g., YOLO-based vehicle counting) while ATCS rotates through lanes, allocating green time dynamically.

## Project structure

traffic_management_system/
- atcs.py
- input_retrieval.py
- runDetections.py
- yolo-coco/
  - coco.names
  - yolov7-tiny.cfg
  - yolov7-tiny.weights (large; consider Git LFS)
- input_videos/
- example_gif/
- README.md
- requirements.txt
- .gitignore

## Git and large files

- Large binaries like `.weights` and videos are ignored by default (.gitignore). If you must version them, use Git LFS:
  - brew install git-lfs
  - git lfs install
  - git lfs track "*.weights" "*.mp4"

## Troubleshooting

- Module not found: Ensure your venv is activated and dependencies are installed.
- Missing `yolo_video.py`: Add your detection script to the root, or adjust `runDetections.py` to your own command.
- Paths: This project uses `input_videos/` and `output_videos/`. Adjust as needed.

## Roadmap / Ideas

- Add a proper `yolo_video.py` with counting hooks that update the ATCS singleton.
- Convert scripts into a small package with a CLI entry point.
- Add tests for ATCS logic and lane selection edge cases.
