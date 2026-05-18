#!/usr/bin/env python3
"""列出已连接的 RealSense，并逐个预览 5 秒，便于区分臂载 / 墙上相机。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def list_devices() -> list[tuple[str, str]]:
    import pyrealsense2 as rs

    ctx = rs.context()
    out: list[tuple[str, str]] = []
    for i in range(ctx.query_devices().size()):
        dev = ctx.query_devices()[i]
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        out.append((name, serial))
    return out


def preview_serial(serial: str, seconds: float, width: int, height: int, fps: int) -> None:
    import yaml

    from drivers.camera import make_camera

    cfg_path = PROJECT_ROOT / "config" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("camera", {})
    cfg["camera"]["type"] = "realsense_d435"
    cfg["camera"]["serial"] = serial
    cfg["camera"]["color_width"] = width
    cfg["camera"]["color_height"] = height
    cfg["camera"]["fps"] = fps

    cam = make_camera(cfg)
    cam.open()
    win = f"RealSense {serial}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    t_end = time.time() + seconds
    try:
        while time.time() < t_end:
            color, _ = cam.get_frame()
            if color is None:
                continue
            cv2.putText(
                color,
                f"serial={serial}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow(win, color)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cam.close()
        cv2.destroyWindow(win)


def main() -> None:
    parser = argparse.ArgumentParser(description="RealSense 设备枚举与预览")
    parser.add_argument("--preview-seconds", type=float, default=5.0, help="每台相机预览秒数")
    parser.add_argument("--serial", type=str, default=None, help="只预览指定序列号")
    parser.add_argument("--no-preview", action="store_true", help="仅打印列表")
    args = parser.parse_args()

    devices = list_devices()
    if not devices:
        print("未发现 RealSense 设备")
        sys.exit(1)

    print("已连接 RealSense:")
    for idx, (name, serial) in enumerate(devices):
        print(f"  [{idx}] {name}  serial={serial}")

    print(
        "\n请将机械臂末端相机的 serial 写入 config/default.yaml -> camera.serial\n"
        "墙上固定相机写入 camera.wall_serial（主抓取流程目前只用臂载相机）"
    )

    if args.no_preview:
        return

    targets = devices
    if args.serial:
        targets = [(n, s) for n, s in devices if s == args.serial]
        if not targets:
            print(f"未找到 serial={args.serial}")
            sys.exit(1)

    cfg_path = PROJECT_ROOT / "config" / "default.yaml"
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cam_cfg = cfg.get("camera", {})
    w = cam_cfg.get("color_width", 1280)
    h = cam_cfg.get("color_height", 720)
    fps = cam_cfg.get("fps", 30)

    for name, serial in targets:
        print(f"\n预览 {name} ({serial})，按 Q 可提前结束 …")
        preview_serial(serial, args.preview_seconds, w, h, fps)


if __name__ == "__main__":
    main()
