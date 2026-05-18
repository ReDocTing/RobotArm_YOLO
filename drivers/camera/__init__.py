from __future__ import annotations

from pathlib import Path

from .base import CameraDriver
from .orbbec_gemini2 import OrbbecGemini2
from .realsense import RealsenseCamera

__all__ = ["CameraDriver", "OrbbecGemini2", "RealsenseCamera", "make_camera"]


def make_camera(cfg: dict) -> CameraDriver:
    """根据 config/default.yaml 实例化对应的相机驱动。

    读取 cfg["camera"]["type"] 决定驱动类型，标定目录自动指向
    config/calibration/{cam_type}/。

    Args:
        cfg: 已加载的 YAML 配置字典

    Returns:
        未打开的 CameraDriver 实例；调用方需自行调用 .open()

    Raises:
        ValueError: cam_type 不在已支持列表中
    """
    cam_cfg  = cfg.get("camera", {})
    cam_type = cam_cfg.get("type", "").lower()
    w   = cam_cfg.get("color_width",  1280)
    h   = cam_cfg.get("color_height", 720)
    fps = cam_cfg.get("fps", 30)

    # cameraws/ 项目根目录（__file__ 在 drivers/camera/ 下，上溯两级）
    _root     = Path(__file__).resolve().parent.parent.parent
    calib_dir = str(_root / "config" / "calibration" / cam_type)

    serial = cam_cfg.get("serial")

    if "orbbec" in cam_type:
        return OrbbecGemini2(w, h, fps, calib_dir=calib_dir)
    elif "realsense" in cam_type:
        return RealsenseCamera(w, h, fps, calib_dir=calib_dir, serial=serial)
    else:
        raise ValueError(
            f"不支持的相机类型: {cam_type!r}\n"
            f"请在 config/default.yaml 中将 camera.type 设为:\n"
            f"  orbbec_gemini2 | realsense_d435i | realsense_d405"
        )
