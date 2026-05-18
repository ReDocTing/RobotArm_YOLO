"""ArUco 标记检测与位姿估计。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


def _marker_object_points(marker_length_m: float) -> np.ndarray:
    """ArUco 平面四角 3D 坐标（标记坐标系，中心为原点）。"""
    h = marker_length_m * 0.5
    return np.array(
        [[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
        dtype=np.float64,
    )


def _estimate_marker_pose(
    corner: np.ndarray,
    marker_length_m: float,
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """估计单标记位姿；兼容 OpenCV 4.7+（无 estimatePoseSingleMarkers）。"""
    img_pts = np.asarray(corner, dtype=np.float64).reshape(-1, 2)
    obj_pts = _marker_object_points(marker_length_m)

    if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corner], marker_length_m, K, D
        )
        if rvecs is None:
            return None, None
        return rvecs[0, 0], tvecs[0, 0]

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        K,
        D,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None, None
    return rvec.flatten(), tvec.flatten()


@dataclass
class MarkerPose:
    """单个 ArUco 标记的检测结果。"""
    id: int
    T_marker2cam: np.ndarray   # (4, 4) 标记到相机坐标系的变换


class ArUcoDetector:
    """ArUco 标记检测器，返回最接近相机的目标标记的位姿。

    Args:
        marker_length_m: 标记实际边长（米，黑色边框外沿）
        aruco_dict_id:   cv2.aruco 字典 ID，默认 DICT_4X4_50 = 0
        target_marker_id: 指定只检测该 ID 的标记；None = 使用检测到的第一个
    """

    def __init__(
        self,
        marker_length_m: float = 0.05,
        aruco_dict_id: int = 0,
        target_marker_id: Optional[int] = None,
    ) -> None:
        self._length = marker_length_m
        self._tid = target_marker_id
        self._dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self._params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._dict, self._params)

    def detect(
        self,
        bgr: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
    ) -> Optional[MarkerPose]:
        """
        在图像中检测 ArUco 标记并返回位姿。

        Args:
            bgr: BGR 图像 (H, W, 3)
            K:   相机内参矩阵 (3, 3)
            D:   畸变系数 (1, N) 或 (N,)

        Returns:
            MarkerPose 或 None（未检测到）
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return None

        ids_flat = ids.flatten()

        # 筛选目标 ID
        if self._tid is not None:
            mask = ids_flat == self._tid
            if not np.any(mask):
                return None
            corners = [corners[i] for i in np.where(mask)[0]]
            ids_flat = ids_flat[mask]

        # 多标记时取最近（Z 最小）的
        best: Optional[MarkerPose] = None
        best_z = float("inf")

        for corner, mid in zip(corners, ids_flat):
            rvec, tvec = _estimate_marker_pose(corner, self._length, K, D)
            if rvec is None:
                continue
            z = float(tvec[2])
            if z < best_z:
                best_z = z
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3,  3] = tvec
                best = MarkerPose(id=int(mid), T_marker2cam=T)

        return best

    def draw_detected(
        self,
        bgr: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
        axis_length: float = 0.03,
    ) -> np.ndarray:
        """
        在图像上绘制检测到的所有 ArUco 标记（框 + 坐标轴）。

        Returns:
            带标注的 BGR 图像副本
        """
        vis = bgr.copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return vis

        cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        for corner in corners:
            rvec, tvec = _estimate_marker_pose(corner, self._length, K, D)
            if rvec is not None:
                cv2.drawFrameAxes(vis, K, D, rvec, tvec, axis_length)

        return vis
