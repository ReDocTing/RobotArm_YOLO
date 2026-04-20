"""
Camera-only tabletop grasp pose estimation pipeline.

This v1 pipeline intentionally stays on the perception side:
    RGB-D camera -> YOLO segmentation -> support plane -> top-grasp candidates

It does not import or call any robot interfaces. A later integration step can
transform the best camera-frame grasp into robot coordinates.
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

import cv2
import yaml
import numpy as np
from ultralytics import YOLO

try:
    import open3d as o3d
except ImportError:
    print("[ERROR] Missing open3d. Install it in the current environment first.")
    raise SystemExit(1)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEED_ROOT = PROJECT_ROOT.parent
for _path in (PROJECT_ROOT, SEEED_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from drivers.camera import make_camera
from utils.grasp_geometry import (
    SupportPlane,
    TopDownGraspResult,
    build_scene_cloud,
    create_virtual_gripper,
    estimate_support_plane,
    estimate_topdown_grasp_candidates,
    filter_object_cloud,
    mask_to_object_cloud,
    resolve_named_axis,
)


@dataclass
class GraspCandidate:
    class_name: str
    det_confidence: float
    grasp_score: float
    position: Optional[np.ndarray]
    pregrasp_position: Optional[np.ndarray]
    rotation: Optional[np.ndarray]
    euler_rpy: Optional[np.ndarray]
    tcp_rotation: Optional[np.ndarray]
    tcp_euler_rpy: Optional[np.ndarray]
    jaw_width_m: Optional[float]
    object_size_m: Optional[np.ndarray]
    point_count: int
    partial: bool
    rejected_reason: Optional[str]
    bbox_xyxy: tuple[int, int, int, int]
    center_px: tuple[int, int]
    candidate_label: str = "major"
    object_profile: str = "compact"
    contact_clearance_m: Optional[float] = None
    table_collision_risk: float = 0.0

    @property
    def is_valid(self) -> bool:
        return self.rejected_reason is None and self.position is not None and self.rotation is not None


_PALETTE = [
    (255, 80, 80),
    (80, 200, 80),
    (80, 120, 255),
    (255, 200, 0),
    (0, 220, 220),
    (200, 0, 200),
    (255, 140, 0),
    (160, 80, 240),
]


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _as_abs_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def ensure_segmentation_model(model_name: str, model: YOLO) -> None:
    task = str(getattr(model, "task", "") or "").lower()
    name = model_name.lower()
    if task == "segment" or "-seg" in name:
        return
    raise RuntimeError(
        f"Model '{model_name}' does not look like a segmentation model. "
        "Use a model with mask outputs, for example yoloe-26s-seg.pt or yoloe-26l-seg.pt."
    )


def load_hand_eye(project_root: Path, cam_type: str) -> tuple[Optional[np.ndarray], Optional[str]]:
    hand_eye_path = project_root / "config" / "calibration" / cam_type / "hand_eye.npz"
    if not hand_eye_path.exists():
        return None, None

    data = np.load(str(hand_eye_path), allow_pickle=False)
    T = data["T_result"].astype(np.float64)
    mode = str(data["mode"][0])
    return T, mode


def init_robot_pose_reader(cfg: dict[str, Any]):
    try:
        from drivers.robot.rebot_arm import RebotArm
    except Exception as exc:
        print(f"[HGGD] robot pose reader unavailable: {exc}")
        return None

    robot_cfg = cfg.get("robot", {})
    try:
        robot = RebotArm(
            config_path=robot_cfg.get("config_path"),
            urdf_path=robot_cfg.get("urdf_path"),
            repo_root=robot_cfg.get("repo_root"),
        )
        robot.connect(enable=False)
        print("[HGGD] robot pose reader connected (read-only)")
        return robot
    except Exception as exc:
        print(f"[HGGD] robot pose reader disabled: {exc}")
        return None


def compute_base_down_axis_camera(
    hand_eye_T: Optional[np.ndarray],
    hand_eye_mode: Optional[str],
    robot_reader,
) -> Optional[np.ndarray]:
    if hand_eye_T is None or hand_eye_mode is None:
        return None

    if hand_eye_mode == "eye_to_hand":
        R_cam2base = hand_eye_T[:3, :3]
    elif hand_eye_mode == "eye_in_hand":
        if robot_reader is None:
            return None
        try:
            T_gripper2base = robot_reader.get_tcp_pose()
        except Exception:
            return None
        R_cam2base = T_gripper2base[:3, :3] @ hand_eye_T[:3, :3]
    else:
        return None

    R_base2cam = R_cam2base.T
    axis = R_base2cam @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        return None
    return (axis / norm).astype(np.float32)


def project_px(point_xyz: np.ndarray, K: np.ndarray) -> tuple[int, int]:
    z = float(point_xyz[2])
    if abs(z) < 1e-6:
        return int(K[0, 2]), int(K[1, 2])
    u = int(round(float(point_xyz[0]) * float(K[0, 0]) / z + float(K[0, 2])))
    v = int(round(float(point_xyz[1]) * float(K[1, 1]) / z + float(K[1, 2])))
    return u, v


def mask_to_fullres(mask_tensor: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    image_h, image_w = image_shape
    if mask_tensor.shape[:2] != (image_h, image_w):
        mask_tensor = cv2.resize(mask_tensor, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
    return (mask_tensor > 0.5).astype(np.uint8)


def mask_touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def erode_mask(mask: np.ndarray, erode_px: int) -> np.ndarray:
    if erode_px <= 0:
        return mask
    kernel_size = max(1, int(erode_px) * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded if int(eroded.sum()) > 32 else mask


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.28) -> None:
    color_layer = np.zeros_like(image)
    color_layer[mask > 0] = color
    cv2.addWeighted(color_layer, alpha, image, 1.0, 0.0, image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, 1, cv2.LINE_AA)


def draw_candidate(
    image: np.ndarray,
    candidate: GraspCandidate,
    K: np.ndarray,
    color: tuple[int, int, int],
    rank: Optional[int] = None,
    highlight: bool = False,
) -> None:
    x1, y1, x2, y2 = candidate.bbox_xyxy
    box_thickness = 3 if highlight else 1
    cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

    if candidate.is_valid:
        center_uv = project_px(candidate.position, K)
        cv2.circle(image, center_uv, 7 if highlight else 5, color, -1)
        cv2.circle(image, center_uv, 7 if highlight else 5, (255, 255, 255), 1)

        axis_len = 0.05 if highlight else 0.04
        axis_colors = [(0, 0, 255), (0, 255, 0), (255, 140, 0)]
        for axis_index in range(3):
            end_point = candidate.position + candidate.rotation[:, axis_index] * axis_len
            end_uv = project_px(end_point, K)
            cv2.arrowedLine(image, center_uv, end_uv, axis_colors[axis_index], 2, tipLength=0.25)

        label_prefix = f"#{rank} " if rank is not None else ""
        lines = [
            f"{label_prefix}{candidate.class_name}:{candidate.candidate_label} s={candidate.grasp_score:.2f}",
            f"w={candidate.jaw_width_m * 100:.1f}cm pts={candidate.point_count} {candidate.object_profile}",
            f"xyz=({candidate.position[0]:+.3f},{candidate.position[1]:+.3f},{candidate.position[2]:+.3f})",
        ]
        if candidate.table_collision_risk > 0.0:
            lines.append(f"risk={candidate.table_collision_risk:.2f}")
        if candidate.partial:
            lines.append("partial mask")
    else:
        lines = [
            f"{candidate.class_name}:{candidate.candidate_label} det={candidate.det_confidence:.2f}",
            candidate.rejected_reason or "rejected",
        ]
        if candidate.object_profile:
            lines.append(candidate.object_profile)

    line_height = 16
    origin_y = max(12, y1 - line_height * len(lines) - 4)
    width = max(len(line) for line in lines) * 8 + 10
    cv2.rectangle(image, (x1, origin_y), (x1 + width, origin_y + line_height * len(lines) + 4), (0, 0, 0), -1)
    for idx, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x1 + 4, origin_y + (idx + 1) * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color if idx == 0 else (220, 220, 220),
            1,
            cv2.LINE_AA,
        )


def serialize_candidate(candidate: GraspCandidate) -> dict[str, Any]:
    return {
        "class_name": candidate.class_name,
        "candidate_label": candidate.candidate_label,
        "object_profile": candidate.object_profile,
        "det_confidence": round(float(candidate.det_confidence), 6),
        "grasp_score": round(float(candidate.grasp_score), 6),
        "position": None if candidate.position is None else [round(float(v), 6) for v in candidate.position.tolist()],
        "pregrasp_position": None if candidate.pregrasp_position is None else [round(float(v), 6) for v in candidate.pregrasp_position.tolist()],
        "rotation": None if candidate.rotation is None else [[round(float(v), 6) for v in row] for row in candidate.rotation.tolist()],
        "euler_rpy": None if candidate.euler_rpy is None else [round(float(v), 6) for v in candidate.euler_rpy.tolist()],
        "tcp_rotation": None if candidate.tcp_rotation is None else [[round(float(v), 6) for v in row] for row in candidate.tcp_rotation.tolist()],
        "tcp_euler_rpy": None if candidate.tcp_euler_rpy is None else [round(float(v), 6) for v in candidate.tcp_euler_rpy.tolist()],
        "jaw_width_m": None if candidate.jaw_width_m is None else round(float(candidate.jaw_width_m), 6),
        "object_size_m": None if candidate.object_size_m is None else [round(float(v), 6) for v in candidate.object_size_m.tolist()],
        "contact_clearance_m": None if candidate.contact_clearance_m is None else round(float(candidate.contact_clearance_m), 6),
        "table_collision_risk": round(float(candidate.table_collision_risk), 6),
        "point_count": int(candidate.point_count),
        "partial": bool(candidate.partial),
        "rejected_reason": candidate.rejected_reason,
        "bbox_xyxy": [int(v) for v in candidate.bbox_xyxy],
        "center_px": [int(candidate.center_px[0]), int(candidate.center_px[1])],
    }


def build_snapshot_record(
    frame_index: int,
    timestamp: str,
    cfg: dict[str, Any],
    model_name: str,
    support_plane: Optional[SupportPlane],
    candidates: list[GraspCandidate],
) -> dict[str, Any]:
    return {
        "frame_index": int(frame_index),
        "timestamp": timestamp,
        "camera_type": cfg.get("camera", {}).get("type"),
        "model_name": model_name,
        "support_plane_normal": None if support_plane is None else [round(float(v), 6) for v in support_plane.normal.tolist()],
        "support_plane_model": None if support_plane is None else [round(float(v), 6) for v in support_plane.model.tolist()],
        "candidates": [serialize_candidate(candidate) for candidate in candidates],
    }


def build_open3d_debug_view(
    scene_points: np.ndarray,
    object_clouds: list[tuple[np.ndarray, tuple[int, int, int]]],
    candidates: list[GraspCandidate],
    output_cfg: dict[str, Any],
) -> None:
    if not output_cfg.get("snapshot_open3d", True):
        return

    flip = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    geometries: list[Any] = []

    if len(scene_points) > 0:
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(scene_points.astype(np.float64))
        scene_pcd.paint_uniform_color([0.55, 0.55, 0.55])
        scene_pcd = scene_pcd.voxel_down_sample(voxel_size=0.004)
        scene_pcd.transform(flip)
        geometries.append(scene_pcd)

    for points, color in object_clouds:
        if len(points) == 0:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.paint_uniform_color([channel / 255.0 for channel in color[::-1]])
        pcd.transform(flip)
        geometries.append(pcd)

    for idx, candidate in enumerate(candidates):
        if not candidate.is_valid:
            continue
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = candidate.rotation.astype(np.float64)
        transform[:3, 3] = candidate.position.astype(np.float64)
        grip_color = (1.0, 0.0, 0.0) if idx == 0 else (0.95, 0.45, 0.05)
        geometries.append(
            create_virtual_gripper(
                flip @ transform,
                width=float(candidate.jaw_width_m),
                depth=0.06,
                color=grip_color,
            )
        )

    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    o3d.visualization.draw_geometries(geometries, window_name="HGGD Grasp Debug")


def run_yolo(
    model: YOLO,
    color_bgr: np.ndarray,
    device: str,
    conf: float,
    iou: float,
) -> list[Any]:
    return model.predict(color_bgr, verbose=False, device=device, conf=conf, iou=iou)


def extract_candidates(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    results: list[Any],
    support_plane: Optional[SupportPlane],
    gp_cfg: dict[str, Any],
    preferred_approach_axis: Optional[np.ndarray] = None,
) -> tuple[list[GraspCandidate], list[tuple[np.ndarray, tuple[int, int, int]]]]:
    candidates: list[GraspCandidate] = []
    object_clouds: list[tuple[np.ndarray, tuple[int, int, int]]] = []

    if not results:
        return candidates, object_clouds

    grasp_cfg = gp_cfg.get("grasp", {})
    min_depth_m = float(gp_cfg.get("min_depth_m", 0.05))
    max_depth_m = float(gp_cfg.get("max_depth_m", 1.2))
    min_points = int(gp_cfg.get("min_points", 120))
    support_cfg = gp_cfg.get("support_plane", {})
    preferred_axis = resolve_named_axis(grasp_cfg.get("preferred_axis", "camera_x"))
    mask_erode_px = int(grasp_cfg.get("profile_mask_erode_px", 3))

    image_h, image_w = color_bgr.shape[:2]
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        has_masks = result.masks is not None and len(result.masks.data) == len(result.boxes)
        for det_index, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0])
            class_name = result.names.get(cls_id, str(cls_id))
            det_conf = float(box.conf[0])
            center_px = ((x1 + x2) // 2, (y1 + y2) // 2)
            bbox_w = max(1, x2 - x1)
            bbox_h = max(1, y2 - y1)
            image_aspect_ratio = float(bbox_h) / float(bbox_w)
            color = _PALETTE[len(object_clouds) % len(_PALETTE)]
            appended_candidate = False

            mask = None
            partial = False
            rejected_reason = None
            filtered_points = np.empty((0, 3), dtype=np.float32)
            grasp_result = TopDownGraspResult(
                position=None,
                pregrasp_position=None,
                rotation=None,
                euler_rpy=None,
                tcp_rotation=None,
                tcp_euler_rpy=None,
                jaw_width_m=None,
                object_size_m=None,
                grasp_score=0.0,
                point_count=0,
                rejected_reason="unknown",
                candidate_label="major",
                object_profile="compact",
                contact_clearance_m=None,
                table_collision_risk=1.0,
            )

            if not has_masks:
                rejected_reason = "no_mask_output"
            elif support_plane is None:
                rejected_reason = "plane_not_found"
            else:
                mask = mask_to_fullres(result.masks.data[det_index].cpu().numpy(), (image_h, image_w))
                partial = mask_touches_border(mask)
                estimation_mask = erode_mask(mask, mask_erode_px)
                raw_points = mask_to_object_cloud(
                    mask=estimation_mask,
                    depth_mm=depth_mm,
                    K=K,
                    min_depth_m=min_depth_m,
                    max_depth_m=max_depth_m,
                )
                filtered_points = filter_object_cloud(
                    points=raw_points,
                    plane=support_plane,
                    object_clearance_m=float(support_cfg.get("object_clearance_m", 0.006)),
                    voxel_size_m=float(gp_cfg.get("voxel_size_m", 0.003)),
                    outlier_nb_neighbors=int(gp_cfg.get("outlier_nb_neighbors", 20)),
                    outlier_std_ratio=float(gp_cfg.get("outlier_std_ratio", 2.0)),
                )
                if len(filtered_points) < min_points:
                    rejected_reason = "sparse_object_cloud"
                else:
                    grasp_results = estimate_topdown_grasp_candidates(
                        points=filtered_points,
                        plane=support_plane,
                        det_confidence=det_conf,
                        partial=partial,
                        max_width_m=float(grasp_cfg.get("max_width_m", 0.085)),
                        width_percentile_low=float(grasp_cfg.get("width_percentile_low", 5)),
                        width_percentile_high=float(grasp_cfg.get("width_percentile_high", 95)),
                        pregrasp_offset_m=float(grasp_cfg.get("pregrasp_offset_m", 0.08)),
                        grasp_height_ratio=float(grasp_cfg.get("grasp_height_ratio", 0.5)),
                        min_height_above_plane_m=float(grasp_cfg.get("min_height_above_plane_m", 0.012)),
                        candidates_per_object=int(grasp_cfg.get("candidates_per_object", 2)),
                        allow_minor_axis_grasp=bool(grasp_cfg.get("allow_minor_axis_grasp", True)),
                        preferred_axis=preferred_axis,
                        preferred_axis_target=str(grasp_cfg.get("preferred_axis_target", "open_axis")),
                        preferred_axis_weight=float(grasp_cfg.get("preferred_axis_weight", 0.18)),
                        extent_percentile_low=float(grasp_cfg.get("extent_percentile_low", 10)),
                        extent_percentile_high=float(grasp_cfg.get("extent_percentile_high", 90)),
                        flat_object_max_height_m=float(grasp_cfg.get("flat_object_max_height_m", 0.045)),
                        flat_object_max_height_ratio=float(grasp_cfg.get("flat_object_max_height_ratio", 0.30)),
                        tall_object_min_height_m=float(grasp_cfg.get("tall_object_min_height_m", 0.070)),
                        tall_object_min_height_ratio=float(grasp_cfg.get("tall_object_min_height_ratio", 0.60)),
                        flat_grasp_height_ratio=float(grasp_cfg.get("flat_grasp_height_ratio", 0.82)),
                        tall_grasp_height_ratio=float(grasp_cfg.get("tall_grasp_height_ratio", 0.55)),
                        min_contact_clearance_m=float(grasp_cfg.get("min_contact_clearance_m", 0.024)),
                        low_profile_reject_height_m=float(grasp_cfg.get("low_profile_reject_height_m", 0.018)),
                        low_profile_min_contact_clearance_m=float(
                            grasp_cfg.get("low_profile_min_contact_clearance_m", 0.024)
                        ),
                        collision_penalty_weight=float(grasp_cfg.get("collision_penalty_weight", 0.22)),
                        image_aspect_ratio=image_aspect_ratio,
                        flat_image_aspect_ratio=float(grasp_cfg.get("flat_image_aspect_ratio", 0.78)),
                        tall_image_aspect_ratio=float(grasp_cfg.get("tall_image_aspect_ratio", 1.15)),
                        elongated_planar_aspect_ratio=float(grasp_cfg.get("elongated_planar_aspect_ratio", 1.8)),
                        preferred_approach_axis=preferred_approach_axis,
                        prefer_base_down_for_flat=bool(grasp_cfg.get("prefer_base_down_for_flat", True)),
                        prefer_base_down_profiles=grasp_cfg.get(
                            "prefer_base_down_profiles",
                            ["flat", "compact", "tall"],
                        ),
                        base_down_max_tilt_deg=float(grasp_cfg.get("base_down_max_tilt_deg", 35.0)),
                    )
                    valid_grasps = [item for item in grasp_results if item.rejected_reason is None]
                    if valid_grasps:
                        for grasp_result in valid_grasps:
                            candidates.append(
                                GraspCandidate(
                                    class_name=class_name,
                                    det_confidence=det_conf,
                                    grasp_score=float(grasp_result.grasp_score),
                                    position=grasp_result.position,
                                    pregrasp_position=grasp_result.pregrasp_position,
                                    rotation=grasp_result.rotation,
                                    euler_rpy=grasp_result.euler_rpy,
                                    tcp_rotation=grasp_result.tcp_rotation,
                                    tcp_euler_rpy=grasp_result.tcp_euler_rpy,
                                    jaw_width_m=grasp_result.jaw_width_m,
                                    object_size_m=grasp_result.object_size_m,
                                    point_count=int(grasp_result.point_count or len(filtered_points)),
                                    partial=partial,
                                    rejected_reason=None,
                                    bbox_xyxy=(x1, y1, x2, y2),
                                    center_px=center_px,
                                    candidate_label=grasp_result.candidate_label,
                                    object_profile=grasp_result.object_profile,
                                    contact_clearance_m=grasp_result.contact_clearance_m,
                                    table_collision_risk=float(grasp_result.table_collision_risk),
                                )
                            )
                        appended_candidate = True
                    elif grasp_results:
                        grasp_result = grasp_results[0]
                        rejected_reason = grasp_result.rejected_reason
                        candidates.append(
                            GraspCandidate(
                                class_name=class_name,
                                det_confidence=det_conf,
                                grasp_score=float(grasp_result.grasp_score),
                                position=grasp_result.position,
                                pregrasp_position=grasp_result.pregrasp_position,
                                rotation=grasp_result.rotation,
                                euler_rpy=grasp_result.euler_rpy,
                                tcp_rotation=grasp_result.tcp_rotation,
                                tcp_euler_rpy=grasp_result.tcp_euler_rpy,
                                jaw_width_m=grasp_result.jaw_width_m,
                                object_size_m=grasp_result.object_size_m,
                                point_count=int(grasp_result.point_count or len(filtered_points)),
                                partial=partial,
                                rejected_reason=rejected_reason,
                                bbox_xyxy=(x1, y1, x2, y2),
                                center_px=center_px,
                                candidate_label=grasp_result.candidate_label,
                                object_profile=grasp_result.object_profile,
                                contact_clearance_m=grasp_result.contact_clearance_m,
                                table_collision_risk=float(grasp_result.table_collision_risk),
                            )
                        )
                        appended_candidate = True

            if rejected_reason is not None and not appended_candidate:
                candidates.append(
                    GraspCandidate(
                        class_name=class_name,
                        det_confidence=det_conf,
                        grasp_score=float(grasp_result.grasp_score),
                        position=grasp_result.position,
                        pregrasp_position=grasp_result.pregrasp_position,
                        rotation=grasp_result.rotation,
                        euler_rpy=grasp_result.euler_rpy,
                        tcp_rotation=grasp_result.tcp_rotation,
                        tcp_euler_rpy=grasp_result.tcp_euler_rpy,
                        jaw_width_m=grasp_result.jaw_width_m,
                        object_size_m=grasp_result.object_size_m,
                        point_count=int(grasp_result.point_count or len(filtered_points)),
                        partial=partial,
                        rejected_reason=rejected_reason,
                        bbox_xyxy=(x1, y1, x2, y2),
                        center_px=center_px,
                        candidate_label=grasp_result.candidate_label,
                        object_profile=grasp_result.object_profile,
                        contact_clearance_m=grasp_result.contact_clearance_m,
                        table_collision_risk=float(grasp_result.table_collision_risk),
                    )
                )

            if len(filtered_points) > 0:
                object_clouds.append((filtered_points, color))

    candidates.sort(
        key=lambda item: (item.is_valid, item.grasp_score, item.det_confidence),
        reverse=True,
    )
    return candidates, object_clouds


def render_display(
    base_image: np.ndarray,
    candidates: list[GraspCandidate],
    results: list[Any],
    K: np.ndarray,
    live_top_k: int,
    support_plane: Optional[SupportPlane],
    status_text: str,
) -> np.ndarray:
    display = base_image.copy()

    overlay_count = 0
    if results:
        image_h, image_w = display.shape[:2]
        for result in results:
            has_masks = result.masks is not None and result.boxes is not None and len(result.masks.data) == len(result.boxes)
            if not has_masks:
                continue
            for idx in range(len(result.boxes)):
                mask = mask_to_fullres(result.masks.data[idx].cpu().numpy(), (image_h, image_w))
                overlay_mask(display, mask, _PALETTE[overlay_count % len(_PALETTE)], alpha=0.18)
                overlay_count += 1

    valid_candidates = [candidate for candidate in candidates if candidate.is_valid]
    top_valid = valid_candidates[: max(1, int(live_top_k))]
    highlight_ids = {id(candidate): rank for rank, candidate in enumerate(top_valid, start=1)}

    invalid_candidates = [candidate for candidate in candidates if not candidate.is_valid]
    drawn_candidates = invalid_candidates + top_valid

    for idx, candidate in enumerate(drawn_candidates):
        color = _PALETTE[idx % len(_PALETTE)]
        rank = highlight_ids.get(id(candidate))
        draw_candidate(
            image=display,
            candidate=candidate,
            K=K,
            color=color,
            rank=rank,
            highlight=rank == 1,
        )

    cv2.putText(display, status_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    if support_plane is not None:
        plane_text = (
            f"plane=({support_plane.normal[0]:+.2f},{support_plane.normal[1]:+.2f},{support_plane.normal[2]:+.2f}) "
            f"inliers={support_plane.inlier_count}"
        )
        cv2.putText(display, plane_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 255), 1, cv2.LINE_AA)

    if valid_candidates:
        best = valid_candidates[0]
        best_text = (
            f"best={best.class_name}:{best.candidate_label}/{best.object_profile} score={best.grasp_score:.2f} "
            f"width={best.jaw_width_m * 100:.1f}cm risk={best.table_collision_risk:.2f} "
            f"xyz=({best.position[0]:+.3f},{best.position[1]:+.3f},{best.position[2]:+.3f})"
        )
        cv2.putText(
            display,
            best_text,
            (10, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (120, 255, 140),
            2,
            cv2.LINE_AA,
        )

    return display


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera-only tabletop grasp estimation pipeline")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", default=None, help="Override YOLO model name from the config")
    parser.add_argument("--device", default=None, help="Override YOLO device from the config")
    parser.add_argument("--conf", type=float, default=None, help="Override YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="Override YOLO IoU threshold")
    parser.add_argument("--save-dir", default=None, help="Override snapshot save directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)

    yolo_cfg = cfg.get("yolo", {})
    model_name = args.model or yolo_cfg.get("model_name", "yoloe-26s-seg.pt")
    device = args.device or yolo_cfg.get("device", "cpu")
    conf = float(args.conf if args.conf is not None else cfg.get("detection", {}).get("conf_threshold", 0.25))
    iou = float(args.iou if args.iou is not None else cfg.get("detection", {}).get("iou_threshold", 0.45))
    use_world = bool(yolo_cfg.get("use_world", True))
    custom_classes = list(yolo_cfg.get("custom_classes", ["bottle", "cup", "book"]))

    model_path = PROJECT_ROOT / "models" / model_name
    print(f"=== Loading YOLO: {model_name} ===")
    model = YOLO(str(model_path))
    if use_world and ("world" in model_name.lower() or "yoloe" in model_name.lower()):
        model.set_classes(custom_classes)
        print(f"  Classes: {custom_classes}")
    ensure_segmentation_model(model_name, model)

    gp_cfg = cfg.get("grasp_pipeline", {})
    grasp_cfg = gp_cfg.get("grasp", {})
    support_cfg = gp_cfg.get("support_plane", {})
    output_cfg = gp_cfg.get("output", {})
    save_dir_value = args.save_dir or output_cfg.get("save_dir", "data/grasp_snapshots")
    save_dir = _as_abs_path(PROJECT_ROOT, str(save_dir_value))

    print(f"=== Camera: {cfg.get('camera', {}).get('type')} ===")
    cam = make_camera(cfg)
    cam.open()
    cam.warm_up(15)
    K = cam.K.astype(np.float32)

    cam_type = str(cfg.get("camera", {}).get("type", "")).lower()
    prefer_base_down_for_flat = bool(grasp_cfg.get("prefer_base_down_for_flat", True))
    prefer_base_down_profiles = grasp_cfg.get(
        "prefer_base_down_profiles",
        ["flat", "compact", "tall"],
    )
    hand_eye_T, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
    robot_reader = None
    base_down_enabled = prefer_base_down_for_flat or bool(prefer_base_down_profiles)
    if base_down_enabled and hand_eye_T is not None:
        if hand_eye_mode == "eye_in_hand":
            robot_reader = init_robot_pose_reader(cfg)
        print(
            "[HGGD] base-down approach guidance "
            f"{'enabled' if (hand_eye_mode == 'eye_to_hand' or robot_reader is not None) else 'fallback-only'} "
            f"(mode={hand_eye_mode})"
        )
    elif base_down_enabled:
        print("[HGGD] base-down approach guidance unavailable: missing hand-eye calibration")

    # Future integration point:
    # The best camera-frame candidate can later be transformed via cam_to_robot()
    # and handed off to the grasp.py state machine using tcp_rotation / tcp_euler_rpy.
    # This v1 script keeps that path disabled.

    window_name = "HGGD Grasp Pipeline"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("\n[Keys]  S=snapshot+save  R=resume live  Q/ESC=quit\n")

    frame_index = 0
    fps_counter = 0
    fps_timer = time.perf_counter()
    fps_value = 0.0

    last_results: list[Any] = []
    last_candidates: list[GraspCandidate] = []
    last_support_plane: Optional[SupportPlane] = None
    last_display = None
    last_scene_points = np.empty((0, 3), dtype=np.float32)
    last_object_clouds: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    frozen = False

    plane_refresh_frames = max(1, int(support_cfg.get("plane_refresh_frames", 15)))
    infer_every_live = max(1, int(gp_cfg.get("infer_every_live", 3)))
    min_depth_m = float(gp_cfg.get("min_depth_m", 0.05))
    max_depth_m = float(gp_cfg.get("max_depth_m", 1.20))
    live_top_k = int(output_cfg.get("live_top_k", 3))

    try:
        while True:
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                continue

            frame_index += 1
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                fps_value = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            should_infer = (frame_index % infer_every_live == 0) or not last_results
            if should_infer:
                preferred_approach_axis = compute_base_down_axis_camera(
                    hand_eye_T=hand_eye_T,
                    hand_eye_mode=hand_eye_mode,
                    robot_reader=robot_reader,
                )
                last_results = run_yolo(model, color_bgr, device=device, conf=conf, iou=iou)
                refresh_plane = last_support_plane is None or frame_index % plane_refresh_frames == 0
                if refresh_plane:
                    last_scene_points = build_scene_cloud(
                        depth_mm=depth_mm,
                        K=K,
                        min_depth_m=min_depth_m,
                        max_depth_m=max_depth_m,
                        stride=2,
                    )
                    last_support_plane = estimate_support_plane(
                        scene_points=last_scene_points,
                        distance_threshold_m=float(support_cfg.get("distance_threshold_m", 0.008)),
                        ransac_n=int(support_cfg.get("ransac_n", 3)),
                        num_iterations=int(support_cfg.get("num_iterations", 120)),
                    )
                last_candidates, last_object_clouds = extract_candidates(
                    color_bgr=color_bgr,
                    depth_mm=depth_mm,
                    K=K,
                    results=last_results,
                    support_plane=last_support_plane,
                    gp_cfg=gp_cfg,
                    preferred_approach_axis=preferred_approach_axis,
                )

            status = f"LIVE {fps_value:.1f}fps | {model_name} | S=snapshot R=resume Q=quit"
            if frozen and last_display is not None:
                display = last_display.copy()
                cv2.putText(display, "[FROZEN]", (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2, cv2.LINE_AA)
            else:
                display = render_display(
                    base_image=color_bgr,
                    candidates=last_candidates,
                    results=last_results,
                    K=K,
                    live_top_k=live_top_k,
                    support_plane=last_support_plane,
                    status_text=status,
                )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                frozen = False
                last_display = None
                print("[Resume] live preview")
                continue

            if key in (ord("s"), ord("S")):
                print("\n[Snapshot] recomputing full-quality grasp candidates...")
                preferred_approach_axis = compute_base_down_axis_camera(
                    hand_eye_T=hand_eye_T,
                    hand_eye_mode=hand_eye_mode,
                    robot_reader=robot_reader,
                )
                scene_points = build_scene_cloud(
                    depth_mm=depth_mm,
                    K=K,
                    min_depth_m=min_depth_m,
                    max_depth_m=max_depth_m,
                    stride=1,
                )
                support_plane = estimate_support_plane(
                    scene_points=scene_points,
                    distance_threshold_m=float(support_cfg.get("distance_threshold_m", 0.008)),
                    ransac_n=int(support_cfg.get("ransac_n", 3)),
                    num_iterations=int(support_cfg.get("num_iterations", 120)),
                )
                snapshot_results = run_yolo(model, color_bgr, device=device, conf=conf, iou=iou)
                snapshot_candidates, snapshot_object_clouds = extract_candidates(
                    color_bgr=color_bgr,
                    depth_mm=depth_mm,
                    K=K,
                    results=snapshot_results,
                    support_plane=support_plane,
                    gp_cfg=gp_cfg,
                    preferred_approach_axis=preferred_approach_axis,
                )
                snapshot_display = render_display(
                    base_image=color_bgr,
                    candidates=snapshot_candidates,
                    results=snapshot_results,
                    K=K,
                    live_top_k=live_top_k,
                    support_plane=support_plane,
                    status_text=f"SNAPSHOT | {model_name}",
                )

                timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir.mkdir(parents=True, exist_ok=True)
                image_path = save_dir / f"hggd_grasp_{timestamp}.png"
                json_path = save_dir / f"hggd_grasp_{timestamp}.json"
                cv2.imwrite(str(image_path), snapshot_display)
                snapshot_record = build_snapshot_record(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    cfg=cfg,
                    model_name=model_name,
                    support_plane=support_plane,
                    candidates=snapshot_candidates,
                )
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(snapshot_record, f, indent=2, ensure_ascii=False)

                print(f"  saved {image_path.name}")
                print(f"  saved {json_path.name}")
                valid_candidates = [candidate for candidate in snapshot_candidates if candidate.is_valid]
                if valid_candidates:
                    best = valid_candidates[0]
                    print(
                        f"  best={best.class_name}:{best.candidate_label}/{best.object_profile} score={best.grasp_score:.3f} "
                        f"risk={best.table_collision_risk:.2f} "
                        f"xyz=({best.position[0]:+.3f},{best.position[1]:+.3f},{best.position[2]:+.3f}) "
                        f"width={best.jaw_width_m * 100:.1f}cm"
                    )
                else:
                    print("  no valid candidates in this snapshot")

                build_open3d_debug_view(
                    scene_points=scene_points,
                    object_clouds=snapshot_object_clouds,
                    candidates=[candidate for candidate in snapshot_candidates if candidate.is_valid],
                    output_cfg=output_cfg,
                )

                frozen = True
                last_display = snapshot_display
                last_support_plane = support_plane
                last_scene_points = scene_points
                last_candidates = snapshot_candidates
                last_results = snapshot_results
                last_object_clouds = snapshot_object_clouds

    finally:
        if robot_reader is not None:
            try:
                robot_reader.disconnect()
            except Exception:
                pass
        cam.close()
        cv2.destroyAllWindows()
        print("\nExiting camera-only grasp pipeline.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
