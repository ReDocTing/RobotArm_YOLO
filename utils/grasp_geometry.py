"""Geometry helpers for tabletop top-grasp estimation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover - depends on local env
    raise RuntimeError("open3d is required for grasp geometry helpers") from exc

from .transforms import (
    grasp_axes_to_rebot_tcp_rotation,
    rotation_matrix_to_euler_zyx,
)


@dataclass
class SupportPlane:
    """Estimated support plane in camera coordinates."""

    model: np.ndarray          # (4,), normalized plane model ax + by + cz + d = 0
    normal: np.ndarray         # (3,), points toward free space / camera side
    origin: np.ndarray         # (3,), a point on the plane
    inlier_count: int


@dataclass
class TopDownGraspResult:
    """Result of tabletop top-grasp estimation for one object."""

    position: Optional[np.ndarray]
    pregrasp_position: Optional[np.ndarray]
    rotation: Optional[np.ndarray]
    euler_rpy: Optional[np.ndarray]
    tcp_rotation: Optional[np.ndarray]
    tcp_euler_rpy: Optional[np.ndarray]
    jaw_width_m: Optional[float]
    object_size_m: Optional[np.ndarray]
    grasp_score: float
    point_count: int
    rejected_reason: Optional[str]
    candidate_label: str = "major"
    orientation_score: float = 1.0
    object_profile: str = "compact"
    contact_clearance_m: Optional[float] = None
    table_collision_risk: float = 0.0


def build_scene_cloud(
    depth_mm: np.ndarray,
    K: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    stride: int = 2,
) -> np.ndarray:
    """Project a full depth frame to a camera-frame point cloud."""
    if depth_mm is None or depth_mm.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    depth_m = depth_mm.astype(np.float32) / 1000.0
    height, width = depth_m.shape
    stride = max(1, int(stride))

    ys = np.arange(0, height, stride, dtype=np.int32)
    xs = np.arange(0, width, stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    sampled_depth = depth_m[grid_y, grid_x]
    valid = np.isfinite(sampled_depth) & (sampled_depth > min_depth_m) & (sampled_depth < max_depth_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    z = sampled_depth[valid]
    u = grid_x[valid].astype(np.float32)
    v = grid_y[valid].astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


def estimate_support_plane(
    scene_points: np.ndarray,
    distance_threshold_m: float,
    ransac_n: int,
    num_iterations: int,
) -> Optional[SupportPlane]:
    """Estimate the dominant support plane and orient it toward the camera."""
    if len(scene_points) < 128:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_points.astype(np.float64))
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=float(distance_threshold_m),
        ransac_n=int(ransac_n),
        num_iterations=int(num_iterations),
    )
    if len(inliers) < 64:
        return None

    model = np.asarray(plane_model, dtype=np.float64)
    normal = model[:3]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return None

    model /= norm
    normal = model[:3]
    inlier_points = scene_points[np.asarray(inliers, dtype=np.int32)]
    origin = inlier_points.mean(axis=0).astype(np.float64)

    # Orient the support-plane normal toward the camera / free-space side.
    if float(np.dot(normal, -origin)) < 0.0:
        model = -model
        normal = -normal

    origin = (origin - (np.dot(normal, origin) + model[3]) * normal).astype(np.float64)
    return SupportPlane(
        model=model.astype(np.float32),
        normal=normal.astype(np.float32),
        origin=origin.astype(np.float32),
        inlier_count=int(len(inliers)),
    )


def mask_to_object_cloud(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """Project a segmented object mask to a camera-frame point cloud."""
    if mask.shape[:2] != depth_mm.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape[:2]} does not match depth shape {depth_mm.shape[:2]}."
        )

    depth_m = depth_mm.astype(np.float32) / 1000.0
    valid = (mask > 0) & np.isfinite(depth_m) & (depth_m > min_depth_m) & (depth_m < max_depth_m)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    z = depth_m[ys, xs]
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


def filter_object_cloud(
    points: np.ndarray,
    plane: SupportPlane,
    object_clearance_m: float,
    voxel_size_m: float,
    outlier_nb_neighbors: int,
    outlier_std_ratio: float,
) -> np.ndarray:
    """Remove table leakage, downsample, and filter outliers from object points."""
    if len(points) == 0:
        return points.astype(np.float32)

    normal = plane.normal.astype(np.float32)
    offset = float(plane.model[3])
    signed_distance = points @ normal + offset
    keep = signed_distance > float(object_clearance_m)
    filtered = points[keep]
    if len(filtered) == 0:
        return np.empty((0, 3), dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered.astype(np.float64))

    if voxel_size_m > 0.0 and len(filtered) > 1:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size_m))

    if len(pcd.points) >= max(8, int(outlier_nb_neighbors)):
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=int(outlier_nb_neighbors),
            std_ratio=float(outlier_std_ratio),
        )

    result = np.asarray(pcd.points, dtype=np.float32)
    return result if len(result) > 0 else np.empty((0, 3), dtype=np.float32)


def plane_basis_from_normal(
    normal: np.ndarray,
    reference_axis: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a stable orthonormal basis on the support plane."""
    normal = np.asarray(normal, dtype=np.float32)
    normal /= max(float(np.linalg.norm(normal)), 1e-8)

    reference = (
        np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if reference_axis is None
        else np.asarray(reference_axis, dtype=np.float32)
    )
    axis_u = reference - np.dot(reference, normal) * normal
    if np.linalg.norm(axis_u) < 1e-6:
        axis_u = np.array([0.0, 0.0, 1.0], dtype=np.float32) - normal[2] * normal
    if np.linalg.norm(axis_u) < 1e-6:
        axis_u = np.array([0.0, 1.0, 0.0], dtype=np.float32) - normal[1] * normal
    axis_u /= max(float(np.linalg.norm(axis_u)), 1e-8)
    axis_v = np.cross(normal, axis_u)
    axis_v /= max(float(np.linalg.norm(axis_v)), 1e-8)
    axis_u = np.cross(axis_v, normal)
    axis_u /= max(float(np.linalg.norm(axis_u)), 1e-8)
    return axis_u.astype(np.float32), axis_v.astype(np.float32), normal.astype(np.float32)


def resolve_named_axis(name: Optional[str]) -> Optional[np.ndarray]:
    """Map a simple config string to a camera-frame reference axis."""
    if name is None:
        return None

    key = str(name).strip().lower()
    if key in {"", "none", "off", "disabled"}:
        return None

    sign = 1.0
    if key.startswith("-"):
        sign = -1.0
        key = key[1:]
    elif key.startswith("+"):
        key = key[1:]

    mapping = {
        "camera_x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "camera_y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "camera_z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "image_x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "image_y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    axis = mapping.get(key)
    if axis is None:
        return None
    return (axis * sign).astype(np.float32)


def project_axis_to_plane(axis: Optional[np.ndarray], normal: np.ndarray) -> Optional[np.ndarray]:
    """Project a reference axis onto the support plane."""
    if axis is None:
        return None
    axis = np.asarray(axis, dtype=np.float32)
    normal = np.asarray(normal, dtype=np.float32)
    projected = axis - float(np.dot(axis, normal)) * normal
    norm = float(np.linalg.norm(projected))
    if norm < 1e-6:
        return None
    return (projected / norm).astype(np.float32)


def project_vector_to_plane(axis: np.ndarray, normal: np.ndarray) -> Optional[np.ndarray]:
    """Project a 3-D vector onto the plane orthogonal to ``normal``."""
    axis = np.asarray(axis, dtype=np.float32)
    normal = np.asarray(normal, dtype=np.float32)
    projected = axis - float(np.dot(axis, normal)) * normal
    norm = float(np.linalg.norm(projected))
    if norm < 1e-6:
        return None
    return (projected / norm).astype(np.float32)


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to ZYX Euler roll/pitch/yaw."""
    return rotation_matrix_to_euler_zyx(R).astype(np.float32)


def _rejected_result(
    point_count: int,
    rejected_reason: str,
    *,
    jaw_width_m: Optional[float] = None,
    object_size_m: Optional[np.ndarray] = None,
    candidate_label: str = "major",
    object_profile: str = "compact",
    contact_clearance_m: Optional[float] = None,
    table_collision_risk: float = 1.0,
) -> TopDownGraspResult:
    return TopDownGraspResult(
        position=None,
        pregrasp_position=None,
        rotation=None,
        euler_rpy=None,
        tcp_rotation=None,
        tcp_euler_rpy=None,
        jaw_width_m=jaw_width_m,
        object_size_m=object_size_m,
        grasp_score=0.0,
        point_count=int(point_count),
        rejected_reason=rejected_reason,
        candidate_label=candidate_label,
        orientation_score=0.0,
        object_profile=object_profile,
        contact_clearance_m=contact_clearance_m,
        table_collision_risk=table_collision_risk,
    )


def _robust_bounds(values: np.ndarray, low_pct: float, high_pct: float) -> tuple[float, float]:
    """Return robust low/high bounds from percentiles."""
    low, high = np.percentile(values, [float(low_pct), float(high_pct)])
    low = float(low)
    high = float(high)
    if high < low:
        low, high = high, low
    return low, high


def _classify_object_profile(
    object_height: float,
    grip_extent: float,
    open_extent: float,
    flat_object_max_height_m: float,
    flat_object_max_height_ratio: float,
    tall_object_min_height_m: float,
    tall_object_min_height_ratio: float,
    image_aspect_ratio: Optional[float] = None,
    flat_image_aspect_ratio: float = 0.78,
    tall_image_aspect_ratio: float = 1.15,
    elongated_planar_aspect_ratio: float = 1.8,
) -> str:
    """Classify the object into a coarse shape profile for scoring."""
    planar_span = max(float(grip_extent), float(open_extent), 1e-6)
    planar_minor = max(min(float(grip_extent), float(open_extent)), 1e-6)
    planar_aspect_ratio = planar_span / planar_minor
    height_ratio = float(object_height) / planar_span

    is_flat = (
        float(object_height) <= float(flat_object_max_height_m)
        or height_ratio <= float(flat_object_max_height_ratio)
    )
    is_tall = (
        float(object_height) >= float(tall_object_min_height_m)
        or height_ratio >= float(tall_object_min_height_ratio)
    )

    if image_aspect_ratio is not None:
        image_aspect_ratio = float(image_aspect_ratio)
        image_is_flat = image_aspect_ratio <= float(flat_image_aspect_ratio)
        image_is_tall = image_aspect_ratio >= float(tall_image_aspect_ratio)
        if image_is_tall and float(object_height) >= float(flat_object_max_height_m) * 0.75:
            return "tall"
        if image_is_flat and planar_aspect_ratio >= float(elongated_planar_aspect_ratio):
            if float(object_height) <= float(flat_object_max_height_m) * 3.0:
                return "flat"

    if is_flat and not is_tall:
        return "flat"
    if is_tall:
        return "tall"
    return "compact"


def estimate_topdown_grasp_candidates(
    points: np.ndarray,
    plane: SupportPlane,
    det_confidence: float,
    partial: bool,
    max_width_m: float,
    width_percentile_low: float,
    width_percentile_high: float,
    pregrasp_offset_m: float,
    grasp_height_ratio: float,
    min_height_above_plane_m: float,
    candidates_per_object: int = 2,
    allow_minor_axis_grasp: bool = True,
    preferred_axis: Optional[np.ndarray] = None,
    preferred_axis_target: str = "open_axis",
    preferred_axis_weight: float = 0.18,
    extent_percentile_low: float = 10.0,
    extent_percentile_high: float = 90.0,
    flat_object_max_height_m: float = 0.045,
    flat_object_max_height_ratio: float = 0.30,
    tall_object_min_height_m: float = 0.070,
    tall_object_min_height_ratio: float = 0.60,
    flat_grasp_height_ratio: float = 0.82,
    tall_grasp_height_ratio: float = 0.55,
    min_contact_clearance_m: float = 0.024,
    low_profile_reject_height_m: float = 0.018,
    low_profile_min_contact_clearance_m: Optional[float] = None,
    collision_penalty_weight: float = 0.22,
    image_aspect_ratio: Optional[float] = None,
    flat_image_aspect_ratio: float = 0.78,
    tall_image_aspect_ratio: float = 1.15,
    elongated_planar_aspect_ratio: float = 1.8,
    preferred_approach_axis: Optional[np.ndarray] = None,
    prefer_base_down_for_flat: bool = False,
    prefer_base_down_profiles: Optional[list[str]] = None,
    base_down_max_tilt_deg: float = 35.0,
) -> list[TopDownGraspResult]:
    """Estimate one or more tabletop top-grasp candidates from an object cloud."""
    point_count = int(len(points))
    if point_count < 16:
        return [_rejected_result(point_count, "too_few_points")]

    approach = plane.normal.astype(np.float32)
    plane_u, plane_v, approach = plane_basis_from_normal(approach)
    origin = plane.origin.astype(np.float32)
    relative = points.astype(np.float32) - origin[None, :]
    planar_coords = np.stack([relative @ plane_u, relative @ plane_v], axis=1)
    planar_centered = planar_coords - planar_coords.mean(axis=0, keepdims=True)

    if len(points) >= 3:
        cov = np.cov(planar_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_2d = eigvecs[:, int(np.argmax(eigvals))].astype(np.float32)
        minor_2d = eigvecs[:, int(np.argmin(eigvals))].astype(np.float32)
    else:
        major_2d = np.array([1.0, 0.0], dtype=np.float32)
        minor_2d = np.array([0.0, 1.0], dtype=np.float32)

    candidate_specs: list[tuple[str, np.ndarray]] = [("major", major_2d)]
    if allow_minor_axis_grasp:
        candidate_specs.append(("minor", minor_2d))

    preferred_axis_target = str(preferred_axis_target or "open_axis").strip().lower()
    if preferred_axis_target not in {"open_axis", "grip_axis"}:
        preferred_axis_target = "open_axis"
    preferred_axis_on_plane = project_axis_to_plane(preferred_axis, approach)
    preferred_axis_weight = float(np.clip(preferred_axis_weight, 0.0, 0.8))
    preferred_approach = None
    if preferred_approach_axis is not None:
        preferred_approach = np.asarray(preferred_approach_axis, dtype=np.float32)
        preferred_approach /= max(float(np.linalg.norm(preferred_approach)), 1e-8)
        if float(np.dot(preferred_approach, approach)) < 0.0:
            preferred_approach = -preferred_approach
    base_down_profiles = None
    if prefer_base_down_profiles is not None:
        base_down_profiles = {
            str(profile).strip().lower()
            for profile in prefer_base_down_profiles
            if str(profile).strip()
        }
    max_tilt_cos = float(np.cos(np.deg2rad(max(float(base_down_max_tilt_deg), 0.0))))
    low_profile_min_contact_clearance_m = float(
        min_contact_clearance_m
        if low_profile_min_contact_clearance_m is None
        else low_profile_min_contact_clearance_m
    )

    results: list[TopDownGraspResult] = []
    for candidate_label, grip_2d in candidate_specs:
        grip = (grip_2d[0] * plane_u + grip_2d[1] * plane_v).astype(np.float32)
        grip /= max(float(np.linalg.norm(grip)), 1e-8)
        open_axis = np.cross(approach, grip).astype(np.float32)
        open_axis /= max(float(np.linalg.norm(open_axis)), 1e-8)
        grip = np.cross(open_axis, approach).astype(np.float32)
        grip /= max(float(np.linalg.norm(grip)), 1e-8)

        target_axis = open_axis if preferred_axis_target == "open_axis" else grip
        if preferred_axis_on_plane is not None:
            if float(np.dot(target_axis, preferred_axis_on_plane)) < 0.0:
                grip = -grip
                open_axis = -open_axis
                target_axis = -target_axis
        elif float(np.dot(grip, plane_u)) < 0.0:
            grip = -grip
            open_axis = -open_axis
            target_axis = open_axis if preferred_axis_target == "open_axis" else grip

        local_grip = relative @ grip
        local_open = relative @ open_axis
        local_height = relative @ approach

        grip_min, grip_max = _robust_bounds(local_grip, extent_percentile_low, extent_percentile_high)
        open_min, open_max = _robust_bounds(local_open, extent_percentile_low, extent_percentile_high)

        core_mask = (
            (local_grip >= grip_min)
            & (local_grip <= grip_max)
            & (local_open >= open_min)
            & (local_open <= open_max)
        )
        core_height = local_height[core_mask]
        if len(core_height) < max(32, point_count // 4):
            core_height = local_height

        height_min, height_max = _robust_bounds(core_height, extent_percentile_low, extent_percentile_high)
        object_height = max(float(height_max - height_min), 0.0)
        grip_extent = max(float(grip_max - grip_min), 0.0)
        open_extent = max(float(open_max - open_min), 0.0)
        object_size_m = np.array(
            [grip_extent, open_extent, object_height],
            dtype=np.float32,
        )
        object_profile = _classify_object_profile(
            object_height=object_height,
            grip_extent=grip_extent,
            open_extent=open_extent,
            flat_object_max_height_m=flat_object_max_height_m,
            flat_object_max_height_ratio=flat_object_max_height_ratio,
            tall_object_min_height_m=tall_object_min_height_m,
            tall_object_min_height_ratio=tall_object_min_height_ratio,
            image_aspect_ratio=image_aspect_ratio,
            flat_image_aspect_ratio=flat_image_aspect_ratio,
            tall_image_aspect_ratio=tall_image_aspect_ratio,
            elongated_planar_aspect_ratio=elongated_planar_aspect_ratio,
        )

        if object_height < float(min_height_above_plane_m):
            results.append(
                _rejected_result(
                    point_count,
                    "object_too_flat",
                    object_size_m=object_size_m,
                    candidate_label=candidate_label,
                    object_profile=object_profile,
                )
            )
            continue

        jaw_low, jaw_high = np.percentile(
            local_open,
            [float(width_percentile_low), float(width_percentile_high)],
        )
        jaw_width = float(max(jaw_high - jaw_low, 0.0))
        if jaw_width <= 1e-5:
            results.append(
                _rejected_result(
                    point_count,
                    "zero_width",
                    object_size_m=object_size_m,
                    candidate_label=candidate_label,
                    object_profile=object_profile,
                )
            )
            continue
        if jaw_width > float(max_width_m):
            results.append(
                _rejected_result(
                    point_count,
                    "width_exceeds_limit",
                    jaw_width_m=jaw_width,
                    object_size_m=object_size_m,
                    candidate_label=candidate_label,
                    object_profile=object_profile,
                )
            )
            continue

        if object_profile == "flat":
            grasp_height_ratio_eff = float(flat_grasp_height_ratio)
        elif object_profile == "tall":
            grasp_height_ratio_eff = float(tall_grasp_height_ratio)
        else:
            grasp_height_ratio_eff = float(grasp_height_ratio)
        grasp_height_ratio_eff = float(np.clip(grasp_height_ratio_eff, 0.05, 0.95))

        center_height = height_min + grasp_height_ratio_eff * object_height
        contact_clearance = max(float(center_height - height_min), 0.0)
        collision_risk = float(
            np.clip(
                1.0 - contact_clearance / max(float(min_contact_clearance_m), 1e-6),
                0.0,
                1.0,
            )
        )
        if object_profile == "flat" and object_height < float(low_profile_reject_height_m):
            results.append(
                _rejected_result(
                    point_count,
                    "low_profile_collision_risk",
                    jaw_width_m=jaw_width,
                    object_size_m=object_size_m,
                    candidate_label=candidate_label,
                    object_profile=object_profile,
                    contact_clearance_m=contact_clearance,
                    table_collision_risk=collision_risk,
                )
            )
            continue
        if object_profile == "flat" and contact_clearance < float(low_profile_min_contact_clearance_m):
            results.append(
                _rejected_result(
                    point_count,
                    "low_profile_contact_clearance",
                    jaw_width_m=jaw_width,
                    object_size_m=object_size_m,
                    candidate_label=candidate_label,
                    object_profile=object_profile,
                    contact_clearance_m=contact_clearance,
                    table_collision_risk=collision_risk,
                )
            )
            continue

        center_local = np.array(
            [
                0.5 * (grip_min + grip_max),
                0.5 * (open_min + open_max),
                center_height,
            ],
            dtype=np.float32,
        )
        final_approach = approach.copy()
        use_base_down_for_profile = False
        if base_down_profiles is None:
            use_base_down_for_profile = prefer_base_down_for_flat and object_profile == "flat"
        else:
            use_base_down_for_profile = object_profile in base_down_profiles
        if (
            preferred_approach is not None
            and use_base_down_for_profile
        ):
            if float(np.dot(preferred_approach, approach)) >= max_tilt_cos:
                final_approach = preferred_approach.copy()

        final_grip = grip.copy()
        if not np.allclose(final_approach, approach):
            projected_grip = project_vector_to_plane(grip, final_approach)
            if projected_grip is None:
                projected_grip = project_vector_to_plane(open_axis, final_approach)
            if projected_grip is not None:
                final_grip = projected_grip
        final_grip /= max(float(np.linalg.norm(final_grip)), 1e-8)
        final_open = np.cross(final_approach, final_grip).astype(np.float32)
        final_open /= max(float(np.linalg.norm(final_open)), 1e-8)
        final_grip = np.cross(final_open, final_approach).astype(np.float32)
        final_grip /= max(float(np.linalg.norm(final_grip)), 1e-8)

        position = (
            origin
            + final_grip * center_local[0]
            + final_open * center_local[1]
            + final_approach * center_local[2]
        ).astype(np.float32)
        pregrasp_position = (position + final_approach * float(pregrasp_offset_m)).astype(np.float32)
        rotation = np.column_stack([final_grip, final_open, final_approach]).astype(np.float32)
        if np.linalg.det(rotation) < 0.0:
            rotation[:, 1] = -rotation[:, 1]
        tcp_rotation = grasp_axes_to_rebot_tcp_rotation(
            grip_axis=final_grip,
            open_axis=final_open,
            approach_axis=final_approach,
        ).astype(np.float32)

        point_score = min(1.0, point_count / 1200.0)
        width_ratio = jaw_width / max(float(max_width_m), 1e-6)
        width_score = max(0.0, 1.0 - width_ratio)
        height_score = min(1.0, object_height / max(float(min_height_above_plane_m) * 3.0, 1e-6))
        partial_score = 0.8 if partial else 1.0
        clearance_score = 1.0 - collision_risk
        base_score = float(
            np.clip(
                0.45 * float(det_confidence)
                + 0.25 * point_score
                + 0.15 * width_score
                + 0.10 * partial_score
                + 0.05 * height_score
                + 0.05 * clearance_score,
                0.0,
                1.0,
            )
        )
        base_score = float(
            np.clip(
                base_score - float(collision_penalty_weight) * collision_risk,
                0.0,
                1.0,
            )
        )

        if preferred_axis_on_plane is not None:
            orientation_score = float(np.clip(np.dot(target_axis, preferred_axis_on_plane), 0.0, 1.0))
            grasp_score = float(
                np.clip(
                    (1.0 - preferred_axis_weight) * base_score + preferred_axis_weight * orientation_score,
                    0.0,
                    1.0,
                )
            )
        else:
            orientation_score = 1.0
            grasp_score = base_score

        results.append(
            TopDownGraspResult(
                position=position,
                pregrasp_position=pregrasp_position,
                rotation=rotation,
                euler_rpy=rotation_matrix_to_euler(rotation),
                tcp_rotation=tcp_rotation,
                tcp_euler_rpy=rotation_matrix_to_euler_zyx(tcp_rotation).astype(np.float32),
                jaw_width_m=jaw_width,
                object_size_m=object_size_m,
                grasp_score=grasp_score,
                point_count=point_count,
                rejected_reason=None,
                candidate_label=candidate_label,
                orientation_score=orientation_score,
                object_profile=object_profile,
                contact_clearance_m=contact_clearance,
                table_collision_risk=collision_risk,
            )
        )

    valid_results = [result for result in results if result.rejected_reason is None]
    if valid_results:
        valid_results.sort(key=lambda item: item.grasp_score, reverse=True)
        return valid_results[: max(1, int(candidates_per_object))]

    return results[:1]


def estimate_topdown_grasp(
    points: np.ndarray,
    plane: SupportPlane,
    det_confidence: float,
    partial: bool,
    max_width_m: float,
    width_percentile_low: float,
    width_percentile_high: float,
    pregrasp_offset_m: float,
    grasp_height_ratio: float,
    min_height_above_plane_m: float,
) -> TopDownGraspResult:
    """Estimate a tabletop top-grasp candidate from an object point cloud."""
    candidates = estimate_topdown_grasp_candidates(
        points=points,
        plane=plane,
        det_confidence=det_confidence,
        partial=partial,
        max_width_m=max_width_m,
        width_percentile_low=width_percentile_low,
        width_percentile_high=width_percentile_high,
        pregrasp_offset_m=pregrasp_offset_m,
        grasp_height_ratio=grasp_height_ratio,
        min_height_above_plane_m=min_height_above_plane_m,
        candidates_per_object=1,
        allow_minor_axis_grasp=False,
        preferred_axis=None,
        preferred_axis_target="open_axis",
        preferred_axis_weight=0.0,
        extent_percentile_low=10.0,
        extent_percentile_high=90.0,
        flat_object_max_height_m=0.045,
        flat_object_max_height_ratio=0.30,
        tall_object_min_height_m=0.070,
        tall_object_min_height_ratio=0.60,
        flat_grasp_height_ratio=0.82,
        tall_grasp_height_ratio=0.55,
        min_contact_clearance_m=0.024,
        low_profile_reject_height_m=0.018,
        low_profile_min_contact_clearance_m=0.024,
        collision_penalty_weight=0.22,
        image_aspect_ratio=None,
        flat_image_aspect_ratio=0.78,
        tall_image_aspect_ratio=1.15,
        elongated_planar_aspect_ratio=1.8,
    )
    return candidates[0]


def create_virtual_gripper(
    T: np.ndarray,
    width: float = 0.08,
    depth: float = 0.06,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
):
    """Create a simple line-set gripper for Open3D debugging."""
    half_width = float(width) * 0.5
    points_local = np.array(
        [
            [0.0, 0.0, 0.04],
            [0.0, -half_width, 0.0],
            [0.0, half_width, 0.0],
            [0.0, -half_width, -depth],
            [0.0, half_width, -depth],
        ],
        dtype=np.float64,
    )
    points_world = (T @ np.hstack([points_local, np.ones((len(points_local), 1))]).T).T[:, :3]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_world)
    line_set.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [1, 2], [1, 3], [2, 4]])
    line_set.colors = o3d.utility.Vector3dVector([list(color)] * 5)
    return line_set
