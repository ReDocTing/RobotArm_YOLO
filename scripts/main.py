"""
main.py — 基于短轴估计的机械臂夹取主程序
=========================================
流程：
  1. 机械臂 + 夹爪使能，移动到预备高位
  2. 实时相机预览 + YOLO 检测 + 短轴夹取姿态估计
  3. G 键：冻结当前帧并执行夹取（--dry-run 只打印坐标）
  4. R 键：恢复实时预览
  5. Q 键：退出，释放夹爪并回零位
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT,):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from drivers.camera import make_camera
from drivers.robot.rebot_arm import RebotArm
from utils.ordinary_grasp import GraspPose, draw_grasp, estimate_grasps, select_best_grasp
from utils.transforms import (
    canonicalize_parallel_gripper_tcp_rotation,
    mat4_to_pose6d,
    rotation_matrix_to_euler_zyx,
)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_hand_eye(project_root: Path, cam_type: str) -> tuple[Optional[np.ndarray], Optional[str]]:
    hand_eye_path = project_root / "config" / "calibration" / cam_type / "hand_eye.npz"
    if not hand_eye_path.exists():
        return None, None

    data = np.load(str(hand_eye_path), allow_pickle=False)
    T = data["T_result"].astype(np.float64)
    mode = str(data["mode"][0])
    return T, mode


def _move_ready(robot: RebotArm, ready_cfg: dict[str, Any]) -> None:
    duration = float(ready_cfg.get("duration", 3.0))
    robot.move_to(
        float(ready_cfg.get("x", 0.25)),
        float(ready_cfg.get("y", 0.0)),
        float(ready_cfg.get("z", 0.35)),
        float(ready_cfg.get("roll", 0.0)),
        float(ready_cfg.get("pitch", 1.2)),
        float(ready_cfg.get("yaw", 0.0)),
        duration=duration,
    )
    robot.wait_motion(duration)


def _cam_to_base(T_hand_eye: np.ndarray, robot: RebotArm) -> np.ndarray:
    return robot.get_tcp_pose() @ T_hand_eye


def _load_grasp_depth_cfg(cfg: dict[str, Any]) -> tuple[float, float, float, dict[str, Any]]:
    """返回 depth_quantile, depth_offset_mm, approach_offset_m, safety_cfg。"""
    gp_cfg = cfg.get("grasp_pipeline", {})
    grasp_cfg = gp_cfg.get("grasp", {})
    safety_cfg = dict(gp_cfg.get("safety", {}))

    depth_quantile = float(grasp_cfg.get("depth_quantile", 0.75))
    max_depth_off = float(safety_cfg.get("max_depth_offset_mm", 40.0))
    depth_offset_mm = float(np.clip(float(grasp_cfg.get("depth_offset_mm", 0.0)), -max_depth_off, max_depth_off))

    max_app = float(safety_cfg.get("max_approach_offset_m", 0.04))
    approach_offset_m = float(np.clip(float(grasp_cfg.get("approach_offset_m", 0.0)), 0.0, max_app))

    return depth_quantile, depth_offset_mm, approach_offset_m, safety_cfg


def _check_grasp_safe(grasp6d: tuple[float, ...], safety_cfg: dict[str, Any]) -> bool:
    """基座系高度护栏，防止深度补偿过度导致撞桌或抬得过高。"""
    zg = float(grasp6d[2])
    min_z = safety_cfg.get("min_grasp_z_m")
    max_z = safety_cfg.get("max_grasp_z_m")
    if min_z is not None and zg < float(min_z):
        print(f"[Safety] 抓取 z={zg:.3f}m 低于 min_grasp_z_m={float(min_z):.3f}m，已中止")
        return False
    if max_z is not None and zg > float(max_z):
        print(f"[Safety] 抓取 z={zg:.3f}m 高于 max_grasp_z_m={float(max_z):.3f}m，已中止")
        return False
    return True


def _transform_grasp(
    grasp: GraspPose,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    approach_offset_m: float = 0.0,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    T_grasp_cam = np.eye(4, dtype=np.float64)
    T_grasp_cam[:3, :3] = grasp.tcp_rotation.astype(np.float64)
    T_grasp_cam[:3, 3] = grasp.position.astype(np.float64)

    T_grasp_base = T_cam2base @ T_grasp_cam
    grasp_pos_base = T_grasp_base[:3, 3].copy()
    grasp_rot_base = canonicalize_parallel_gripper_tcp_rotation(T_grasp_base[:3, :3])
    T_grasp_base[:3, :3] = grasp_rot_base

    if approach_offset_m > 0:
        # tcp_x 指向接近/进入物体方向；正值 = 在抓取位基础上再下探一点
        grasp_pos_base = grasp_pos_base + grasp_rot_base[:, 0] * float(approach_offset_m)
        T_grasp_base[:3, 3] = grasp_pos_base

    pregrasp_pos_base = grasp_pos_base - grasp_rot_base[:, 0] * float(pregrasp_offset_m)
    T_pregrasp_base = np.eye(4, dtype=np.float64)
    T_pregrasp_base[:3, :3] = grasp_rot_base
    T_pregrasp_base[:3, 3] = pregrasp_pos_base

    return mat4_to_pose6d(T_grasp_base), mat4_to_pose6d(T_pregrasp_base)


def _execute_grasp(
    robot: RebotArm,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    ready_cfg: dict[str, Any],
    dry_run: bool,
) -> bool:
    xg, yg, zg, rxg, ryg, rzg = grasp6d
    xp, yp, zp, rxp, ryp, rzp = pre6d

    print(f"[Grasp] pregrasp  xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f})  rpy=({rxp:+.3f},{ryp:+.3f},{rzp:+.3f})")
    print(f"[Grasp] grasp     xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f})  rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f})")

    if dry_run:
        print("[Grasp] --dry-run: 跳过执行")
        return False

    print("[Grasp] 打开夹爪...")
    robot.open_gripper()

    print("[Grasp] 移动到预夹取位...")
    if not robot.move_to(xp, yp, zp, rxp, ryp, rzp, duration=2.0):
        print("[Grasp] 预夹取 IK 失败，中止")
        return False
    robot.wait_motion(2.0)

    print("[Grasp] 移动到夹取位...")
    if not robot.move_to(xg, yg, zg, rxg, ryg, rzg, duration=1.5):
        print("[Grasp] 夹取 IK 失败，中止")
        return False
    robot.wait_motion(1.5)

    print("[Grasp] 夹取中...")
    ok = robot.grasp()
    print("[Grasp] ✓ 夹取成功，力控保持中" if ok else "[Grasp] 空夹取")

    print("[Grasp] 返回预备位...")
    _move_ready(robot, ready_cfg)
    return ok


def _render_display(
    image: np.ndarray,
    grasps: list[GraspPose],
    best: Optional[GraspPose],
    status_text: str,
) -> np.ndarray:
    display = image.copy()
    for grasp in grasps:
        draw_grasp(display, grasp)

    cv2.putText(display, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    if best is not None:
        x_m, y_m, z_m = best.position.tolist()
        best_text = (
            f"best={best.class_name} conf={best.conf:.2f} "
            f"xyz=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f}) jaw={best.jaw_width_m * 100:.1f}cm"
        )
        cv2.putText(
            display,
            best_text,
            (10, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 255, 140),
            2,
        )
    return display


def _print_best_grasp(grasp: GraspPose) -> None:
    tcp_rotation = canonicalize_parallel_gripper_tcp_rotation(grasp.tcp_rotation)
    print("\n[G] 当前最佳夹取:")
    print(f"  class={grasp.class_name} conf={grasp.conf:.3f}")
    print(f"  center_px={grasp.center_px} angle_deg={grasp.angle_deg:.2f}")
    print(f"  jaw_width_m={grasp.jaw_width_m:.4f} object_length_m={grasp.object_length_m:.4f}")
    print(f"  position_xyz={grasp.position.tolist()}")
    print(f"  grasp_rpy={rotation_matrix_to_euler_zyx(grasp.rotation).tolist()}")
    print(f"  tcp_rpy={rotation_matrix_to_euler_zyx(tcp_rotation).tolist()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于短轴估计的机械臂夹取主程序")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="只估计夹取姿态，不移动机械臂")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)

    robot_cfg = cfg.get("robot", {})
    ready_cfg = robot_cfg.get(
        "ready_pose",
        {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
    )

    print("=== 初始化机械臂 ===")
    robot = RebotArm(
        config_path=robot_cfg.get("config_path"),
        urdf_path=robot_cfg.get("urdf_path"),
        repo_root=robot_cfg.get("repo_root"),
    )
    robot.connect(enable=True)
    robot.init_gripper()

    print("[Robot] 移动到预备位置...")
    _move_ready(robot, ready_cfg)

    cam_type = str(cfg.get("camera", {}).get("type", "")).lower()
    T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
    if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
        print("[WARN] 手眼标定不可用或非 eye_in_hand，夹取执行将被禁用")
        T_hand_eye = None

    print(f"=== 相机: {cfg.get('camera', {}).get('type')} ===")
    cam = make_camera(cfg)
    cam.open()
    cam.warm_up(15)
    K = cam.K.astype(np.float32)

    yolo_cfg = cfg.get("yolo", {})
    det_cfg = cfg.get("detection", {})
    gp_cfg = cfg.get("grasp_pipeline", {})

    model_name = yolo_cfg.get("model_name", "yoloe-26s-seg.pt")
    yolo_device = yolo_cfg.get("device", "cpu")
    conf = float(det_cfg.get("conf_threshold", 0.25))
    iou = float(det_cfg.get("iou_threshold", 0.45))
    grasp_cfg = gp_cfg.get("grasp", {})
    pregrasp_offset_m = float(grasp_cfg.get("pregrasp_offset_m", 0.08))
    depth_quantile, depth_offset_mm, approach_offset_m, safety_cfg = _load_grasp_depth_cfg(cfg)
    infer_every = max(1, int(gp_cfg.get("infer_every_live", 2)))

    if depth_offset_mm != 0 or approach_offset_m != 0:
        print(
            f"[Depth] quantile={depth_quantile:.2f}  offset={depth_offset_mm:+.1f}mm  "
            f"approach_extra={approach_offset_m * 1000:.1f}mm"
        )

    print(f"=== 加载 YOLO: {model_name} ===")
    model = YOLO(str(PROJECT_ROOT / "models" / model_name))
    if yolo_cfg.get("use_world", True) and ("world" in model_name.lower() or "yoloe" in model_name.lower()):
        model.set_classes(list(yolo_cfg.get("custom_classes", [])))

    last_results: list[Any] = []
    last_grasps: list[GraspPose] = []
    frozen = False
    last_display: Optional[np.ndarray] = None
    frame_index = 0
    fps_counter = 0
    fps_timer = time.perf_counter()
    fps_value = 0.0

    window_name = "Main — Ordinary Grasp"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("\n[Keys]  G=夹取  R=恢复  Q/ESC=退出\n")

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

            if not frozen and (frame_index % infer_every == 0 or not last_results):
                last_results = model.predict(
                    color_bgr,
                    verbose=False,
                    device=yolo_device,
                    conf=conf,
                    iou=iou,
                )
                last_grasps = estimate_grasps(
                    last_results,
                    depth_mm,
                    K,
                    depth_quantile=depth_quantile,
                    depth_offset_mm=depth_offset_mm,
                )

            status = f"{'FROZEN' if frozen else 'LIVE'} {fps_value:.1f}fps | G=夹取 R=恢复 Q=退出"
            best_live = select_best_grasp(last_grasps)
            if frozen and last_display is not None:
                display = last_display.copy()
                cv2.putText(display, "[FROZEN]", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
            else:
                display = _render_display(color_bgr, last_grasps, best_live, status)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                frozen = False
                last_display = None
                continue

            if key in (ord("g"), ord("G")):
                print("\n[G] 采帧并估计夹取姿态...")
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None or snap_depth is None:
                    print("[G] 采帧失败")
                    continue

                snap_results = model.predict(
                    snap_color,
                    verbose=False,
                    device=yolo_device,
                    conf=conf,
                    iou=iou,
                )
                snap_grasps = estimate_grasps(
                    snap_results,
                    snap_depth,
                    K,
                    depth_quantile=depth_quantile,
                    depth_offset_mm=depth_offset_mm,
                )
                best = select_best_grasp(snap_grasps)
                if best is None:
                    print("[G] 未找到有效夹取候选")
                    continue

                _print_best_grasp(best)

                snap_display = _render_display(snap_color, snap_grasps, best, "SNAPSHOT")
                frozen = True
                last_display = snap_display
                last_results = snap_results
                last_grasps = snap_grasps

                if T_hand_eye is None:
                    print("[G] 手眼标定不可用，无法执行夹取")
                    continue

                T_cam2base = _cam_to_base(T_hand_eye, robot)
                grasp6d, pre6d = _transform_grasp(
                    best,
                    T_cam2base,
                    pregrasp_offset_m,
                    approach_offset_m=approach_offset_m,
                )
                print(
                    f"  → 相机深度 Z={float(best.position[2]):.3f}m（画面显示）  "
                    f"基座抓取高度 z={float(grasp6d[2]):.3f}m（机械臂实际去的高度）"
                )
                if not _check_grasp_safe(grasp6d, safety_cfg):
                    continue
                _execute_grasp(robot, grasp6d, pre6d, ready_cfg, dry_run=args.dry_run)

    finally:
        print("\n[退出] 释放夹爪并回零...")
        try:
            robot.release_gripper()
            robot.safe_home()
        except Exception as exc:
            print(f"[退出] {exc}")
        robot.disconnect()
        cam.close()
        cv2.destroyAllWindows()
        print("已退出。")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
