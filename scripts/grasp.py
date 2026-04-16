"""机械臂目标夹取控制脚本 (Eye-in-Hand)

功能：
  - 实时 YOLO-World 目标检测（支持 D435i / Gemini2）
  - 相机坐标 → 机器人基座坐标（加载手眼标定结果）
  - 按 [G] 触发自动抓取状态机：张开→接近→下降→夹紧→提升

前提条件：
  1. 已完成相机内参标定（get_camera_intrinsics.py）
  2. 已完成手眼标定（collect_handeye_eih.py）→ config/calibration/{cam_type}/hand_eye.npz
  3. 机械臂已通过 USB-to-CAN 连接

用法：
    cd /home/chlorine/seeed/cameraws
    python scripts/grasp.py           # 连接机械臂（使能电机）+ 目标抓取
    python scripts/grasp.py --dry-run # 连接机械臂（不使能电机），验证手眼标定坐标映射
    python scripts/grasp.py --manual  # 不连接机械臂，仅显示相机坐标系结果

按键说明：
    G - 开始抓取当前置信度最高目标 / 抓取进行中按 G 则中止
    R - 机械臂回零位（safe_home）
    Q / ESC - 退出
"""

import os
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

import sys
import time
import argparse
from enum import Enum

import cv2
import numpy as np
import yaml
from pathlib import Path
from ultralytics import YOLO


# ==========================================
# 配置加载
# ==========================================
def load_config(yaml_path):
    if not os.path.exists(yaml_path):
        print(f"[错误] 找不到配置文件: {yaml_path}")
        sys.exit(1)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ==========================================
# 手眼标定加载
# ==========================================
def load_hand_eye(npz_path: str):
    """加载手眼标定结果，返回 (T, mode)。"""
    data = np.load(npz_path, allow_pickle=False)
    T = data["T_result"].astype(np.float64)
    mode = str(data["mode"][0])
    n = int(data["n_samples"][0]) if "n_samples" in data else 0
    print(f"[手眼标定] 已加载: mode={mode}, 样本数={n}")
    print(f"  平移: x={T[0,3]:.4f} y={T[1,3]:.4f} z={T[2,3]:.4f} m")
    return T, mode


# ==========================================
# 深度采样
# ==========================================
def get_depth_mm(depth_map: np.ndarray, u: int, v: int, roi: int = 5) -> float:
    """从深度图（uint16，单位 mm）采样中位数深度。"""
    h, w = depth_map.shape
    half = roi // 2
    patch = depth_map[
        max(0, v - half):min(h, v + half + 1),
        max(0, u - half):min(w, u + half + 1),
    ]
    valid = patch[patch > 0]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


# ==========================================
# 坐标变换
# ==========================================
def cam_to_robot(p_cam: np.ndarray, T: np.ndarray,
                 mode: str, T_gripper2base: np.ndarray = None) -> np.ndarray:
    """将相机坐标系中的点变换到机器人基座坐标系。"""
    p_cam_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
    if mode == "eye_to_hand":
        p_base_h = T @ p_cam_h
    else:
        if T_gripper2base is None:
            raise ValueError("Eye-in-Hand 模式需要提供 T_gripper2base（FK）")
        p_base_h = T_gripper2base @ T @ p_cam_h
    return p_base_h[:3]


# ==========================================
# 末端姿态读取
# ==========================================
def _get_tcp_rpy(robot) -> tuple:
    """读取当前末端位姿 ZYX-RPY（弧度），失败时返回 (0, 0, 0)。

    注意：用当前姿态作为抓取目标姿态，可避免 roll=pitch=yaw=0 导致的 IK 失败。
    确保按 G 前机械臂已处于合适的抓取朝向（末端面向目标）。
    """
    try:
        T = robot.get_tcp_pose()
        R = T[:3, :3]
        sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
        if sy > 1e-6:
            r  = float(np.arctan2(R[2, 1], R[2, 2]))
            p  = float(np.arctan2(-R[2, 0], sy))
            yw = float(np.arctan2(R[1, 0], R[0, 0]))
        else:
            r  = float(np.arctan2(-R[1, 2], R[1, 1]))
            p  = float(np.arctan2(-R[2, 0], sy))
            yw = 0.0
        return (r, p, yw)
    except Exception as e:
        print(f"[警告] 无法读取末端姿态，使用 RPY=(0,0,0): {e}")
        return (0.0, 0.0, 0.0)


# ==========================================
# 抓取状态机
# ==========================================
# ── 可调参数（根据实际场景修改）─────────────────
APPROACH_Z_OFFSET = 0.08   # 接近点高于目标 (m)
LIFT_Z_OFFSET     = 0.10   # 提升高于抓取点 (m)
GRASP_MOVE_DUR    = 1.5    # 每段轨迹时长 (s)
GRIPPER_OPEN_WAIT       = 0.8    # 张开等待时间 (s)
GRIPPER_OPEN_M          = 0.09   # 张开距离 (m)，对应电机 -5 rad
GRASP_STALL_VEL         = 0.05   # 接触判定速度阈值 (rad/s)，低于此值视为夹到物体
GRASP_STARTUP_DELAY     = 0.3    # 启动阶段跳过时间 (s)，等待电机开始运动
GRASP_DETECT_TIMEOUT    = 2.0    # 接触检测超时 (s)，超时后继续提升
GRASP_STD_TORQUE        = 0.5    # 标准夹取力矩 (Nm) — 根据实际场景调整


class GraspState(Enum):
    IDLE     = "待机"
    OPEN     = "张开夹爪"
    APPROACH = "接近目标"
    DESCEND  = "下降抓取"
    GRASP    = "夹紧"
    LIFT     = "提升"
    DONE     = "完成"


class GraspFSM:
    """非阻塞抓取状态机，每帧调用 tick() 推进。

    序列: IDLE → OPEN → APPROACH → DESCEND → GRASP → LIFT → DONE

    - 夹爪阶段（OPEN/GRASP）：发送指令后等待 GRIPPER_WAIT 秒
    - 运动阶段（APPROACH/DESCEND/LIFT）：调用 robot.move_to 后等待 GRASP_MOVE_DUR+0.3 秒
    - 任意阶段 IK 失败 → 立即回 IDLE
    - 抓取进行中再按 G → abort() 中止并回 IDLE
    """

    def __init__(self):
        self.state        = GraspState.IDLE
        self._deadline    = 0.0      # 当前阶段结束时刻 (time.monotonic)
        self._approach    = None     # (x, y, z, roll, pitch, yaw)
        self._grasp       = None
        self._lift        = None
        self._grasp_start = 0.0      # 发送夹紧指令的时刻，用于跳过启动阶段

    @property
    def active(self) -> bool:
        return self.state not in (GraspState.IDLE, GraspState.DONE)

    def start(self, p_robot: np.ndarray, rpy: tuple) -> None:
        """G 键触发：冻结目标坐标，以当前末端姿态为基准，启动状态机。"""
        x, y, z = p_robot
        r, p, yw = rpy
        self._approach = (x, y, z + APPROACH_Z_OFFSET, r, p, yw)
        self._grasp    = (x, y, z,                      r, p, yw)
        self._lift     = (x, y, z + LIFT_Z_OFFSET,      r, p, yw)
        self._deadline = 0.0      # OPEN 首次 tick 立即执行
        self.state     = GraspState.OPEN
        print(f"\n[抓取] 启动 → 目标 Base X={x:.3f} Y={y:.3f} Z={z:.3f} m")
        print(f"       接近 Z={z + APPROACH_Z_OFFSET:.3f}  抓取 Z={z:.3f}  "
              f"提升 Z={z + LIFT_Z_OFFSET:.3f}  RPY=({r:.2f},{p:.2f},{yw:.2f})")

    def abort(self) -> None:
        self.state = GraspState.IDLE
        print("[抓取] 已中止")

    def tick(self, robot) -> None:
        """每帧调用，驱动状态机推进。无副作用时立即返回。"""
        s = self.state
        if s in (GraspState.IDLE, GraspState.DONE):
            return

        now = time.monotonic()

        if s == GraspState.OPEN:
            if self._deadline == 0.0:          # 首次进入：发送张开指令
                robot.open_gripper(GRIPPER_OPEN_M)
                print("[抓取] 张开夹爪...")
                self._deadline = now + GRIPPER_OPEN_WAIT
            elif now >= self._deadline:
                self._move_to(robot, self._approach, GraspState.APPROACH)

        elif s == GraspState.APPROACH:
            if now >= self._deadline:          # 接近点到达：下降至抓取位置
                self._move_to(robot, self._grasp, GraspState.DESCEND)

        elif s == GraspState.DESCEND:
            if now >= self._deadline:          # 抓取位置到达：发送夹紧指令
                robot.close_gripper()
                print("[抓取] 夹紧中，监测电机速度...")
                self._grasp_start = now
                self.state        = GraspState.GRASP
                self._deadline    = now + GRASP_DETECT_TIMEOUT

        elif s == GraspState.GRASP:
            if now - self._grasp_start < GRASP_STARTUP_DELAY:
                pass   # 启动阶段：等待电机开始运动，跳过速度检测
            else:
                pos, vel, torq = robot.get_gripper_state()
                if abs(vel) <= GRASP_STALL_VEL:
                    # 速度趋零 → 被物体阻挡，施加前馈使夹取力标准化
                    robot.hold_gripper_with_torque(GRASP_STD_TORQUE)
                    print(f"[抓取] 接触（vel={vel:.3f} rad/s, pos={pos:.3f} rad）"
                          f" → 标准力矩 {GRASP_STD_TORQUE} Nm，开始提升")
                    self._move_to(robot, self._lift, GraspState.LIFT)
                elif now >= self._deadline:    # 超时兜底
                    print(f"[抓取] 接触检测超时（vel={vel:.3f}, torq={torq:.3f}），继续提升")
                    self._move_to(robot, self._lift, GraspState.LIFT)

        elif s == GraspState.LIFT:
            if now >= self._deadline:
                print("[抓取] 完成！物体已提升。")
                self.state = GraspState.DONE

    def _move_to(self, robot, pos6: tuple, next_state: GraspState) -> None:
        """发送轨迹移动指令并切换到 next_state，IK 失败则回 IDLE。"""
        x, y, z, r, p, yw = pos6
        print(f"[抓取] → {next_state.value}  X={x:.3f} Y={y:.3f} Z={z:.3f} m")
        ok = robot.move_to(x, y, z, roll=r, pitch=p, yaw=yw, duration=GRASP_MOVE_DUR)
        if ok:
            self.state     = next_state
            self._deadline = time.monotonic() + GRASP_MOVE_DUR + 0.3
        else:
            print("[抓取] IK 失败，中止序列")
            self.state = GraspState.IDLE


# ==========================================
# 机器人接口初始化
# ==========================================
def init_robot(manual_mode: bool, dry_run: bool, cfg: dict):
    if manual_mode:
        print("[机器人] 手动模式（不连接机械臂，无法显示 Base 坐标）")
        return None

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root.parent))
    try:
        from cameraws.drivers.robot.rebot_arm import RebotArm
        robot_cfg = cfg.get("robot", {})
        robot = RebotArm(
            config_path=robot_cfg.get("config_path"),
            urdf_path=robot_cfg.get("urdf_path"),
            repo_root=robot_cfg.get("repo_root"),
        )
        robot.connect(enable=not dry_run)
        if dry_run:
            print("[机器人] 只读模式（电机未使能，仅读 FK 验证坐标映射）")
        else:
            print("[机器人] RebotArm 连接成功，电机已使能")
            # 初始化夹爪（可选，失败不影响主流程）
            gripper_cfg = robot_cfg.get("gripper_config_path")
            try:
                robot.init_gripper(gripper_cfg)
            except Exception as e:
                print(f"[机器人] 夹爪初始化失败（无夹爪运行）: {e}")
        return robot
    except Exception as e:
        print(f"[机器人] 连接失败: {e}")
        print("[机器人] 退回手动模式（仅显示相机坐标）")
        return None


# ==========================================
# 相机初始化
# ==========================================
def init_camera(cfg, root):
    """根据 config 实例化并打开相机驱动，返回 (cam, fx, fy, cx, cy)。"""
    sys.path.insert(0, str(root.parent))
    from cameraws.drivers.camera import make_camera
    cam = make_camera(cfg)
    cam.open()
    K = cam.K
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    cam_type = cfg.get("camera", {}).get("type", "")
    print(f"[相机就绪] {cam_type} (fx={fx:.1f}, cx={cx:.1f})")
    return cam, fx, fy, cx, cy


# ==========================================
# 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="机械臂目标夹取控制")
    parser.add_argument("--manual", action="store_true",
                        help="不连接机械臂，仅显示相机坐标")
    parser.add_argument("--dry-run", action="store_true",
                        help="连接机械臂但按 G 时不实际移动（调试用）")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root / "config" / "default.yaml")

    cam_type = cfg["camera"].get("type", "orbbec_gemini2").lower()

    # ── 手眼标定 ──
    hand_eye_path = root / "config" / "calibration" / cam_type / "hand_eye.npz"
    if not hand_eye_path.exists():
        print(f"[警告] 手眼标定文件不存在: {hand_eye_path}")
        print("       请先运行: python scripts/collect_handeye_eih.py")
        T_handeye, handeye_mode = None, None
    else:
        T_handeye, handeye_mode = load_hand_eye(str(hand_eye_path))

    # ── YOLO 模型 ──
    yolo_cfg = cfg.get("yolo", {})
    model_name    = yolo_cfg.get("model_name", "yolov8s-world.pt")
    device        = yolo_cfg.get("device", "cpu")
    use_world     = yolo_cfg.get("use_world", False)
    custom_classes = yolo_cfg.get("custom_classes", ["person", "cup"])

    model_path = root / "models" / model_name
    print(f"\n=== 加载 YOLO 模型: {model_name} ===")
    model = YOLO(str(model_path))

    is_open_vocab = use_world and ("world" in model_name.lower() or "yoloe" in model_name.lower())
    if is_open_vocab:
        model.set_classes(custom_classes)
        print(f"开放词汇模式，类别: {custom_classes}")

    # ── 机器人 ──
    robot     = init_robot(args.manual, args.dry_run, cfg)
    has_robot = robot is not None

    # ── 相机 ──
    print(f"\n=== 初始化相机: {cam_type} ===")
    try:
        cam, fx, fy, cx_cam, cy_cam = init_camera(cfg, root)
    except Exception as e:
        print(f"[错误] 相机初始化失败: {e}")
        sys.exit(1)

    if args.dry_run:
        print("\n[操作提示] 验证模式：电机未使能，机械臂不会运动")
        print("           查看窗口中 Bot 坐标确认手眼标定效果，Q/ESC 退出")
    else:
        print("\n[操作提示] G=开始抓取/中止  R=回零位  Q/ESC=退出")
        if has_robot and not robot.has_gripper:
            print("           (夹爪未连接，将跳过夹爪动作)")

    WIN = f"Grasp Control ({cam_type})"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    grasp_fsm = GraspFSM()

    best_target_robot = None
    _frame_cnt        = 0
    _PRINT_INTERVAL   = 30
    _frame_detections = []

    try:
        while True:
            # ── 先读 FK，再采帧，缩小时序误差 ──
            T_gripper2base = None
            if has_robot and handeye_mode == "eye_in_hand" and T_handeye is not None:
                try:
                    T_gripper2base = robot.get_tcp_pose()
                except Exception:
                    pass

            color_image, depth_mm = cam.get_frame()
            if color_image is None:
                continue

            if has_robot and handeye_mode == "eye_in_hand" and T_handeye is not None \
                    and T_gripper2base is None:
                cv2.putText(color_image, "FK Error", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # ── 推进抓取状态机 ──
            if has_robot and not args.dry_run:
                grasp_fsm.tick(robot)

            # ── YOLO 检测 ──
            results = model.predict(color_image, verbose=False, device=device)

            best_conf         = -1.0
            best_target_robot = None
            _frame_cnt       += 1
            _frame_detections.clear()

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id     = int(box.cls[0])
                    conf       = float(box.conf[0])
                    class_name = model.names[cls_id]

                    u = (x1 + x2) // 2
                    v = (y1 + y2) // 2

                    z_m = 0.0
                    if depth_mm is not None:
                        z_raw = get_depth_mm(depth_mm, u, v)
                        z_m = z_raw / 1000.0 if z_raw > 0 else 0.0

                    if z_m <= 0:
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(color_image, f"{class_name} (No Depth)",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
                        continue

                    p_cam = np.array([
                        (u - cx_cam) * z_m / fx,
                        (v - cy_cam) * z_m / fy,
                        z_m,
                    ], dtype=np.float64)

                    p_robot = None
                    if T_handeye is not None:
                        if handeye_mode == "eye_to_hand":
                            p_robot = cam_to_robot(p_cam, T_handeye, handeye_mode)
                        elif T_gripper2base is not None:
                            p_robot = cam_to_robot(p_cam, T_handeye, handeye_mode, T_gripper2base)

                    _frame_detections.append((class_name, conf, p_cam.copy(),
                                              p_robot.copy() if p_robot is not None else None))

                    if conf > best_conf:
                        best_conf         = conf
                        best_target_robot = p_robot.copy() if p_robot is not None else None

                    color_box = (0, 255, 0) if p_robot is not None else (0, 200, 200)
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), color_box, 2)
                    cv2.circle(color_image, (u, v), 5, (0, 0, 255), -1)

                    lines = [f"{class_name} {conf:.2f}"]
                    lines.append(f"Cam: X{p_cam[0]:.3f} Y{p_cam[1]:.3f} Z{p_cam[2]:.3f}m")
                    if p_robot is not None:
                        lines.append(f"Bot: X{p_robot[0]:.3f} Y{p_robot[1]:.3f} Z{p_robot[2]:.3f}m")

                    label_h = 18
                    bg_y0   = y1 - len(lines) * label_h - 4
                    bg_w    = max(len(ln) for ln in lines) * 9
                    cv2.rectangle(color_image, (x1, bg_y0), (x1 + bg_w, y1), (0, 0, 0), -1)
                    for i, ln in enumerate(lines):
                        lcolor = (0, 255, 0) if i == 0 else (0, 255, 255) if i == 1 else (100, 255, 100)
                        cv2.putText(color_image, ln,
                                    (x1 + 4, bg_y0 + (i + 1) * label_h - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, lcolor, 1)

            # ── 状态栏 ──
            status = f"{cam_type.upper()} | {model_name}"
            if has_robot and args.dry_run:
                status += " | 验证模式  [Q]=退出"
            elif has_robot:
                status += f" | [{grasp_fsm.state.value}]  [G]=抓取/中止  [R]=回零  [Q]=退出"
            else:
                status += " | 手动模式  [Q]=退出"
            cv2.putText(color_image, status, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 目标坐标底部显示
            if best_target_robot is not None:
                info = (f"目标→Base: X{best_target_robot[0]:.3f} "
                        f"Y{best_target_robot[1]:.3f} Z{best_target_robot[2]:.3f} m")
                cv2.putText(color_image, info, (10, color_image.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

            # 抓取状态高亮（非 IDLE/DONE 时）
            if grasp_fsm.active:
                label = f"[抓取中] {grasp_fsm.state.value}"
                cv2.putText(color_image, label,
                            (10, color_image.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

            cv2.imshow(WIN, color_image)

            # ── 终端周期打印（每 N 帧）──
            if _frame_cnt % _PRINT_INTERVAL == 0 and _frame_detections:
                print(f"\n[Frame {_frame_cnt}] 检测到 {len(_frame_detections)} 个目标:")
                for i, (cls, cf, pc, pr) in enumerate(_frame_detections):
                    cam_str = f"Cam X={pc[0]:+.3f} Y={pc[1]:+.3f} Z={pc[2]:+.3f} m"
                    if pr is not None:
                        bot_str = f"  Bot X={pr[0]:+.3f} Y={pr[1]:+.3f} Z={pr[2]:+.3f} m"
                    elif T_handeye is None:
                        bot_str = "  Bot (未加载手眼标定文件)"
                    elif not has_robot:
                        bot_str = "  Bot (未连接机械臂，无法读取 FK)"
                    else:
                        bot_str = "  Bot (FK 读取失败)"
                    print(f"  [{i+1}] {cls} ({cf:.2f})  {cam_str}{bot_str}")

            # ── 按键 ──
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key in [ord('q'), ord('Q'), 27]:
                break

            elif key in [ord('g'), ord('G')]:
                if grasp_fsm.active:
                    grasp_fsm.abort()
                elif not has_robot:
                    print("[G] 未连接机械臂，无法抓取")
                elif best_target_robot is None:
                    print("[G] 无有效目标（未检测到物体或手眼标定缺失）")
                elif args.dry_run:
                    x, y, z = best_target_robot
                    print(f"[G/Dry-run] 目标 Base: X={x:.4f} Y={y:.4f} Z={z:.4f} m（未实际移动）")
                else:
                    rpy = _get_tcp_rpy(robot)
                    print(f"  当前末端 RPY (rad): {rpy[0]:.3f} {rpy[1]:.3f} {rpy[2]:.3f}")
                    grasp_fsm.start(best_target_robot, rpy)

            elif key in [ord('r'), ord('R')]:
                if not has_robot:
                    print("[R] 未连接机械臂")
                elif grasp_fsm.active:
                    print("[R] 抓取进行中，请先按 G 中止")
                else:
                    print("[R] 回零位...")
                    try:
                        robot.safe_home()
                        print("  [✓] 已回零位")
                    except Exception as e:
                        print(f"  [✗] 回零位失败: {e}")

    finally:
        cam.close()
        cv2.destroyAllWindows()
        if has_robot:
            if robot.has_gripper:
                try:
                    print("[退出] 夹爪复位：张开...")
                    robot.open_gripper()
                    time.sleep(GRIPPER_OPEN_WAIT)
                    print("[退出] 夹爪复位：闭合...")
                    robot.close_gripper()
                    time.sleep(0.5)
                except Exception as e:
                    print(f"[退出] 夹爪复位失败: {e}")
            try:
                robot.disconnect()
            except Exception:
                pass
        print("\n退出。")


if __name__ == "__main__":
    main()
