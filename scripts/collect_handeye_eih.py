"""
Eye-in-Hand 手眼标定 — 数据采集与计算（Gemini2 + reBotArm）

【模式】
  自动模式（默认）：机械臂自动移动到 36 个预设姿态，每个姿态按 Enter 采集。
  手动模式（--manual）：重力补偿控制，用户手动推动机械臂到任意位置，
                        放手后臂自动锁定，按 Enter 采集。

【布置方式】
  相机装在机械臂末端（随末端运动）
  ArUco 标记贴在工作台固定位置（不动）

【用法】
    cd /home/chlorine/seeed/cameraws
    python scripts/collect_handeye_eih.py           # 自动模式
    python scripts/collect_handeye_eih.py --manual  # 手动重力补偿模式
"""

import os
import sys
import math
import threading
import argparse
import queue
import time
import cv2
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
_REBOT_ROOT = Path(__file__).resolve().parent.parent.parent / "reBotArm_control_py"
if _REBOT_ROOT.exists() and str(_REBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_REBOT_ROOT))

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

from cameraws.drivers.camera import make_camera
from cameraws.calibration.hand_eye import CalibMode, HandEyeCalibrator


# ==========================================
# 预设标定姿态（笛卡尔空间，单位：米/弧度）
# (x, y, z, roll, pitch, yaw)
# pitch > 0 = 末端朝斜下方（相机俯视 ArUco）
# ==========================================
CALIB_POSES_XYZ = [
    # ── 中心区域，大 pitch，yaw 扫描 ──
    (0.28, -0.16, 0.26, -0.30, 0.80, -0.90),
    (0.28, -0.08, 0.26,  0.30, 0.80, -0.45),
    (0.28,  0.00, 0.26, -0.30, 0.80,  0.00),
    (0.28,  0.08, 0.26,  0.30, 0.80,  0.45),
    (0.28,  0.16, 0.26, -0.30, 0.80,  0.90),
    # ── 中心区域，中 pitch ──
    (0.27, -0.16, 0.31,  0.30, 0.55, -0.90),
    (0.27, -0.08, 0.31, -0.30, 0.55, -0.45),
    (0.27,  0.00, 0.31,  0.30, 0.55,  0.00),
    (0.27,  0.08, 0.31, -0.30, 0.55,  0.45),
    (0.27,  0.16, 0.31,  0.30, 0.55,  0.90),
    # ── 中心区域，小 pitch ──
    (0.26, -0.14, 0.34, -0.40, 0.35, -0.80),
    (0.26,  0.00, 0.34,  0.40, 0.35,  0.00),
    (0.26,  0.14, 0.34, -0.40, 0.35,  0.80),
    # ── 偏前方，yaw ±1 rad ──
    (0.37,  0.00, 0.27,  0.00, 0.65,  0.00),
    (0.37,  0.00, 0.27,  0.00, 0.65,  1.00),
    (0.37,  0.00, 0.27,  0.00, 0.65, -1.00),
    (0.37,  0.08, 0.27,  0.50, 0.65,  0.50),
    (0.37, -0.08, 0.27, -0.50, 0.65, -0.50),
    # ── 侧向，x 大 ──
    (0.33,  0.18, 0.27,  0.50, 0.50,  0.55),
    (0.33, -0.18, 0.27, -0.50, 0.50, -0.55),
    # ── 侧向，y 大 ──
    (0.20,  0.22, 0.28,  0.60, 0.40,  0.70),
    (0.20, -0.22, 0.28, -0.60, 0.40, -0.70),
    # ── 斜前偏 y，roll 大 ──
    (0.24, -0.20, 0.31,  0.70, 0.45, -1.00),
    (0.24,  0.20, 0.31, -0.70, 0.45,  1.00),
    (0.25, -0.15, 0.29, -0.60, 0.62, -0.50),
    (0.25,  0.15, 0.29,  0.60, 0.62,  0.50),
    # ── 高位 ──
    (0.21, -0.09, 0.40, -0.40, 0.25, -0.60),
    (0.21,  0.00, 0.40,  0.40, 0.25,  0.00),
    (0.21,  0.09, 0.40, -0.40, 0.25,  0.60),
    (0.20, -0.09, 0.40,  0.40, 0.28, -0.60),
    (0.20,  0.09, 0.40, -0.40, 0.28,  0.60),
    # ── 低位 ──
    (0.30, -0.10, 0.24,  0.40, 0.70, -0.70),
    (0.30,  0.00, 0.24,  0.00, 0.75,  0.00),
    (0.30,  0.10, 0.24, -0.40, 0.70,  0.70),
    # ── roll 极值 ──
    (0.26,  0.12, 0.30,  0.80, 0.50,  0.30),
    (0.26, -0.12, 0.30, -0.80, 0.50, -0.30),
    # ── 大 roll + 大 yaw 组合（旋转多样性补充）──
    (0.29, -0.10, 0.28,  0.90, 0.60, -0.40),
    (0.29,  0.10, 0.28, -0.90, 0.60,  0.40),
    (0.28, -0.18, 0.30,  0.85, 0.55, -0.80),
    (0.28,  0.18, 0.30, -0.85, 0.55,  0.80),
    # ── 前伸 + 不同 roll/yaw 组合 ──
    (0.35, -0.12, 0.30,  0.60, 0.58, -0.70),
    (0.35,  0.12, 0.30, -0.60, 0.58,  0.70),
    (0.34, -0.06, 0.28,  0.40, 0.72,  0.80),
    (0.34,  0.06, 0.28, -0.40, 0.72, -0.80),
    # ── 高位 + 大 roll（当前高位姿态 roll 偏小）──
    (0.22, -0.14, 0.38,  0.75, 0.32, -0.70),
    (0.22,  0.14, 0.38, -0.75, 0.32,  0.70),
    # ── 侧向 + 大 pitch + 反向 yaw ──
    (0.31,  0.20, 0.27,  0.30, 0.68,  0.95),
    (0.31, -0.20, 0.27, -0.30, 0.68, -0.95),
    # ── 中距离，roll/pitch/yaw 三轴均衡覆盖 ──
    (0.30,  0.05, 0.32,  0.75, 0.42,  0.65),
    (0.30, -0.05, 0.32, -0.75, 0.42, -0.65),
]


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_input_thread(line_queue: queue.Queue) -> threading.Thread:
    def _loop():
        while True:
            try:
                line_queue.put(input())
            except EOFError:
                line_queue.put(None)
                break
            except KeyboardInterrupt:
                line_queue.put(None)
                break
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


# ==========================================
# 重力补偿控制器（手动模式）
# ==========================================
class GravityCompController:
    """MIT 模式 + 末端速度锁止，用于手动推臂定位。

    参考 reBotArm_control_py/example/10_gravity_compensation_lock.py
    """
    KP = 8.0
    KD = 1.5
    V_THRESH  = 0.04   # 末端线速度阈值 (m/s)
    W_THRESH  = 0.08   # 末端角速度阈值 (rad/s)

    def __init__(self) -> None:
        from reBotArm_control_py.actuator import RobotArm
        from reBotArm_control_py.dynamics import compute_generalized_gravity
        from reBotArm_control_py.kinematics import (
            load_robot_model, compute_fk, get_end_effector_frame_id,
        )
        import pinocchio as pin

        self._compute_gravity = compute_generalized_gravity
        self._compute_fk = compute_fk
        self._pin = pin

        self._arm = RobotArm()
        self._model = load_robot_model()
        self._data  = self._model.createData()
        self._ee_id = get_end_effector_frame_id(self._model)

        self._n = None          # 关节数（connect 后确定）
        self._q_target = None   # 锁止目标，list[ndarray] 供线程共享
        self._integral = None
        self._gc_running = threading.Event()
        self._gc_thread: threading.Thread | None = None

    def start(self) -> None:
        """连接、使能、切换 MIT 模式，启动重力补偿线程。"""
        self._arm.connect()
        print("[GravityComp] 已连接")
        self._arm.enable()
        print("[GravityComp] 已使能")

        n = self._arm.num_joints
        self._n = n
        q0 = self._arm.get_positions(request=True)
        self._q_target = [q0.copy()]
        self._integral = [np.zeros(n)]

        self._arm.mode_mit(
            kp=np.full(n, self.KP),
            kd=np.full(n, self.KD),
        )
        print(f"[GravityComp] MIT 模式，kp={self.KP} kd={self.KD}，可手动推臂")

        self._gc_running.set()
        self._gc_thread = threading.Thread(target=self._worker, daemon=True)
        self._gc_thread.start()

    def safe_home(self) -> None:
        """停止重力补偿，用 ArmEndPos 轨迹回零位，然后断开连接。"""
        self._gc_running.clear()
        if self._gc_thread is not None:
            self._gc_thread.join(timeout=1.0)
        try:
            print("[GravityComp] 回零位中...")
            from reBotArm_control_py.controllers import ArmEndPos
            ctrl = ArmEndPos(self._arm)
            ctrl.start()
            ctrl.end()
        except Exception as e:
            print(f"[GravityComp] 回零位失败: {e}")
        try:
            self._arm.disconnect()
        except Exception:
            pass
        print("[GravityComp] 已断开")

    def stop(self) -> None:
        """停止重力补偿线程并断开连接（不复位）。"""
        self._gc_running.clear()
        if self._gc_thread is not None:
            self._gc_thread.join(timeout=1.0)
        try:
            self._arm.disconnect()
        except Exception:
            pass
        print("[GravityComp] 已断开")

    def get_tcp_pose(self) -> np.ndarray:
        """读取当前末端位姿（4×4 T_gripper2base）。"""
        q = self._arm.get_positions()
        pos, rot, _ = self._compute_fk(self._model, q)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3,  3] = pos
        return T

    def _worker(self) -> None:
        pin = self._pin
        model, data, ee_id = self._model, self._data, self._ee_id
        n = self._n
        KP, KD = self.KP, self.KD

        while self._gc_running.is_set():
            try:
                q  = self._arm.get_positions()
                qd = self._arm.get_velocities()
                tau_g = self._compute_gravity(q=q)

                q_err = self._q_target[0] - q
                self._integral[0] += q_err * 1.0
                np.clip(self._integral[0], -0.5, 0.5, out=self._integral[0])

                pin.computeJointJacobians(model, data, q)
                pin.updateFramePlacements(model, data)
                J = pin.getFrameJacobian(model, data, ee_id, pin.ReferenceFrame.WORLD)
                v = J @ qd

                if (np.linalg.norm(v[:3]) > self.V_THRESH or
                        np.linalg.norm(v[3:]) > self.W_THRESH):
                    self._q_target[0] = q.copy()
                    self._integral[0] *= 0.9

                self._arm.mit(
                    pos=self._q_target[0],
                    vel=np.zeros(n),
                    kp=np.full(n, KP),
                    kd=np.full(n, KD),
                    tau=tau_g + self._integral[0],
                )
            except Exception:
                pass
            time.sleep(0.002)


# ==========================================
# 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Eye-in-Hand 手眼标定采集")
    parser.add_argument("--manual", action="store_true",
                        help="手动模式：重力补偿，用户推臂到目标位置后按 Enter 采集")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg  = load_config(root / "config" / "default.yaml")

    cam_type   = cfg["camera"]["type"]
    calib_dir  = root / "config" / "calibration" / cam_type
    aruco_cfg  = cfg["calibration"]["aruco"]
    he_method  = cfg["calibration"].get("hand_eye_method", "TSAI")
    save_path  = calib_dir / "hand_eye.npz"

    # ── 相机 ──
    cam = make_camera(cfg)
    cam.setup_aruco(
        marker_length_m=aruco_cfg["marker_length_m"],
        dict_id=aruco_cfg.get("dict_id", 0),
        target_marker_id=aruco_cfg.get("target_marker_id"),
    )

    # ── 标定器 ──
    calibrator = HandEyeCalibrator(CalibMode.EYE_IN_HAND, method=he_method)

    # ── 机器人 ──
    mode_str  = "手动（重力补偿）" if args.manual else f"自动（{len(CALIB_POSES_XYZ)} 个预设姿态）"
    gc_ctrl: GravityCompController | None = None
    arm_ctrl  = [None]   # ArmEndPos instance（自动模式）
    _pose_idx = [0]

    robot_cfg = cfg.get("robot", {})
    try:
        if args.manual:
            gc_ctrl = GravityCompController()
            gc_ctrl.start()
            print(f"[机器人] 手动模式就绪 — 推动机械臂定位后按 Enter 采集")
        else:
            from cameraws.drivers.robot.rebot_arm import RebotArm
            from reBotArm_control_py.controllers import ArmEndPos as _ArmEndPos
            robot = RebotArm(
                config_path=robot_cfg.get("config_path"),
                urdf_path=robot_cfg.get("urdf_path"),
                repo_root=robot_cfg.get("repo_root"),
            )
            ctrl = _ArmEndPos(robot._arm)
            ctrl.start()
            arm_ctrl[0] = ctrl
            print(f"[机器人] 自动模式就绪，共 {len(CALIB_POSES_XYZ)} 个预设姿态，按 n 开始移动")
    except Exception as e:
        print(f"[机器人] 连接失败: {e}")
        sys.exit(1)

    print(f"\n=== Eye-in-Hand 手眼标定 ===")
    print(f"相机: {cam_type}  |  模式: {mode_str}  |  算法: {he_method}")
    print(f"ArUco 边长: {aruco_cfg['marker_length_m']*100:.0f}cm  |  保存: {save_path}")
    print()
    print("【操作】Enter=采集  c=计算保存  pos=当前末端位置  q=退出")
    if not args.manual:
        print(f"【移动】n=下一个预设姿态  m x y z [roll pitch yaw]=手动指定位置（米/弧度）")
    print()

    # ── 开启相机 ──
    cam.open()
    print("预热相机...", end="", flush=True)
    cam.warm_up(20)
    print(" 就绪\n")

    latest_pose = [None]
    line_queue: queue.Queue = queue.Queue()
    make_input_thread(line_queue)

    def _get_fk() -> np.ndarray:
        """读取当前末端位姿（两种模式均适用）。"""
        if args.manual:
            return gc_ctrl.get_tcp_pose()
        else:
            return robot.get_tcp_pose()

    def handle_line(raw: str) -> bool:
        if raw is None:
            print("\n[中断] 退出")
            return True

        line = raw.strip().lower()

        if line == "q":
            return True

        # pos → 显示当前末端 FK
        if line == "pos":
            try:
                T = _get_fk()
                t = T[:3, 3]
                R = T[:3, :3]
                _p = math.atan2(-R[2, 0], math.sqrt(R[0, 0]**2 + R[1, 0]**2))
                _r = math.atan2(R[2, 1] / math.cos(_p), R[2, 2] / math.cos(_p))
                _y = math.atan2(R[1, 0] / math.cos(_p), R[0, 0] / math.cos(_p))
                print(f"  FK: x={t[0]:+.3f} y={t[1]:+.3f} z={t[2]:+.3f} m"
                      f"  rpy=[{_r:+.2f} {_p:+.2f} {_y:+.2f}] rad")
            except Exception as e:
                print(f"  [错误] {e}")
            return False

        # n → 下一个预设姿态（仅自动模式）
        if line == "n" and arm_ctrl[0] is not None:
            idx = _pose_idx[0] % len(CALIB_POSES_XYZ)
            x, y, z, roll, pitch, yaw = CALIB_POSES_XYZ[idx]
            _pose_idx[0] += 1
            duration = 3.0
            print(f"  -> 姿态 {idx+1}/{len(CALIB_POSES_XYZ)}: "
                  f"pos=({x:.2f},{y:.2f},{z:.2f}) rpy=({roll:.2f},{pitch:.2f},{yaw:.2f}) ...")
            ok = arm_ctrl[0].move_to_traj(x=x, y=y, z=z,
                                           roll=roll, pitch=pitch, yaw=yaw,
                                           duration=duration)
            if ok:
                time.sleep(duration + 0.5)
                print("  到位，可按 Enter 采集")
            else:
                print("  IK 无解，输入 n 跳到下一个")
            return False

        # m x y z [roll pitch yaw] → 手动笛卡尔位姿（仅自动模式）
        if line.startswith("m ") and arm_ctrl[0] is not None:
            try:
                vals = [float(v) for v in raw.strip().split()[1:]]
                if len(vals) not in (3, 6):
                    print("  格式: m x y z [roll pitch yaw]")
                    return False
                x, y, z = vals[0], vals[1], vals[2]
                roll  = vals[3] if len(vals) == 6 else 0.0
                pitch = vals[4] if len(vals) == 6 else 0.0
                yaw   = vals[5] if len(vals) == 6 else 0.0
                ok = arm_ctrl[0].move_to_traj(x=x, y=y, z=z,
                                               roll=roll, pitch=pitch, yaw=yaw,
                                               duration=3.0)
                if ok:
                    time.sleep(3.5)
                    print("  到位，可按 Enter 采集")
                else:
                    print("  IK 无解")
            except Exception as e:
                print(f"  [错误] {e}")
            return False

        # c → 计算并保存
        if line == "c":
            if calibrator.n_samples < 5:
                print(f"  样本数不足（{calibrator.n_samples} < 5），继续采集")
                return False
            print(f"\n  计算中（{calibrator.n_samples} 个样本）...")
            try:
                result = calibrator.calibrate(min_samples=5)
                HandEyeCalibrator.save(result, save_path)
                t = result.T_result[:3, 3]
                R = result.T_result[:3, :3]
                print(f"  T_cam2gripper 平移: x={t[0]:.4f} y={t[1]:.4f} z={t[2]:.4f} m")
                print(f"  旋转矩阵:\n{R}")
                print(f"  [OK] 已保存至 {save_path}")
                if calibrator.n_samples < 15:
                    print("  提示：样本 < 15，建议继续采集以提高精度")
            except Exception as e:
                print(f"  [错误] 计算失败: {e}")
            return False

        # Enter（空行）→ 采集
        cur = latest_pose[0]
        if cur is None:
            print("  [跳过] 标记不在视野中，请调整后重试")
            return False

        print(f"\n[样本 {calibrator.n_samples + 1}]")
        print(f"  ArUco: x={cur.T_marker2cam[0,3]:.3f} "
              f"y={cur.T_marker2cam[1,3]:.3f} "
              f"z={cur.T_marker2cam[2,3]:.3f} m")
        try:
            T_g2b = _get_fk()
            t = T_g2b[:3, 3]
            print(f"  末端 (FK): x={t[0]:.4f} y={t[1]:.4f} z={t[2]:.4f} m")
            calibrator.add_sample(T_g2b, cur.T_marker2cam)
            print(f"  [OK] 已记录，共 {calibrator.n_samples} 个样本"
                  + ("  <- 可输入 c 计算" if calibrator.n_samples >= 15 else ""))
        except Exception as e:
            print(f"  [错误] 获取末端位姿失败: {e}")
        return False

    # ── 主循环 ──
    WIN = "Eye-in-Hand Calibration  (operate in terminal)"
    try:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

        while True:
            bgr, _ = cam.get_frame()
            if bgr is not None:
                pose = cam.detect_aruco(bgr)
                latest_pose[0] = pose
                vis  = cam.draw_aruco(bgr)
                n    = calibrator.n_samples

                def osd(text, y, color=(220, 220, 220)):
                    cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, color, 1, cv2.LINE_AA)

                if pose:
                    osd(f"[ID={pose.id}] z={pose.T_marker2cam[2,3]:.3f}m  "
                        f"samples:{n}  Enter=capture  c=compute  q=quit",
                        28, (80, 220, 80))
                else:
                    osd(f"No marker  samples:{n}  move arm to see marker",
                        28, (80, 80, 220))

                filled = min(n, 15) * (400 // 15)
                cv2.rectangle(vis, (10, 40), (10 + filled, 52), (0, 200, 100), -1)
                cv2.rectangle(vis, (10, 40), (410, 52), (160, 160, 160), 1)
                cv2.putText(vis, f"{n}/15", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                mode_label = "MANUAL(GravComp)" if args.manual else "AUTO"
                cv2.putText(vis, mode_label, (vis.shape[1] - 200, vis.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 60), 1)

                cv2.imshow(WIN, vis)

            if cv2.waitKey(30) & 0xFF in [ord('q'), ord('Q'), 27]:
                break
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break

            try:
                if handle_line(line_queue.get_nowait()):
                    break
            except queue.Empty:
                pass

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 退出")

    finally:
        cv2.destroyAllWindows()
        cam.close()
        if gc_ctrl is not None:
            gc_ctrl.safe_home()
        elif arm_ctrl[0] is not None:
            try:
                arm_ctrl[0].end()
            except (Exception, KeyboardInterrupt):
                try:
                    arm_ctrl[0].arm.disconnect()
                except Exception:
                    pass

    print(f"\n结束，共 {calibrator.n_samples} 个样本。")
    if calibrator.n_samples > 0 and not save_path.exists():
        print("提示：标定结果未保存，可重新运行后输入 c 计算。")


if __name__ == "__main__":
    main()
    os._exit(0)
