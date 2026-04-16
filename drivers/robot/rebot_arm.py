"""
cameraws.drivers.robot.rebot_arm
=================================
轻量包装层，将 reBotArm_control_py 的低层 API 封装为
相机感知系统需要的简洁接口：

    connect()         — 使能电机
    disconnect()      — 失能并关闭
    get_tcp_pose()    — 通过 FK 读取末端位姿 (4×4 T_gripper2base)
    move_to(x,y,z)    — 通过 IK + 轨迹控制器移动末端
    safe_home()       — 回零位

依赖：~/seeed/reBotArm_control_py（pip install -e .）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


def _find_repo_root(hint: Optional[str] = None) -> Path:
    """查找 reBotArm_control_py 仓库根目录。"""
    candidates = []
    if hint:
        candidates.append(Path(hint).expanduser().resolve())
    candidates += [
        Path.home() / "seeed" / "reBotArm_control_py",
        Path("/home/chlorine/seeed/reBotArm_control_py"),
    ]
    for p in candidates:
        if (p / "reBotArm_control_py").is_dir():
            return p
    raise FileNotFoundError(
        "找不到 reBotArm_control_py 仓库，请在 config/default.yaml 中设置 "
        "robot.repo_root 或执行 pip install -e ~/seeed/reBotArm_control_py"
    )


class RebotArm:
    """
    相机感知系统 ↔ 机械臂接口。

    Args:
        config_path: robot.yaml 路径；None = 使用仓库默认
        urdf_path:   URDF 路径；None = 使用仓库默认
        repo_root:   reBotArm_control_py 仓库根目录；None = 自动搜索
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        urdf_path:   Optional[str] = None,
        repo_root:   Optional[str] = None,
    ) -> None:
        repo = _find_repo_root(repo_root)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        from reBotArm_control_py.actuator import RobotArm
        from reBotArm_control_py.kinematics import (
            load_robot_model,
            compute_fk,
            get_end_effector_frame_id,
        )
        from reBotArm_control_py.controllers import ArmEndPos

        cfg = str(config_path) if config_path else None
        self._arm = RobotArm(cfg_path=cfg)

        if urdf_path:
            self._model = load_robot_model(urdf_path=str(urdf_path))
        else:
            self._model = load_robot_model()

        self._data = self._model.createData()
        self._ee_frame_id = get_end_effector_frame_id(self._model)
        self._compute_fk = compute_fk

        self._endpos_ctrl: Optional[ArmEndPos] = None
        self._ArmEndPos = ArmEndPos

        self._connected    = False
        self._gripper_mot  = None  # 夹爪电机句柄，由 init_gripper() 注册
        self._gripper_kp   = 5.0   # MIT kp，从 gripper.yaml 加载
        self._gripper_kd   = 1.0   # MIT kd，从 gripper.yaml 加载
        self._gripper_ctrl = None  # 共用的 Controller 引用

    # ── 生命周期 ─────────────────────────────────────────────────────────────

    def connect(self, enable: bool = True) -> None:
        """连接机械臂。

        Args:
            enable: True = 使能电机（运动控制用）
                    False = 仅读取编码器，电机保持失能（只读模式）
        """
        self._arm.connect()
        if enable:
            self._arm.enable()
            time.sleep(0.5)
            self._endpos_ctrl = self._ArmEndPos(self._arm)
            self._endpos_ctrl.start()
            print("[RebotArm] 连接成功，电机已使能")
        else:
            self._arm._request_and_poll()
            print("[RebotArm] 连接成功，电机保持失能（只读模式）")
        self._connected = True

    def disconnect(self) -> None:
        """停止控制器，失能电机，关闭连接。"""
        if self._endpos_ctrl is not None:
            try:
                self._endpos_ctrl.end()
            except Exception:
                pass
            self._endpos_ctrl = None
        if self._gripper is not None:
            try:
                self._gripper.disconnect()
            except Exception:
                pass
            self._gripper = None
        try:
            self._arm.disconnect()
        except Exception:
            pass
        self._connected = False
        print("[RebotArm] 已断开连接")

    # ── 夹爪 ─────────────────────────────────────────────────────────────────
    # 夹爪电机直接注册到机械臂已有的 CAN Controller，共用同一条总线，
    # 不再实例化独立的 Gripper 类，避免同端口双重连接冲突。

    # 夹爪物理参数
    _GRIPPER_MAX_DIST_M = 0.09   # 最大开合距离 (m)
    _GRIPPER_ANGLE_OPEN = -5.0   # 完全张开对应电机角度 (rad)
    # 映射：0 rad = 闭合(0 cm)，-5 rad = 完全张开(9 cm)

    def init_gripper(self, cfg_path: Optional[str] = None) -> None:
        """将夹爪电机注册到机械臂已有的 CAN 总线控制器。

        夹爪和机械臂共用同一个 Controller 实例，不另开串口连接。

        Args:
            cfg_path: gripper.yaml 路径；None = 使用 reBotArm_control_py/config/gripper.yaml
        """
        from reBotArm_control_py.actuator.gripper import load_cfg as load_gripper_cfg
        from motorbridge import Mode, CallError

        if cfg_path is None:
            repo = _find_repo_root()
            cfg_path = str(repo / "config" / "gripper.yaml")

        gcfg = load_gripper_cfg(cfg_path)
        gc = gcfg["gripper"]

        # 复用机械臂已有的 Controller（同 vendor 则共用，否则报错）
        vendor = gc.vendor
        if vendor not in self._arm._ctrl_map:
            raise RuntimeError(
                f"夹爪 vendor={vendor!r} 与机械臂 vendor 不同，"
                "无法共用 Controller，请确认配置文件"
            )
        ctrl = self._arm._ctrl_map[vendor]

        # 注册夹爪电机到已有控制器
        if vendor == "damiao":
            self._gripper_mot = ctrl.add_damiao_motor(
                gc.motor_id, gc.feedback_id, gc.model
            )
        elif vendor == "myactuator":
            self._gripper_mot = ctrl.add_myactuator_motor(
                gc.motor_id, gc.feedback_id, gc.model
            )
        elif vendor == "robstride":
            self._gripper_mot = ctrl.add_robstride_motor(
                gc.motor_id, gc.feedback_id, gc.model
            )
        else:
            raise ValueError(f"不支持的夹爪 vendor: {vendor!r}")

        self._gripper_kp   = gc.kp   # ← gripper.yaml MIT.kp
        self._gripper_kd   = gc.kd   # ← gripper.yaml MIT.kd
        self._gripper_ctrl = ctrl    # 保存引用，用于 poll

        # 使能并切换 MIT 模式
        try:
            ctrl.enable_all()
            time.sleep(0.3)
        except CallError as e:
            print(f"[RebotArm] 夹爪使能警告: {e}")
        try:
            self._gripper_mot.ensure_mode(Mode.MIT, 1000)
        except CallError as e:
            raise RuntimeError(f"夹爪 MIT 模式切换失败: {e}") from e

        print("[RebotArm] 夹爪已注册到 CAN 总线 (MIT)")

    @property
    def has_gripper(self) -> bool:
        return self._gripper_mot is not None

    def set_gripper_opening(self, distance_m: float) -> None:
        """按开合距离控制夹爪（MIT 模式，非阻塞）。

        Args:
            distance_m: 开合距离（米），0.0 = 完全夹紧，0.09 = 完全张开
        """
        if self._gripper_mot is None:
            return
        d = float(np.clip(distance_m, 0.0, self._GRIPPER_MAX_DIST_M))
        angle = (d / self._GRIPPER_MAX_DIST_M) * self._GRIPPER_ANGLE_OPEN
        try:
            self._gripper_mot.send_mit(
                float(angle), 0.0,
                float(self._gripper_kp), float(self._gripper_kd), 0.0,
            )
            self._gripper_mot.request_feedback()
            self._gripper_ctrl.poll_feedback_once()
        except Exception as e:
            print(f"[RebotArm] 夹爪指令发送失败: {e}")

    def open_gripper(self, distance_m: float = 0.09) -> None:
        """张开夹爪（非阻塞）。

        Args:
            distance_m: 张开距离（米），默认完全张开 0.09 m
        """
        self.set_gripper_opening(distance_m)

    def close_gripper(self) -> None:
        """夹紧夹爪（非阻塞）。"""
        self.set_gripper_opening(0.0)

    def get_gripper_state(self) -> tuple:
        """读取夹爪电机状态。

        Returns:
            (pos_rad, vel_rad_s, torq_nm)，未初始化时返回 (0.0, 0.0, 0.0)。
        """
        if self._gripper_mot is None:
            return (0.0, 0.0, 0.0)
        try:
            self._gripper_mot.request_feedback()
            self._gripper_ctrl.poll_feedback_once()
            st = self._gripper_mot.get_state()
            if st is not None:
                return (float(st.pos), float(st.vel), float(st.torq))
        except Exception:
            pass
        return (0.0, 0.0, 0.0)

    def hold_gripper_with_torque(self, target_torque_nm: float) -> None:
        """接触后施加前馈扭矩，将夹取力标准化为 target_torque_nm。

        原理：MIT 总输出 = kp*(q_des - q_actual) + tau_ff
             令总输出 = target_torque_nm，解得 tau_ff：
             tau_ff = target_torque_nm - kp*(0 - q_contact)
                    = target_torque_nm + kp*q_contact   （q_contact < 0）

        Args:
            target_torque_nm: 目标夹取力矩（Nm），根据应用场景调整
        """
        if self._gripper_mot is None:
            return
        try:
            self._gripper_mot.request_feedback()
            self._gripper_ctrl.poll_feedback_once()
            st = self._gripper_mot.get_state()
            if st is None:
                return
            q_contact = float(st.pos)
            tau_ff = target_torque_nm - self._gripper_kp * (0.0 - q_contact)
            self._gripper_mot.send_mit(
                0.0, 0.0,
                float(self._gripper_kp), float(self._gripper_kd),
                float(tau_ff),
            )
            self._gripper_mot.request_feedback()
            self._gripper_ctrl.poll_feedback_once()
        except Exception as e:
            print(f"[RebotArm] 夹爪前馈控制失败: {e}")

    # ── 状态读取 ─────────────────────────────────────────────────────────────

    def get_tcp_pose(self) -> np.ndarray:
        """通过正运动学读取当前末端位姿。

        Returns:
            T_gripper2base: (4, 4) 齐次变换矩阵
        """
        self._arm._request_and_poll()
        q, _, _ = self._arm.get_state()
        position, rotation, _ = self._compute_fk(self._model, q)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotation
        T[:3,  3] = position
        return T

    # ── 运动控制 ─────────────────────────────────────────────────────────────

    def move_to(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        duration: float = 2.0,
    ) -> bool:
        """通过 IK + 轨迹规划将末端移动到目标位置。"""
        if self._endpos_ctrl is None:
            raise RuntimeError("未连接机械臂，请先调用 connect()")
        return bool(self._endpos_ctrl.move_to_traj(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, duration=duration,
        ))

    def safe_home(self, duration: float = 3.0) -> None:
        """回零位（关节全部归零）。"""
        if self._endpos_ctrl is None:
            raise RuntimeError("未连接机械臂，请先调用 connect()")
        q_zero = np.zeros(self._arm.num_joints, dtype=np.float64)
        pos_zero, _, _ = self._compute_fk(self._model, q_zero)
        ok = self._endpos_ctrl.move_to_traj(
            x=float(pos_zero[0]), y=float(pos_zero[1]), z=float(pos_zero[2]),
            duration=duration,
        )
        if not ok:
            self._arm.mode_pos_vel()
            self._arm.pos_vel(q_zero)
            time.sleep(duration)

    # ── 上下文管理器 ─────────────────────────────────────────────────────────

    def __enter__(self) -> "RebotArm":
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()
