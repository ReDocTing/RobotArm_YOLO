"""
cameraws.drivers.robot.rebot_gripper
=====================================
独立夹爪驱动，基于 reBotArm_control_py.actuator.Gripper。

纯力矩推进闭合 + 速度阈值接触检测 + 前馈力控保持。
通过 RebotArm.gripper 属性访问，共享生命周期。

接口:
    open(distance_m)  — 张开到指定宽度
    close()           — 纯力矩闭合（非阻塞）
    grasp(force)      — 柔性夹取：闭合 → 接触检测 → 力控保持（阻塞）
    release()         — 张开并回零
    get_state()       — 读取 (pos, vel, torq)
    set_zero()        — 设置当前位置为零点
"""

from __future__ import annotations

import time
import threading
from typing import Optional

import numpy as np


# ── 物理参数 & 控制常量 ──────────────────────────────────────────
MAX_DIST_M      = 0.09    # 最大开合距离 (m)
ANGLE_OPEN      = -5.0    # 完全张开对应电机角度 (rad)
OPEN_SOFT_LIMIT = -4.9    # 软限位 (rad)
ARRIVE_TOL      = 0.12    # 到位判定容差 (rad)
HARD_STOP_ANGLE = -0.05   # 大于此角度视为硬止位（空夹取）
TAU_MAX         = 1.5     # 安全力矩上限 (Nm)

# 张开 / 回零
KP_MOVE   = 5.0
KD_MOVE   = 1.0
OPEN_RATE = 4.0           # 渐进张开速度 (rad/s)

# 闭合（纯力矩推进）
CLOSE_TORQUE  = 0.5       # 闭合推进力矩 (Nm)
KD_CLOSE      = 0.5
STALL_VEL     = 0.05      # 接触判定速度阈值 (rad/s)
STARTUP_DIST  = 0.30      # 行程保护 (rad)

# 力控保持
KP_HOLD = 5.0
KD_HOLD = 1.0
DEFAULT_FORCE = 0.30      # 默认夹取力矩 (Nm)

CTRL_RATE = 500.0         # 控制循环频率 (Hz)


class _State:
    IDLE    = 0
    OPENING = 1
    CLOSING = 2
    CONTACT = 3
    HOLDING = 4
    HOMING  = 5


class RebotGripper:
    """力反馈柔性夹爪驱动。

    不独立管理连接——由外部创建 Gripper 对象后传入，共享生命周期。

    Args:
        gripper_obj: reBotArm_control_py.actuator.Gripper 实例
    """

    def __init__(self, gripper_obj) -> None:
        self._g = gripper_obj
        self._lock  = threading.Lock()
        self._state = _State.IDLE
        self._target_force = DEFAULT_FORCE

        self._pos_start       = 0.0
        self._q_contact       = 0.0
        self._contact_elapsed = 0.0
        self._open_q_des      = OPEN_SOFT_LIMIT
        self._open_target     = OPEN_SOFT_LIMIT

        self._pos  = 0.0
        self._vel  = 0.0
        self._torq = 0.0
        self._loop_running = False

    # ── 内部：安全 MIT 发包 ───────────────────────────────────────
    def _safe_mit(self, gripper, pos, vel, kp, kd, tau_ff=0.0):
        pos_cmd  = float(np.clip(pos, OPEN_SOFT_LIMIT, 0.0))
        pos_term = kp * (pos_cmd - self._pos) + kd * (-self._vel)
        tau_total = float(np.clip(pos_term + tau_ff, -TAU_MAX, TAU_MAX))
        tau_ff_safe = tau_total - pos_term
        gripper.mit(pos=pos_cmd, vel=float(vel),
                    kp=float(kp), kd=float(kd),
                    tau=float(tau_ff_safe))

    # ── 内部：控制循环回调 ────────────────────────────────────────
    def _ctrl(self, gripper, dt):
        pos, vel, torq = gripper.get_state(request=False)
        self._pos  = pos
        self._vel  = vel
        self._torq = torq

        with self._lock:
            s  = self._state
            tf = self._target_force

        if s == _State.OPENING:
            with self._lock:
                self._open_q_des = max(
                    self._open_q_des - OPEN_RATE * dt, self._open_target)
                q = self._open_q_des
            self._safe_mit(gripper, q, 0.0, KP_MOVE, KD_MOVE)
            if abs(pos - self._open_target) < ARRIVE_TOL:
                with self._lock:
                    self._state = _State.IDLE

        elif s == _State.CLOSING:
            self._safe_mit(gripper, 0.0, 0.0, 0.0, KD_CLOSE, CLOSE_TORQUE)
            with self._lock:
                ps = self._pos_start
            moved = abs(pos - ps)
            if moved >= STARTUP_DIST:
                if pos > HARD_STOP_ANGLE:
                    with self._lock:
                        self._state = _State.IDLE
                elif abs(vel) < STALL_VEL:
                    with self._lock:
                        self._q_contact       = pos
                        self._contact_elapsed = 0.0
                        self._state           = _State.CONTACT

        elif s == _State.CONTACT:
            with self._lock:
                qc = self._q_contact
            self._safe_mit(gripper, qc, 0.0, KP_HOLD, KD_HOLD)
            with self._lock:
                self._contact_elapsed += dt
                if self._contact_elapsed >= 0.02:
                    self._state = _State.HOLDING

        elif s == _State.HOLDING:
            with self._lock:
                qc = self._q_contact
            self._safe_mit(gripper, qc, 0.0, KP_HOLD, KD_HOLD, tf)

        elif s == _State.HOMING:
            self._safe_mit(gripper, 0.0, 0.0, KP_MOVE, KD_MOVE)
            if abs(pos) < ARRIVE_TOL:
                with self._lock:
                    self._state = _State.IDLE

    # ── 内部：启停控制循环 ────────────────────────────────────────
    def _ensure_loop(self):
        if not self._loop_running:
            self._g.start_control_loop(self._ctrl, rate=CTRL_RATE)
            self._loop_running = True

    def _wait_idle(self, timeout: float = 3.0) -> bool:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._lock:
                if self._state == _State.IDLE:
                    return True
            time.sleep(0.01)
        return False

    # ── 生命周期 ──────────────────────────────────────────────────
    def enable(self) -> bool:
        if not self._g.enable():
            return False
        if not self._g.mode_mit():
            self._g.disable()
            return False
        p, v, t = self._g.get_state()
        self._pos, self._vel, self._torq = p, v, t
        self._ensure_loop()
        return True

    def disable(self):
        if self._loop_running:
            self._g.stop_control_loop()
            self._loop_running = False
        try:
            self._g.mit(pos=self._pos, vel=0.0, kp=0.0, kd=KD_MOVE, tau=0.0)
        except Exception:
            pass
        self._g.disable()

    def disconnect(self):
        self.disable()
        self._g.disconnect()

    # ── 公开接口 ──────────────────────────────────────────────────

    def get_state(self) -> tuple[float, float, float]:
        """返回 (pos_rad, vel_rad_s, torq_nm)。"""
        return (self._pos, self._vel, self._torq)

    def get_opening_m(self) -> float:
        """返回当前开合距离（米）。"""
        return abs(self._pos) / abs(ANGLE_OPEN) * MAX_DIST_M

    @property
    def is_holding(self) -> bool:
        with self._lock:
            return self._state == _State.HOLDING

    def open(self, distance_m: float = MAX_DIST_M) -> None:
        """张开夹爪到指定宽度（阻塞，最多 3s）。"""
        self._ensure_loop()
        d = float(np.clip(distance_m, 0.0, MAX_DIST_M))
        target = max((d / MAX_DIST_M) * ANGLE_OPEN, OPEN_SOFT_LIMIT)
        with self._lock:
            self._open_target = target
            self._open_q_des  = self._pos
            self._state = _State.OPENING
        self._wait_idle(3.0)

    def close(self) -> None:
        """纯力矩闭合（非阻塞）。"""
        self._ensure_loop()
        with self._lock:
            self._pos_start = self._pos
            self._state = _State.CLOSING

    def grasp(self, force: Optional[float] = None, timeout: float = 5.0) -> bool:
        """柔性夹取：闭合 → 接触检测 → 力控保持（阻塞）。

        Returns:
            True = 成功夹取（HOLDING），False = 空夹取或超时
        """
        self._ensure_loop()
        if force is not None:
            with self._lock:
                self._target_force = float(np.clip(force, 0.05, TAU_MAX))
        with self._lock:
            self._pos_start = self._pos
            self._state = _State.CLOSING

        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._lock:
                s = self._state
            if s == _State.HOLDING:
                return True
            if s == _State.IDLE:
                return False
            time.sleep(0.01)
        with self._lock:
            self._state = _State.IDLE
        return False

    def release(self, timeout: float = 4.0) -> None:
        """张开夹爪并回零（阻塞）。"""
        self._ensure_loop()
        with self._lock:
            self._open_target = OPEN_SOFT_LIMIT
            self._open_q_des  = self._pos
            self._state = _State.OPENING
        self._wait_idle(2.0)
        with self._lock:
            self._state = _State.HOMING
        self._wait_idle(timeout)

    def set_zero(self) -> bool:
        """设置当前位置为零点（会先停控制循环）。"""
        if self._loop_running:
            self._g.stop_control_loop()
            self._loop_running = False
        ok = self._g.set_zero()
        if ok:
            self._g.enable()
            self._g.mode_mit()
            self._ensure_loop()
            with self._lock:
                self._state = _State.IDLE
        return ok

    def set_force(self, force: float) -> None:
        """设置目标夹取力矩 (Nm)。"""
        with self._lock:
            self._target_force = float(np.clip(force, 0.05, TAU_MAX))
