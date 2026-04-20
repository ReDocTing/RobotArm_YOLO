"""
力反馈柔性夹取控制脚本（基于 RebotGripper 驱动）
=================================================

用法:
    cd /home/chlorine/seeed/cameraws
    python scripts/gripper.py
    python scripts/gripper.py --cfg path/to/gripper.yaml

按键:
    g  — 柔性夹取
    o  — 张开夹爪
    +  — 增大目标力矩 0.05 Nm
    -  — 减小目标力矩 0.05 Nm
    s  — 打印当前状态
    z  — 设零（谨慎使用）
    q  — 退出（张开 → 回零）
"""

import sys
import argparse
from pathlib import Path

# ── sys.path ──────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
_seeed_root   = _project_root.parent
_repo         = _seeed_root / "reBotArm_control_py"
for _p in (_repo, _seeed_root):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from reBotArm_control_py.actuator import Gripper
except ImportError as e:
    print(f"[错误] 无法导入 reBotArm_control_py: {e}")
    print("  请确认仓库位于 ~/seeed/reBotArm_control_py 且已 pip install -e .")
    sys.exit(1)

from cameraws.drivers.robot.rebot_gripper import RebotGripper

FORCE_STEP = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description="力反馈柔性夹取控制脚本")
    parser.add_argument("--cfg", default=None,
                        help="gripper.yaml 路径；默认 reBotArm_control_py/config/gripper.yaml")
    args = parser.parse_args()

    g = Gripper(cfg_path=args.cfg)
    grip = RebotGripper(g)

    print("=== 力反馈柔性夹取控制器 ===")
    if not grip.enable():
        print("[错误] 电机使能失败，请检查连接")
        return

    pos, vel, torq = grip.get_state()
    print(f"  初始状态: pos={pos:+.4f}  vel={vel:+.4f}  torq={torq:+.4f}")
    print("\n  按键: g=夹取  o=张开  +=加力  -=减力  s=状态  z=设零  q=退出\n")

    HELP = "按键: g=夹取  o=张开  +=加力  -=减力  s=状态  z=设零  q=退出"

    try:
        while True:
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                raw = "q"

            cmd = raw.lower()
            if not cmd:
                continue

            if cmd == "q":
                break
            elif cmd == "g":
                print("[夹爪] → 柔性夹取...")
                ok = grip.grasp()
                if ok:
                    print("[夹爪] ✓ 夹取成功，力控保持中")
                else:
                    print("[夹爪] 空夹取或超时")
            elif cmd == "o":
                print("[夹爪] → 张开...")
                grip.open()
                print("[夹爪] ✓ 已张开")
            elif cmd in ("+", "="):
                grip.set_force(grip._target_force + FORCE_STEP)
                print(f"[力矩] 目标力矩 → {grip._target_force:.2f} Nm")
            elif cmd == "-":
                grip.set_force(grip._target_force - FORCE_STEP)
                print(f"[力矩] 目标力矩 → {grip._target_force:.2f} Nm")
            elif cmd == "s":
                pos, vel, torq = grip.get_state()
                print(f"\n── 当前状态 ──")
                print(f"  位置  : {pos:+.4f} rad  ({grip.get_opening_m()*1000:.1f} mm)")
                print(f"  速度  : {vel:+.4f} rad/s")
                print(f"  力矩  : {torq:+.4f} Nm")
                print(f"  目标  : {grip._target_force:.2f} Nm")
                print(f"  保持中: {grip.is_holding}")
            elif cmd == "z":
                print("[设零] 确认？(y/n)")
                try:
                    ans = input("  > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    ans = "n"
                if ans == "y":
                    if grip.set_zero():
                        print("[设零] 完成")
                    else:
                        print("[设零] 失败")
                else:
                    print("[设零] 已取消")
            else:
                print(HELP)
    finally:
        print("\n[退出] 释放夹爪...")
        grip.release()
        grip.disconnect()
        print("已退出。")


if __name__ == "__main__":
    main()
