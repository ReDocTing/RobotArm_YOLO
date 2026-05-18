# RobotArm_YOLO — reBot B601-DM Visual Grasping (Personal Fork)

Personal fork based on [Seeed reBot-DevArm-Grasp](https://github.com/Seeed-Projects/reBot-DevArm-Grasp), adapted for **dual Intel RealSense D435** cameras and local calibration.

**Repository:** [ReDocTing/RobotArm_YOLO](https://github.com/ReDocTing/RobotArm_YOLO)

**Full documentation (Chinese, dev log, env, commands):** [README_zh.md](./README_zh.md)

---

## Quick facts

| Item | Value |
|------|--------|
| Conda env | `rebotarm` (Python 3.10) |
| Arm camera (Eye-in-Hand) | RealSense D435 `819612071433` |
| Wall camera (reserved) | RealSense D435 `819312070131` |
| Hand-eye result | `config/calibration/realsense_d435/hand_eye.npz` |
| Detector | `yoloe-26l-seg.pt` on CPU |

---

## Daily workflow

```bash
conda activate rebotarm
cd ~/rebot_grasp

python scripts/object_detection.py              # vision only
python scripts/ordinary_grasp_pipeline.py       # grasp estimate only
python scripts/collect_handeye_eih.py           # hand-eye (arm powered)
python scripts/main.py --dry-run                # pose check
python scripts/main.py                          # real grasp
```

See [README_zh.md](./README_zh.md) for the **2026-05-18** development log and detailed setup.
