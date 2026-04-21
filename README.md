# 🦾 cameraws — Robotic Arm Vision Grasping Demo

<p align="center">
  <img src="https://raw.githubusercontent.com/Seeed-Projects/reBot-DevArm/main/media/v1.0.png" alt="reBot Arm B601">
</p>

<p align="center">
    <a href="./LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
    </a>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/Platform-Ubuntu%2022.04+-orange.svg" alt="Platform">
    <img src="https://img.shields.io/badge/Camera-Orbbec%20Gemini%202-green.svg" alt="Camera">
    <img src="https://img.shields.io/badge/Detection-YOLOv11-yellow.svg" alt="YOLO">
</p>

<p align="center">
  <strong>Depth Perception · Object Detection · Hand-Eye Calibration · Autonomous Grasping · Fully Open Source</strong>
</p>

<p align="center">
  <strong>
    <a href="./README_zh.md">简体中文</a> &nbsp;|&nbsp;
    <a href="./README.md">English</a>
  </strong>
</p>

---

## 📖 Introduction

**cameraws** is a vision-based grasping demo that integrates the [reBot Arm B601](https://github.com/vectorBH6/reBotArm_control_py) robotic arm control library with the **Orbbec Gemini 2** depth camera. The system uses a YOLO model to detect tabletop objects in real time, estimates grasp poses via OBB minimum bounding rectangles, transforms grasp points from camera space to robot base space through hand-eye calibration, and drives the arm to perform autonomous grasping.

### ✨ Core Features

- 📷 **Depth Perception** — Orbbec Gemini 2 provides aligned RGB + depth frames (1280×720 @ 30fps)
- 🔍 **Object Detection** — YOLO model-based recognition with open-vocabulary custom classes
- 📐 **Pose Estimation** — OBB short-axis direction for gripper orientation; depth quantile for grasp height
- 🔄 **Coordinate Transform** — TSAI hand-eye calibration (Eye-in-Hand) to map camera-frame grasp points to robot base frame
- 🦾 **Motion Execution** — reBotArm_control_py IK + trajectory controller with built-in gripper force-control state machine

---

## ⚙️ Hardware Setup

| Component | Model / Requirement |
|-----------|-------------------|
| Robotic Arm | reBot Arm B601-DM (DAMIAO motor variant) |
| Depth Camera | Orbbec Gemini 2 |
| Communication | USB2CAN serial bridge (arm); USB 3.0 (camera) |
| Host PC | Ubuntu 22.04+, Python 3.10, x86_64 |

**Wiring**

1. Connect the Gemini 2 to the host via USB 3.0
2. Connect the USB2CAN adapter to the arm's CAN bus and plug it into the host
3. Set device permissions:

```bash
sudo chmod a+rw /dev/bus/usb/*/*   # Orbbec camera
sudo chmod 666 /dev/ttyUSB0        # USB2CAN (adjust port as needed)
```

---

## 🚀 Quick Start

### Step 1. Clone the repository

```bash
git clone https://github.com/your-org/cameraws.git
cd cameraws
git submodule update --init --recursive
```

### Step 2. Create a conda environment

```bash
conda create -n cameraws python=3.10 -y
conda activate cameraws
```

### Step 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` covers all dependencies for both the perception layer and the robotic arm control library:

```
# perception / detection
numpy<2.0.0
scipy>=1.10
opencv-python<4.10.0
opencv-contrib-python<4.10.0
ultralytics
PyYAML>=6.0
pyrealsense2>=2.54

# robotic arm (reBotArm_control_py)
pin>=3.9.0
meshcat>=0.3.2
matplotlib>=3.10.0
motorbridge>=0.1.7
```

### Step 4. Install the robotic arm control library

```bash
git clone https://github.com/vectorBH6/reBotArm_control_py.git ~/seeed/reBotArm_control_py
cd ~/seeed/reBotArm_control_py
pip install -e .
```

### Step 5. Install the Orbbec SDK (pyorbbecsdk)

This project uses **pyorbbecsdk** — the Python wrapper for Orbbec SDK v2. It is included as a git submodule under `sdk/pyorbbecsdk/`.

**Option 1: Build from submodule (recommended)**

```bash
# Install build dependencies
sudo apt-get install -y cmake build-essential libusb-1.0-0-dev

cd sdk/pyorbbecsdk
pip install -e .
```

**Option 2: Install from GitHub**

```bash
# GitHub
git clone https://github.com/orbbec/pyorbbecsdk.git
# Gitee (China mirror)
git clone https://gitee.com/orbbecdeveloper/pyorbbecsdk.git

cd pyorbbecsdk
pip install -e .
```

**Verify installation**

```bash
python -c "import pyorbbecsdk; print('pyorbbecsdk OK')"
```

**Configure udev rules (required on first use)**

```bash
sudo bash sdk/pyorbbecsdk/scripts/install_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**OrbbecViewer (optional — verify camera)**

Download the prebuilt package and run `OrbbecViewer` to confirm the camera connection and depth stream are working before running the demo.

- GitHub: https://github.com/orbbec/OrbbecSDK_v2/releases
- Gitee: https://gitee.com/orbbecdeveloper/OrbbecSDK_v2/releases

**SDK Resources**

| Resource | Link |
|----------|------|
| Gemini 2 product page | https://www.orbbec.com/products/stereo-vision-camera/gemini-2/ |
| All developer resources | https://www.orbbec.com.cn/index/Download2025/info.html?cate=121&id=1 |
| Orbbec SDK v2 | https://github.com/orbbec/OrbbecSDK_v2 |
| SDK v2 API guide | https://orbbec.github.io/docs/OrbbecSDKv2_API_User_Guide/ |
| pyorbbecsdk | https://github.com/orbbec/pyorbbecsdk |
| pyorbbecsdk docs | https://orbbec.github.io/pyorbbecsdk/index.html |
| ROS2 Wrapper | https://github.com/orbbec/OrbbecSDK_ROS2/tree/v2-main |

---

## 📁 Directory Structure

```
cameraws/
├── config/
│   ├── default.yaml              # Main configuration
│   └── calibration/
│       └── orbbec_gemini2/
│           ├── intrinsics.npz    # Camera intrinsics
│           └── hand_eye.npz      # Hand-eye calibration result
├── drivers/
│   ├── camera/
│   │   ├── base.py               # Abstract camera base class
│   │   ├── orbbec_gemini2.py     # Gemini 2 driver
│   │   └── realsense.py          # RealSense driver (alternative)
│   └── robot/
│       └── rebot_arm.py          # reBotArm wrapper + gripper FSM
├── calibration/
│   ├── aruco_pose.py             # ArUco pose estimation
│   └── hand_eye.py               # Hand-eye calibration solver
├── utils/
│   ├── ordinary_grasp.py         # OBB grasp estimation and visualization
│   └── transforms.py             # Coordinate transform utilities
├── scripts/
│   ├── main.py                   # Main grasping program
│   ├── ordinary_grasp_pipeline.py
│   ├── object_detection.py
│   └── collect_handeye_eih.py
├── sdk/
│   └── pyorbbecsdk/              # Orbbec SDK Python wrapper (submodule)
└── requirements.txt
```

---

## 🛠️ Configuration

### Config file

Edit `config/default.yaml` and verify the key parameters:

```yaml
camera:
  type: orbbec_gemini2
  color_width: 1280
  color_height: 720
  fps: 30

robot:
  repo_root: ~/seeed/reBotArm_control_py   # path to arm library
  ready_pose:
    x: 0.3
    y: 0.0
    z: 0.3
    pitch: 1.0
    duration: 3.0

yolo:
  model_name: "yoloe-26l-seg.pt"
  device: "cpu"          # use "cuda:0" for GPU
  custom_classes:
    - "yellow banana"
    - "water bottle"
    - "cup"
```

### Hand-eye calibration (first-time setup)

```bash
python scripts/collect_handeye_eih.py
```

Follow the prompts to move the arm to multiple poses and capture ArUco marker images (at least 15 poses recommended). The calibration result is saved automatically to `config/calibration/orbbec_gemini2/hand_eye.npz`.

---

## 🎬 Demo Description

### `scripts/main.py` — Main grasping program

The full vision-grasping pipeline:

1. Enable the arm and move to the ready pose
2. Live camera preview with YOLO object detection and instance segmentation
3. OBB short-axis estimation for gripper orientation; depth quantile for grasp height
4. Press `G` to freeze the frame; hand-eye transform computes the target arm pose
5. Arm moves to pre-grasp point → descends → gripper closes → lifts → returns to ready pose

### `scripts/ordinary_grasp_pipeline.py` — Simplified grasp test

Runs OBB grasp pose estimation and visualization without connecting to the arm. Useful for debugging the perception module in isolation.

### `scripts/object_detection.py` — Basic detection demo

Pure YOLO detection with real-time bounding boxes and confidence scores. No grasping logic.

### `scripts/collect_handeye_eih.py` — Hand-eye calibration data collection

Eye-in-Hand hand-eye calibration using ArUco markers. Supports TSAI, PARK, and HORAUD solvers.

---

## 📄 References

- [reBotArm_control_py](https://github.com/vectorBH6/reBotArm_control_py) — Robotic arm control library
- [reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm) — reBot arm open-source project
- [Orbbec Gemini 2](https://www.orbbec.com/products/stereo-vision-camera/gemini-2/)
- [Orbbec SDK v2](https://github.com/orbbec/OrbbecSDK_v2)
- [pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk)
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)

---

## ☎ Contact Us

- **Technical Support**: [Submit an Issue](https://github.com/your-org/cameraws/issues)

---

<p align="center">
  <strong>🌟 If this project helps you, please give us a Star!</strong>
</p>
