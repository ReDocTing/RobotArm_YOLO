# 🦾 reBot Arm B601-DM 视觉夹取 Demo

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
    <img src="https://img.shields.io/badge/Detection-YOLO-yellow.svg" alt="YOLO">
</p>

<p align="center">
  <strong>深度感知 · 目标检测 · 手眼标定 · 自主抓取 · 全开源</strong>
</p>

<p align="center">
  <strong>
    <a href="./README_zh.md">简体中文</a> &nbsp;|&nbsp;
    <a href="./README.md">English</a>
  </strong>
</p>

---

## 📖 项目介绍

**reBot Arm B601-DM 视觉夹取 Demo** 是基于 [reBot Arm B601](https://github.com/vectorBH6/reBotArm_control_py) 机械臂控制库与奥比中光 **Gemini 2** 深度相机的视觉抓取算法演示项目。系统通过 YOLO 模型实时识别桌面物体，利用 OBB 最小外接矩形估计夹取姿态，经手眼标定将相机坐标系下的抓取点变换到机械臂基坐标系，最终驱动机械臂完成自主抓取。

### ✨ 核心功能

- 📷 **深度感知** — Orbbec Gemini 2 提供对齐的 RGB + 深度帧（1280×720 @ 30fps）
- 🔍 **目标检测** — 基于 YOLO 模型识别，支持开放词汇自定义类别
- 📐 **姿态估计** — OBB 最小外接矩形短轴方向估计夹爪朝向，深度分位数估计抓取高度
- 🔄 **坐标变换** — TSAI 手眼标定（Eye-in-Hand），将相机系抓取点变换到机械臂基坐标系
- 🦾 **运动执行** — reBotArm_control_py IK + 轨迹控制器，内置夹爪力控状态机

---

## ⚙️ 硬件配置

| 组件 | 型号 / 要求 |
|------|------------|
| 机械臂 | reBot Arm B601-DM（DAMIAO 电机版） |
| 深度相机 | Orbbec Gemini 2 |
| 通信接口 | USB2CAN 串口桥接器（机械臂）；USB 3.0（相机） |
| 主机 | Ubuntu 22.04+，Python 3.10，x86_64 |

**接线说明**

1. 将 Gemini 2 通过 USB 3.0 连接到主机
2. 将 USB2CAN 适配器连接到机械臂 CAN 总线并插入主机 USB 口
3. 配置设备权限：

```bash
sudo chmod a+rw /dev/bus/usb/*/*   # Orbbec 相机
sudo chmod 666 /dev/ttyUSB0        # USB2CAN（端口号按实际调整）
```

---

## 🚀 快速上手

### Step 1. 克隆仓库

```bash
git clone https://github.com/Seeed-Projects/reBot-DevArm-Grasp.git rebot_grasp
cd rebot_grasp
```

### Step 2. 创建 conda 环境

```bash
conda create -n rebotarm python=3.10 -y
conda activate rebotarm
```

### Step 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含相机感知层及机械臂控制库所需的全部依赖：

```
# 感知 / 检测
numpy<2.0.0
scipy>=1.10
opencv-python<4.10.0
opencv-contrib-python<4.10.0
ultralytics
PyYAML>=6.0
pyrealsense2>=2.54

# 机械臂（reBotArm_control_py）
pin>=3.9.0
meshcat>=0.3.2
matplotlib>=3.10.0
motorbridge>=0.1.7
```

### Step 4. 安装机械臂控制库

```bash
git clone https://github.com/vectorBH6/reBotArm_control_py.git sdk/reBotArm_control_py
cd sdk/reBotArm_control_py
pip install -e .
cd ../..
```

### Step 5. 安装 Orbbec SDK（pyorbbecsdk）

本项目依赖 **pyorbbecsdk**（Orbbec SDK v2 的 Python 封装），但仓库默认不包含 `sdk/pyorbbecsdk`。请先进入 `sdk/` 目录自行拉取官方仓库。

**方式一：从 GitHub 获取（推荐）**

```bash
# 安装编译依赖
sudo apt-get install -y cmake build-essential libusb-1.0-0-dev

cd sdk
git clone https://github.com/orbbec/pyorbbecsdk.git
cd pyorbbecsdk
pip install -e .
```

**方式二：从 Gitee 获取**

```bash
cd sdk
git clone https://gitee.com/orbbecdeveloper/pyorbbecsdk.git
cd pyorbbecsdk
pip install -e .
```

**验证安装**

```bash
python -c "import pyorbbecsdk; print('pyorbbecsdk OK')"
```

**配置 udev 规则（首次使用必须）**

```bash
cd sdk/pyorbbecsdk
sudo bash scripts/install_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**OrbbecViewer（可选，用于验证相机）**

下载预编译包后运行 `OrbbecViewer`，可在运行 Demo 前确认相机连接和深度流正常。

- GitHub：https://github.com/orbbec/OrbbecSDK_v2/releases
- Gitee：https://gitee.com/orbbecdeveloper/OrbbecSDK_v2/releases

**SDK 资料汇总**

| 资料 | 链接 |
|------|------|
| Gemini 2 产品页 | https://www.orbbec.com.cn/index/Product/info.html?cate=38&id=51 |
| 开发资料总链接 | https://www.orbbec.com.cn/index/Download2025/info.html?cate=121&id=1 |
| Orbbec SDK v2 | https://github.com/orbbec/OrbbecSDK_v2 |
| SDK v2 API 文档 | https://orbbec.github.io/docs/OrbbecSDKv2_API_User_Guide/ |
| pyorbbecsdk | https://github.com/orbbec/pyorbbecsdk |
| pyorbbecsdk 文档 | https://orbbec.github.io/pyorbbecsdk/index.html |
| ROS2 Wrapper | https://github.com/orbbec/OrbbecSDK_ROS2/tree/v2-main |

---

## 📁 目录结构

```
rebot_grasp/
├── config/
│   ├── default.yaml              # 主配置文件
│   └── calibration/
│       └── orbbec_gemini2/
│           ├── intrinsics.npz    # 相机内参
│           └── hand_eye.npz      # 手眼标定结果
├── drivers/
│   ├── camera/
│   │   ├── base.py               # 相机抽象基类
│   │   ├── orbbec_gemini2.py     # Gemini 2 驱动
│   │   └── realsense.py          # RealSense 驱动（备用）
│   └── robot/
│       └── rebot_arm.py          # reBotArm 封装 + 夹爪状态机
├── calibration/
│   ├── aruco_pose.py             # ArUco 位姿估计
│   └── hand_eye.py               # 手眼标定求解
├── utils/
│   ├── ordinary_grasp.py         # OBB 抓取姿态估计与可视化
│   └── transforms.py             # 坐标变换工具
├── scripts/
│   ├── main.py                   # 主抓取程序
│   ├── ordinary_grasp_pipeline.py
│   ├── object_detection.py
│   └── collect_handeye_eih.py
├── sdk/
│   ├── pyorbbecsdk/              # Orbbec SDK Python 封装
│   └── reBotArm_control_py/      # reBot Arm SDK
└── requirements.txt
```

---

## 🛠️ 配置说明

### 配置文件

编辑 `config/default.yaml`，确认以下关键参数：

```yaml
camera:
  type: orbbec_gemini2
  color_width: 1280
  color_height: 720
  fps: 30

robot:
  repo_root: null   # 自动识别 sdk/reBotArm_control_py
  ready_pose:
    x: 0.3
    y: 0.0
    z: 0.3
    pitch: 1.0
    duration: 3.0

yolo:
  model_name: "yoloe-26l-seg.pt"
  device: "cpu"          # GPU 可改为 "cuda:0"
  custom_classes:
    - "yellow banana"
    - "water bottle"
    - "cup"
```

### 手眼标定（首次使用）

```bash
python scripts/collect_handeye_eih.py
```

自动模式下，机械臂会自动遍历 50 个预设位姿，检测到 ArUco 稳定后自动采样。正常结束或中途打断时，脚本都会尝试计算并保存标定结果；至少需要 5 个样本，建议 ≥15 个样本以获得更稳的结果。

如需手动推动机械臂采集，可使用：

```bash
python scripts/collect_handeye_eih.py --manual
```

手动模式下，机械臂会进入重力补偿状态。将末端推到合适视角后按 `Enter` 采集，按 `c` 或 `q` 结束并计算。

---

## 🎬 Demo 内容介绍

### `scripts/main.py` — 主抓取程序

完整的视觉抓取流水线：

1. 机械臂使能，移动到预备高位
2. 实时相机预览 + YOLO 目标检测与实例分割
3. OBB 短轴估计夹爪朝向，深度分位数估计抓取高度
4. 按 `G` 冻结帧，经手眼变换计算机械臂目标位姿
5. 机械臂移动到预抓取点 → 下降 → 夹爪闭合 → 提升 → 回预备位

### `scripts/ordinary_grasp_pipeline.py` — 简化抓取测试

不依赖机械臂，仅验证 OBB 抓取姿态估计和可视化效果，适合调试感知模块。

### `scripts/object_detection.py` — 基础检测 Demo

纯 YOLO 检测演示，实时显示检测框和置信度，无抓取逻辑。

### `scripts/collect_handeye_eih.py` — 手眼标定数据采集

Eye-in-Hand 模式手眼标定，支持自动遍历采样和手动重力补偿采样，使用 ArUco 标记，支持 TSAI / PARK / HORAUD 三种求解方法。

---

## 📄 参考资料

- [reBotArm_control_py](https://github.com/vectorBH6/reBotArm_control_py) — 机械臂控制库
- [reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm) — reBot 机械臂开源项目
- [Orbbec Gemini 2 产品页](https://www.orbbec.com.cn/index/Product/info.html?cate=38&id=51)
- [Orbbec SDK v2](https://github.com/orbbec/OrbbecSDK_v2)
- [pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk)
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)

---

## ☎ 联系我们

- **技术支持**：[提交 Issue](https://github.com/Seeed-Projects/reBot-DevArm-Grasp/issues)

---

<p align="center">
  <strong>🌟 如果本项目对你有帮助，欢迎点个 Star！</strong>
</p>
