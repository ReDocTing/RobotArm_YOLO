# reBot Arm B601-DM 视觉夹取（个人实践版）

基于 [Seeed reBot-DevArm-Grasp](https://github.com/Seeed-Projects/reBot-DevArm-Grasp) 与 [Wiki 教程](https://wiki.seeedstudio.com/cn/rebot_arm_b601_dm_grasping_demo/) 修改，记录本人硬件环境与调试过程。

**个人仓库：** [ReDocTing/RobotArm_YOLO](https://github.com/ReDocTing/RobotArm_YOLO)

---

## 当前环境信息

| 项目 | 配置 |
|------|------|
| 主机系统 | Ubuntu 22.04+（内核 6.8.x） |
| 架构 | x86_64 |
| Python 环境 | Miniforge3，`conda` 环境名 **`rebotarm`**，Python **3.10.20** |
| 工作目录 | `~/rebot_grasp` |
| 机械臂 | reBot Arm B601-DM + 夹爪，USB2CAN（如 `/dev/ttyUSB0`） |
| 深度相机 | **2× Intel RealSense D435**（非 Orbbec Gemini 2） |
| 臂载相机（Eye-in-Hand） | serial: `819612071433` → `config/default.yaml` → `camera.serial` |
| 墙上固定相机（预留） | serial: `819312070131` → `camera.wall_serial`（主流程未使用） |
| 检测模型 | `models/yoloe-26l-seg.pt`，Ultralytics **8.4.x**，CPU 推理 |
| 主要依赖 | `numpy 2.2.x`，`opencv-python 4.10`，`pyrealsense2`，`pin 3.9` |
| PyTorch | CPU 版（见 `requirements-torch-cpu.txt`） |
| 机械臂 SDK | `sdk/reBotArm_control_py`（需自行 clone，`sdk/` 不入库） |
| 手眼标定结果 | `config/calibration/realsense_d435/hand_eye.npz` |

### 设备权限（每次上电后如需要）

```bash
sudo chmod a+rw /dev/bus/usb/*/*
sudo chmod 666 /dev/ttyUSB0    # 按实际 CAN 串口修改
```

### 查看已连接的 RealSense

```bash
conda activate rebotarm
cd ~/rebot_grasp
python scripts/list_realsense_cameras.py --no-preview
```

---

## 开发日志

### 2026-05-18

**完成内容**

- [x] 创建 conda 环境 `rebotarm`，安装项目依赖（解决 `numpy<2` 与 `pin`/YOLO 冲突，升级 `ultralytics 8.4`）
- [x] 安装机械臂 SDK：`sdk/reBotArm_control_py`（修复 `pyproject.toml` 包发现后 `pip install -e .`）
- [x] 改用 **双 Intel D435**：臂载 + 墙上固定；配置 `camera.serial` / `wall_serial`
- [x] 新增 `scripts/list_realsense_cameras.py`，RealSense 按序列号打开
- [x] 修复 OpenCV 4.10 下 ArUco `estimatePoseSingleMarkers` 不可用问题
- [x] 下载 YOLO 权重 `yoloe-26l-seg.pt`，跑通 `object_detection.py`
- [x] **Eye-in-Hand 手眼标定**，生成 `config/calibration/realsense_d435/hand_eye.npz`
- [x] 调试抓取高度：深度分位数 + `depth_offset_mm` + 安全高度护栏（3 cm 测试方块）
- [x] `main.py` dry-run / 实机试抓（空夹取、高度微调）

**备注**

- 官方 Wiki 中的 Orbbec `pyorbbecsdk` **本机未安装**，全程使用 `pyrealsense2`
- 墙上相机当前仅配置 serial，不参与 `main.py` 抓取流程

---

## 环境与依赖安装（首次）

```bash
# 1. 克隆（或拉取个人仓库）
git clone https://github.com/ReDocTing/RobotArm_YOLO.git rebot_grasp
cd rebot_grasp

# 2. conda 环境
conda create -n rebotarm python=3.10 -y
conda activate rebotarm

# 3. PyTorch（CPU，先装可避免拉取超大 CUDA 包）
python -m pip install -r requirements-torch-cpu.txt

# 4. 其余依赖（务必用 conda 环境里的 python -m pip）
python -m pip install -r requirements.txt

# 5. 机械臂 SDK
mkdir -p sdk
git clone https://github.com/vectorBH6/reBotArm_control_py.git sdk/reBotArm_control_py
cd sdk/reBotArm_control_py
# 若 pip install -e . 报包发现错误，已为 pyproject 增加 packages.find
python -m pip install -e .
cd ../..

# 6. YOLO 权重（若 models/ 下没有）
mkdir -p models
wget -O models/yoloe-26l-seg.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg.pt
```

---

## 日常使用：启动与调试指令

以下命令均假设：

```bash
conda activate rebotarm
cd ~/rebot_grasp
```

### 1. 确认臂载相机

```bash
# 列出两台 D435
python scripts/list_realsense_cameras.py --no-preview

# 预览臂载相机（serial 与 default.yaml 一致）
python scripts/list_realsense_cameras.py --serial 819612071433
```

### 2. 仅验证 YOLO 检测（不连机械臂）

```bash
python scripts/object_detection.py
```

按键：`Q` 退出。

### 3. 仅验证抓取估计（不连机械臂）

```bash
python scripts/ordinary_grasp_pipeline.py
```

按键：鼠标左键测深度，`G` 打印姿态，`Q` 退出。

### 4. 手眼标定（机械臂上电 + CAN + ArUco 贴桌面固定）

标定前确认 `config/default.yaml` 中 `calibration.aruco.marker_length_m` 与打印尺寸一致（默认 10 cm）。

```bash
# 自动遍历姿态采样
python scripts/collect_handeye_eih.py

# 或手动推臂（重力补偿），Enter 采样
python scripts/collect_handeye_eih.py --manual
```

结果：`config/calibration/realsense_d435/hand_eye.npz`（建议 ≥15 组样本）。

### 5. 主程序抓取

```bash
# 先 dry-run：只算位姿，机械臂不动
python scripts/main.py --dry-run

# 确认 [Grasp] grasp 的基座 z 合理后，实机抓取
python scripts/main.py
```

运行时按键：

| 键 | 功能 |
|----|------|
| `G` | 采当前帧并执行抓取（dry-run 仅打印） |
| `R` | 恢复实时预览（按 G 后画面会 FROZEN） |
| `Q` / `Esc` | 退出 |

### 6. 深度 / 高度微调

编辑 `config/default.yaml` → `grasp_pipeline.grasp`：

| 参数 | 作用 |
|------|------|
| `depth_quantile` | 越大越“深”（俯视时一般更低） |
| `depth_offset_mm` | 深度补偿（俯视 D435 正值≈往下） |
| `approach_offset_m` | 手眼变换后再下探（米） |
| `safety.min_grasp_z_m` | 基座最低高度护栏，防撞桌 |

当前个人调参起点（3 cm 方块）：

```yaml
depth_quantile: 0.88
depth_offset_mm: 14
approach_offset_m: 0.008
min_grasp_z_m: 0.030
```

---

## 配置文件要点

`config/default.yaml` 中与个人硬件相关的项：

```yaml
camera:
  type: realsense_d435
  serial: "819612071433"      # 臂载
  wall_serial: "819312070131" # 墙上（未接入主流程）

calibration:
  aruco:
    marker_length_m: 0.1
```

---

## 仓库说明

| 路径 | 说明 |
|------|------|
| `sdk/` | 机械臂 / 相机 SDK，**.gitignore 忽略**，需本地 clone |
| `models/` | YOLO 权重，**不入库** |
| `config/calibration/realsense_d435/` | 手眼标定结果（已纳入本仓库） |

---

## 参考链接

- [Seeed Wiki：视觉夹取 Demo](https://wiki.seeedstudio.com/cn/rebot_arm_b601_dm_grasping_demo/)
- [上游仓库 Seeed-Projects/reBot-DevArm-Grasp](https://github.com/Seeed-Projects/reBot-DevArm-Grasp)
- [reBotArm_control_py](https://github.com/vectorBH6/reBotArm_control_py)
