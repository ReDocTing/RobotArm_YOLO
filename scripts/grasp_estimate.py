"""
EconomicGrasp 夹取姿态估计 (Perception-only demo)
=================================================

基于 EconomicGrasp (ECCV 2024, iSEE-Laboratory) 的 6-DoF 夹取检测：
    实时预览 (彩色帧 + FPS)
        ↓  按下 [S]
    全场景深度帧反投影 → 相机系点云
        ↓
    随机采样 num_point + 体素化 → 网络输入
        ↓
    net.forward + pred_decode → GraspGroup
        ↓
    NMS + 碰撞过滤 → Top-K
        ↓
    Open3D 可视化 (场景点云 + 夹爪线框 + 坐标系)

用法:
    cd /home/chlorine/seeed/cameraws
    python scripts/grasp_estimate.py

前置条件 (用户自行准备):
    1. 依赖：torch + CUDA 12 + MinkowskiEngine + graspnetAPI + open3d
    2. 仓库：git clone https://github.com/iSEE-Laboratory/EconomicGrasp.git
              到 cameraws/sdk/EconomicGrasp/
    3. 权重：从 EconomicGrasp releases 下载 RealSense checkpoint 到
              cameraws/models/EconomicGrasp/realsense_checkpoint.tar
"""

import os
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

import sys
import time
import argparse
from pathlib import Path

# ── 路径注入：cameraws 根 + EconomicGrasp 仓库根 ─────────────────────
project_root = Path(__file__).resolve().parent.parent      # cameraws/
seeed_root   = project_root.parent                          # seeed/
econgrasp_root = project_root / "sdk" / "EconomicGrasp"

for p in (seeed_root, project_root, econgrasp_root):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cv2
import yaml
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("[ERROR] Missing open3d:  pip install open3d")
    sys.exit(1)

# torch / EconomicGrasp / graspnetAPI 延迟导入（到 Estimator 构造时）


# ──────────────────────────────────────────────────────────────
# 配置加载
# ──────────────────────────────────────────────────────────────
def load_config(yaml_path: Path) -> dict:
    if not yaml_path.exists():
        print(f"[ERROR] Config not found: {yaml_path}")
        sys.exit(1)
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────
# 全场景点云生成（相机坐标系）
# ──────────────────────────────────────────────────────────────
def build_scene_pointcloud(
    depth_mm: np.ndarray,
    color_bgr: np.ndarray,
    K: np.ndarray,
    depth_min_m: float = 0.1,
    depth_max_m: float = 1.5,
):
    """整帧深度反投影为相机坐标系点云。

    Args:
        depth_mm: (H, W) uint16 深度图 (毫米)
        color_bgr: (H, W, 3) uint8 BGR 图
        K: (3, 3) 相机内参
        depth_min_m, depth_max_m: 工作区深度范围

    Returns:
        points:  (N, 3) float32，相机系 XYZ，单位米
        colors:  (N, 3) float32，RGB [0, 1]
    """
    h, w = depth_mm.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    v, u = np.indices((h, w), dtype=np.float32)
    z_m = depth_mm.astype(np.float32) / 1000.0
    valid = (z_m > depth_min_m) & (z_m < depth_max_m) & np.isfinite(z_m)

    x_m = (u - cx) * z_m / fx
    y_m = (v - cy) * z_m / fy

    pts = np.stack([x_m, y_m, z_m], axis=-1)[valid]          # (N, 3)
    cols_rgb = color_bgr[..., ::-1].astype(np.float32) / 255.0
    cols = cols_rgb[valid]                                    # (N, 3)
    return pts.astype(np.float32), cols.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# EconomicGrasp 估计器封装
# ──────────────────────────────────────────────────────────────
class EconomicGraspEstimator:
    """
    EconomicGrasp 推理封装。构造时完成仓库导入、权重加载；
    predict(points, colors) 返回 graspnetAPI.GraspGroup。

    Args:
        checkpoint_path: 预训练权重绝对路径 (.tar)
        num_point:       网络输入采样点数
        voxel_size:      MinkowskiEngine 体素尺寸 (m)
        collision_thresh: 碰撞检测阈值 (m)；<=0 表示关闭
        device:          "cuda:0" / "cpu"
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_point: int = 15000,
        voxel_size: float = 0.005,
        collision_thresh: float = 0.01,
        device: str = "cuda:0",
    ):
        import torch
        self._torch = torch

        # 延迟导入：依赖 sdk/EconomicGrasp 在 sys.path
        try:
            from models.economicgrasp import economicgrasp, pred_decode
        except ImportError as e:
            raise ImportError(
                "无法导入 EconomicGrasp 模型。请确认仓库已克隆至 "
                f"{econgrasp_root}\n原始错误: {e}"
            ) from e
        try:
            from graspnetAPI import GraspGroup
        except ImportError as e:
            raise ImportError(
                "无法导入 graspnetAPI。请 pip install graspnetAPI\n"
                f"原始错误: {e}"
            ) from e

        self._pred_decode = pred_decode
        self._GraspGroup  = GraspGroup

        # 碰撞检测器 (可选)
        self._collision_detector_cls = None
        if collision_thresh > 0:
            try:
                from utils.collision_detector import ModelFreeCollisionDetector
                self._collision_detector_cls = ModelFreeCollisionDetector
            except ImportError:
                print("[WARN] utils.collision_detector 不可用，跳过碰撞过滤")

        self.num_point        = int(num_point)
        self.voxel_size       = float(voxel_size)
        self.collision_thresh = float(collision_thresh)
        self.device = torch.device(
            device if (device.startswith("cuda") and torch.cuda.is_available())
            else "cpu"
        )
        if str(self.device) == "cpu" and device.startswith("cuda"):
            print(f"[WARN] CUDA 不可用，回落到 CPU（推理会非常慢）")

        # 模型构造 + 权重
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_absolute():
            ckpt_path = project_root / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

        print(f"[EconomicGrasp] 构建网络…")
        self._net = economicgrasp(seed_feat_dim=512, is_training=False)
        self._net.to(self.device)

        print(f"[EconomicGrasp] 加载权重: {ckpt_path.name}")
        ckpt = torch.load(str(ckpt_path), map_location=self.device)
        self._net.load_state_dict(ckpt["model_state_dict"])
        self._net.eval()
        print(f"[EconomicGrasp] 就绪 (device={self.device})")

    # ----------------------------------------------------------

    def _sample_points(self, points: np.ndarray, colors: np.ndarray):
        """将点云随机采样到 self.num_point。不足则重复抽样。"""
        n = len(points)
        if n == 0:
            return points, colors
        if n >= self.num_point:
            idx = np.random.choice(n, self.num_point, replace=False)
        else:
            idx1 = np.arange(n)
            idx2 = np.random.choice(n, self.num_point - n, replace=True)
            idx = np.concatenate([idx1, idx2])
        return points[idx], colors[idx]

    def _build_end_points(self, points: np.ndarray, colors: np.ndarray):
        """构造 EconomicGrasp 网络输入字典。

        参照 dataset/graspnet_dataset.py `get_data()` 的 ret_dict:
          point_clouds:          (B, N, 3)  float32
          cloud_colors:          (B, N, 3)  float32
          coordinates_for_voxel: (B, N, 3)  float32  (已按 voxel_size 缩放)
        """
        torch = self._torch
        pts_t  = torch.from_numpy(points).float().unsqueeze(0).to(self.device)
        cols_t = torch.from_numpy(colors).float().unsqueeze(0).to(self.device)
        coord_t = torch.from_numpy(points / self.voxel_size) \
                       .float().unsqueeze(0).to(self.device)
        return {
            "point_clouds":          pts_t,
            "cloud_colors":          cols_t,
            "coordinates_for_voxel": coord_t,
        }

    # ----------------------------------------------------------

    def predict(self, points: np.ndarray, colors: np.ndarray):
        """推理单帧场景点云，返回排序后的 GraspGroup。

        Args:
            points: (N, 3) float32 相机系
            colors: (N, 3) float32 RGB [0, 1]

        Returns:
            GraspGroup (按 score 降序排序；若碰撞检测开启，已过滤)
        """
        torch = self._torch
        if len(points) < 100:
            return self._GraspGroup(np.zeros((0, 17), dtype=np.float32))

        # 1. 采样
        pts_s, cols_s = self._sample_points(points, colors)

        # 2. 构造网络输入
        end_points = self._build_end_points(pts_s, cols_s)

        # 3. 推理
        t0 = time.perf_counter()
        with torch.no_grad():
            end_points = self._net(end_points)
            grasp_preds = self._pred_decode(end_points)
        t_inf = time.perf_counter() - t0

        arr = grasp_preds[0].detach().cpu().numpy()
        gg = self._GraspGroup(arr)
        print(f"[EconomicGrasp] 推理 {t_inf*1000:.1f} ms → {len(gg)} 个原始候选")

        # 4. 碰撞过滤
        if self.collision_thresh > 0 and self._collision_detector_cls \
                and len(gg) > 0:
            detector = self._collision_detector_cls(
                pts_s, voxel_size=self.voxel_size
            )
            collision_mask = detector.detect(
                gg, approach_dist=0.05,
                collision_thresh=self.collision_thresh,
            )
            gg = gg[~collision_mask]
            print(f"[EconomicGrasp] 碰撞过滤后 → {len(gg)} 个候选")

        # 5. NMS + 排序
        if len(gg) > 0:
            gg.nms()
            gg.sort_by_score()
        return gg


# ──────────────────────────────────────────────────────────────
# Open3D 可视化
# ──────────────────────────────────────────────────────────────
def create_virtual_gripper(T: np.ndarray, width: float = 0.08, depth: float = 0.06,
                           color=(1.0, 0.0, 0.0)):
    """4×4 变换 → 红色线框夹爪 (fallback, 当 graspnetAPI 可视化不可用时)。"""
    w2 = width / 2.0
    pts_local = np.array([
        [0,   0,   -0.05],
        [-w2, 0,    0   ],
        [ w2, 0,    0   ],
        [-w2, 0,    depth],
        [ w2, 0,    depth],
    ])
    pts_world = (T @ np.hstack([pts_local, np.ones((5, 1))]).T).T[:, :3]
    lines = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_world)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([list(color)] * len(lines))
    return ls


def rotation_to_euler_deg(R: np.ndarray) -> np.ndarray:
    """ZYX 欧拉角 (roll, pitch, yaw)，单位度。"""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.degrees([rx, ry, rz])


def visualize_grasps(points, colors, gg, top_k: int = 10,
                     window_name: str = "EconomicGrasp"):
    """场景点云 + Top-K 夹爪可视化。

    使用相机坐标系：+X 右、+Y 下、+Z 前 (深度增大方向)。
    为了便于观察，全局施加 flip 矩阵把 Z 朝上 (Open3D 惯例)。
    """
    flip = np.array([[1, 0,  0, 0],
                     [0, -1, 0, 0],
                     [0, 0, -1, 0],
                     [0, 0,  0, 1]], dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(voxel_size=0.003)
    pcd.transform(flip)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    geometries = [pcd, coord]

    if len(gg) == 0:
        print("[Viz] 无有效夹取候选")
        o3d.visualization.draw_geometries(geometries, window_name=window_name)
        return

    top = gg[:top_k] if len(gg) > top_k else gg

    # 优先走 graspnetAPI 内置网格（比线框更直观）
    try:
        meshes = top.to_open3d_geometry_list()
        for m in meshes:
            m.transform(flip)
        geometries.extend(meshes)
    except Exception as e:
        print(f"[Viz] graspnetAPI 可视化失败 ({e})，回落到线框")
        for g in top:
            T = np.eye(4)
            T[:3, :3] = g.rotation_matrix
            T[:3,  3] = g.translation
            geometries.append(create_virtual_gripper(flip @ T, width=float(g.width)))

    o3d.visualization.draw_geometries(geometries, window_name=window_name)


# ──────────────────────────────────────────────────────────────
# S 键快照回调
# ──────────────────────────────────────────────────────────────
def run_snapshot(
    depth_mm: np.ndarray,
    color_bgr: np.ndarray,
    K: np.ndarray,
    estimator: EconomicGraspEstimator,
    ws_cfg: dict,
    top_k: int,
):
    print("\n[Snapshot] 构建点云 …")
    pts, cols = build_scene_pointcloud(
        depth_mm, color_bgr, K,
        depth_min_m=float(ws_cfg.get("min_depth_m", 0.1)),
        depth_max_m=float(ws_cfg.get("max_depth_m", 1.5)),
    )
    print(f"[Snapshot] 工作区有效点数: {len(pts)}")
    if len(pts) < 500:
        print("[Snapshot] 有效点过少，跳过推理")
        return

    print("[Snapshot] EconomicGrasp 推理 …")
    gg = estimator.predict(pts, cols)

    if len(gg) == 0:
        print("[Snapshot] 无夹取候选")
    else:
        show = gg[:top_k] if len(gg) > top_k else gg
        print(f"\n── Top-{len(show)} 夹取候选 (相机系) ──")
        for i, g in enumerate(show):
            T = np.eye(4); T[:3, :3] = g.rotation_matrix; T[:3, 3] = g.translation
            rpy = rotation_to_euler_deg(g.rotation_matrix)
            print(f"  #{i+1:02d}  score={float(g.score):.3f}  "
                  f"pos=({g.translation[0]:+.3f},{g.translation[1]:+.3f},"
                  f"{g.translation[2]:+.3f}) m  "
                  f"RPY=({rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f})°  "
                  f"width={float(g.width)*100:.1f}cm  "
                  f"depth={float(g.depth)*100:.1f}cm")

    print("\n[Open3D] 关闭窗口返回实时预览。")
    visualize_grasps(pts, cols, gg, top_k=top_k,
                     window_name="EconomicGrasp — Scene + Grasps")


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="EconomicGrasp perception demo")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--top-k", type=int, default=None,
                        help="覆盖 YAML 的 top_k")
    args = parser.parse_args()

    cfg = load_config(project_root / args.config)

    ge_cfg  = cfg.get("grasp_estimator", {})
    ws_cfg  = ge_cfg.get("workspace", {})
    top_k   = args.top_k if args.top_k is not None else int(ge_cfg.get("top_k", 10))

    # ── 加载估计器（相机之前加载，早暴露错误）──
    print("=== Loading EconomicGrasp ===")
    estimator = EconomicGraspEstimator(
        checkpoint_path=ge_cfg.get("checkpoint_path",
                                   "models/EconomicGrasp/realsense_checkpoint.tar"),
        num_point=ge_cfg.get("num_point", 15000),
        voxel_size=ge_cfg.get("voxel_size", 0.005),
        collision_thresh=ge_cfg.get("collision_thresh", 0.01),
        device=ge_cfg.get("device", "cuda:0"),
    )

    # ── 相机 ──
    from cameraws.drivers.camera import make_camera
    print(f"\n=== Camera: {cfg.get('camera', {}).get('type')} ===")
    cam = make_camera(cfg)
    cam.open()
    try:
        cam.warm_up(15)
    except AttributeError:
        pass
    K = cam.K
    print(f"  K: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    # ── UI ──
    WIN = "EconomicGrasp Live"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    print(f"\n[Keys]  S=snapshot+grasp   Q/ESC=quit\n")

    _fps_t, _fps_cnt = time.perf_counter(), 0
    fps_disp = 0.0

    try:
        while True:
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                continue

            display = color_bgr.copy()
            _fps_cnt += 1
            now = time.perf_counter()
            if now - _fps_t >= 1.0:
                fps_disp = _fps_cnt / (now - _fps_t)
                _fps_cnt, _fps_t = 0, now

            cv2.putText(display,
                        f"{fps_disp:.1f}fps | S=Grasp  Q=Quit",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WIN, display)

            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in [ord('q'), ord('Q'), 27]:
                break
            if key in [ord('s'), ord('S')]:
                run_snapshot(depth_mm, color_bgr, K, estimator, ws_cfg, top_k)

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Exiting.")
    finally:
        try: cam.close()
        except Exception: pass
        cv2.destroyAllWindows()

    print("Done.")


if __name__ == "__main__":
    main()
    os._exit(0)
