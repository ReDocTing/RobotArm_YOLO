"""
快照式静态点云生成与可视化测试管线 (Snapshot Point Cloud Pipeline)
支持 D435i / Gemini 2，结合 YOLO 实例分割生成高质量局部点云。

操作说明：
    1. 移动相机对准物体
    2. 按下键盘 [S] 键，截取快照并生成 3D 点云
    3. 在弹出的 Open3D 窗口中鼠标拖拽查看 3D 模型
    4. 关闭 3D 窗口继续下一个快照，按 [Q] 退出程序。

用法：
    pip install open3d
    cd /home/chlorine/seeed/cameraws
    python scripts/snapshot.py
"""

import os
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

import sys
import cv2
import yaml
import numpy as np
from pathlib import Path
from ultralytics import YOLO

try:
    import open3d as o3d
except ImportError:
    print("[错误] 缺少 open3d 库，请在终端执行: pip install open3d")
    sys.exit(1)

# ==========================================
# 通用配置加载与沙盒逻辑
# ==========================================
def load_config(yaml_path):
    if not os.path.exists(yaml_path):
        print(f"[错误] 找不到配置文件: {yaml_path}")
        sys.exit(1)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ==========================================
# 核心：向量化点云生成器 (极速版)
# ==========================================
def generate_masked_point_cloud(color_img, depth_map_mm, mask, intrinsics):
    """
    利用分割 Mask 和 Numpy 向量化运算，瞬间从深度图中抠出彩色的局部 3D 点云。
    intrinsics: (fx, fy, cx, cy)
    """
    fx, fy, cx, cy = intrinsics
    
    # 确保 mask 和深度图尺寸一致
    h, w = depth_map_mm.shape
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 创建布尔过滤器：只有在掩码内、且深度有效(>0且<2米)的点才被保留
    valid_mask = (mask > 0) & (depth_map_mm > 0) & (depth_map_mm < 2000)
    
    # 提取有效像素的二维坐标 (v是y, u是x)
    v, u = np.where(valid_mask)
    
    # 提取有效深度并转换为米
    z_m = depth_map_mm[valid_mask] / 1000.0
    
    # 向量化小孔成像逆投影
    x_m = (u - cx) * z_m / fx
    y_m = (v - cy) * z_m / fy
    
    # 组装 3D 坐标矩阵 (N, 3)
    points_3d = np.stack((x_m, y_m, z_m), axis=-1)
    
    # 提取对应点的 RGB 颜色 (OpenCV 默认是 BGR，需要转 RGB)
    colors_bgr = color_img[valid_mask]
    colors_rgb = colors_bgr[:, ::-1] / 255.0  # 归一化到 [0, 1] 给 Open3D 用
    
    # 构建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
    
    # 降采样与统计学去噪 (消除边缘的飞点)
    pcd = pcd.voxel_down_sample(voxel_size=0.002) # 2mm 降采样
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    return pcd

# ==========================================
# 主流程
# ==========================================
def main():
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "default.yaml"
    cfg = load_config(config_path)
    
    cam_cfg = cfg.get("camera", {})
    cam_type = cam_cfg.get("type", "realsense_d435i").lower()
    color_w, color_h = cam_cfg.get("color_width", 640), cam_cfg.get("color_height", 480)
    fps = cam_cfg.get("fps", 30)

    yolo_cfg = cfg.get("yolo", {})
    model_name = yolo_cfg.get("model_name", "yoloe-26s-seg.pt")
    device = yolo_cfg.get("device", "cpu") 
    use_world = yolo_cfg.get("use_world", False)
    custom_classes = yolo_cfg.get("custom_classes", ["person", "cup", "cell phone"])
    
    # 动态解析模型文件夹，默认回退到你的 "models" 目录
    custom_model_dir = yolo_cfg.get("model_dir", "models")
    if os.path.isabs(custom_model_dir): models_dir = Path(custom_model_dir)
    else: models_dir = project_root / custom_model_dir
    models_dir.mkdir(parents=True, exist_ok=True) 

    model_path = models_dir / model_name

    # --- 初始化 YOLO (沙盒模式) ---
    print(f"=== 初始化 YOLO 模型 ===")
    original_cwd = os.getcwd()
    try:
        os.chdir(models_dir)
        print(f"尝试加载模型: {model_path}")
        # 强制传入绝对路径以确保稳定加载
        model = YOLO(str(model_path)) 
        if use_world and ("world" in model_name.lower() or "yoloe" in model_name.lower()):
            print(f"注入概念: {custom_classes}")
            model.set_classes(custom_classes)
    finally:
        os.chdir(original_cwd)
    print(f"YOLO 就绪 (Device: {device})")

    # --- 初始化相机 ---
    print(f"\n=== 初始化相机: {cam_type} ===")
    cam_intrinsics = (0, 0, 0, 0) # (fx, fy, cx, cy)
    
    if "realsense" in cam_type:
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, color_w, color_h, rs.format.z16, fps)
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        
        intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        cam_intrinsics = (intrin.fx, intrin.fy, intrin.ppx, intrin.ppy)
        print("[相机就绪] D435i")
        
    elif "orbbec" in cam_type:
        from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
        pipeline = Pipeline()
        config = Config()
        
        pl = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try: cp = pl.get_video_stream_profile(color_w, color_h, OBFormat.MJPG, fps)
        except: cp = pl.get_default_video_stream_profile()
        config.enable_stream(cp)
        
        dl = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try: dp = dl.get_video_stream_profile(color_w, color_h, OBFormat.Y16, fps)
        except: dp = dl.get_default_video_stream_profile()
        config.enable_stream(dp)

        config.set_align_mode(OBAlignMode.HW_MODE)
        pipeline.start(config)
        
        cprm = pipeline.get_camera_param()
        cam_intrinsics = (cprm.rgb_intrinsic.fx, cprm.rgb_intrinsic.fy, cprm.rgb_intrinsic.cx, cprm.rgb_intrinsic.cy)
        print("[相机就绪] Gemini 2")

    window_name = "Live View (Press 'S' to Snapshot)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # === 取景器模式：只读流，不跑 YOLO，保证极高帧率 ===
            color_image, depth_map_mm = None, None
            
            if "realsense" in cam_type:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                cf, df = aligned_frames.get_color_frame(), aligned_frames.get_depth_frame()
                if not cf or not df: continue
                color_image = np.asanyarray(cf.get_data())
                # D435i 原始 raw data 是 uint16 的毫米数据
                depth_map_mm = np.asanyarray(df.get_data())
                
            elif "orbbec" in cam_type:
                frames = pipeline.wait_for_frames(500)
                if frames is None: continue
                cf, df = frames.get_color_frame(), frames.get_depth_frame()
                if not cf or not df: continue
                
                w_ob, h_ob, fmt_ob = cf.get_width(), cf.get_height(), cf.get_format()
                raw_color = np.ascontiguousarray(np.asanyarray(cf.get_data()), dtype=np.uint8)
                if fmt_ob == OBFormat.MJPG: color_image = cv2.imdecode(raw_color, cv2.IMREAD_COLOR)
                elif fmt_ob == OBFormat.RGB: color_image = cv2.cvtColor(raw_color.reshape((h_ob, w_ob, 3)), cv2.COLOR_RGB2BGR)
                else: color_image = raw_color.reshape((h_ob, w_ob, 3))
                
                depth_map_mm = np.frombuffer(df.get_data(), dtype=np.uint16).reshape((h_ob, w_ob))

            if color_image is None or depth_map_mm is None: continue

            # 绘制取景器 UI
            display_img = color_image.copy()
            cv2.putText(display_img, f"Move Camera & Press [S] to Capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(window_name, display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'): # ESC or Q 退出
                break
            
            # === 核心逻辑：按下 'S' 键触发快照与点云生成 ===
            if key == ord('s') or key == ord('S'):
                print("\n[*] 快照已截取！正在运行实例分割与点云生成...")
                
                # 1. 跑 YOLO
                results = model.predict(color_image, verbose=False, device=device)
                
                # 用于收集所有生成的目标点云
                combined_pcd = o3d.geometry.PointCloud()
                found_targets = False
                
                for r in results:
                    # 如果模型支持分割 (-seg.pt)，提取 Mask
                    has_masks = r.masks is not None
                    
                    for i, box in enumerate(r.boxes):
                        cls_id, conf = int(box.cls[0]), float(box.conf[0])
                        class_name = model.names[cls_id]
                        
                        # 准备该目标的掩码
                        target_mask = np.zeros(depth_map_mm.shape, dtype=np.uint8)
                        
                        if has_masks:
                            # 提取分割模型的精准多边形掩码
                            mask_data = r.masks.data[i].cpu().numpy()
                            target_mask = mask_data
                        else:
                            # 如果用的是纯检测模型(如 yolo26n.pt)，没有 mask，就用粗糙的矩形框替代
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            cv2.rectangle(target_mask, (x1, y1), (x2, y2), 1, -1)
                            
                        # 2. 调用强大的向量化函数生成 3D 局部点云
                        pcd = generate_masked_point_cloud(color_image, depth_map_mm, target_mask, cam_intrinsics)
                        
                        if len(pcd.points) > 50: # 只有点云数量足够才有效
                            combined_pcd += pcd
                            found_targets = True
                            print(f" -> 成功生成 [{class_name}] 的局部点云，点数: {len(pcd.points)}")
                        
                        # 顺便在 2D 图上画出来给大家看看
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cv2.putText(display_img, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # 3. 结果展示
                cv2.imshow(window_name, display_img)
                cv2.waitKey(1) # 刷新一下 2D 窗口
                
                if found_targets:
                    print("[*] 正在打开 3D 点云视窗... (请用鼠标拖拽旋转，关闭 3D 窗口后可继续拍摄)")
                    
                    # 为了在 Open3D 里看起来是正的，我们对相机坐标系做一个 180 度翻转
                    transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
                    combined_pcd.transform(transform)
                    
                    # 添加一个坐标系帮助参考 (红X, 绿Y, 蓝Z)
                    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                    
                    # 阻塞式可视化，关掉窗口才会继续循环
                    o3d.visualization.draw_geometries([combined_pcd, coord_frame], window_name="3D Target Point Cloud")
                else:
                    print("[!] 当前快照未检测到目标，或目标深度无效。")

    finally:
        if 'pipeline' in locals():
            try: pipeline.stop()
            except: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()