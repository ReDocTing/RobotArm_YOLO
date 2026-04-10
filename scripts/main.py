"""
统一版 3D 视觉目标检测管线 (支持 D435i / Gemini 2 双相机接入)
特性：外部 YAML 配置化、动态模型加载、统一 3D 坐标系转换

用法：
    cd /your/path/to/cameraws
    python scripts/main.py
"""

import os
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

import sys
import cv2
import yaml
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# 全局状态与鼠标回调
# ==========================================
clicked_point = {"u": -1, "v": -1}

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point["u"] = x
        clicked_point["v"] = y
        print(f"[测试] 鼠标点击锁定像素: (u={x}, v={y})")

# ==========================================
# 通用辅助函数
# ==========================================
def load_config(yaml_path):
    if not os.path.exists(yaml_path):
        print(f"[错误] 找不到配置文件: {yaml_path}，请检查路径。")
        sys.exit(1)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_center_depth_rs(depth_frame, x, y, roi_size=5):
    w, h = depth_frame.get_width(), depth_frame.get_height()
    half_roi = roi_size // 2
    x_min, x_max = max(0, x - half_roi), min(w, x + half_roi + 1)
    y_min, y_max = max(0, y - half_roi), min(h, y + half_roi + 1)
    
    depths = [depth_frame.get_distance(i, j) for i in range(x_min, x_max) for j in range(y_min, y_max) if depth_frame.get_distance(i, j) > 0]
    return np.median(depths) if len(depths) > 0 else 0.0

def get_center_depth_ob(depth_map, x, y, roi_size=5):
    h, w = depth_map.shape
    half_roi = roi_size // 2
    x_min, x_max = max(0, x - half_roi), min(w, x + half_roi + 1)
    y_min, y_max = max(0, y - half_roi), min(h, y + half_roi + 1)
    
    roi = depth_map[y_min:y_max, x_min:x_max]
    valid_depths = roi[roi > 0]
    return np.median(valid_depths) if len(valid_depths) > 0 else 0.0

# ==========================================
# 主流程
# ==========================================
def main():
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "default.yaml"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True) 
    
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
    
    model_path = models_dir / model_name

    print(f"=== 初始化 YOLO 模型 ===")
    print(f"尝试加载模型: {model_path}")
    model = YOLO(str(model_path)) 
    
    is_open_vocab = use_world and ("world" in model_name.lower() or "yoloe" in model_name.lower())
    if is_open_vocab:
        print(f"启用开放词汇 (Open-Vocabulary) 模式，注入 {len(custom_classes)} 种物品概念...")
        model.set_classes(custom_classes)
        
    print(f"YOLO 模型加载完毕！使用计算平台: {device.upper()}")
    if "26" in model_name:
        print(f"[*] 检测到 YOLO26 家族模型，将启用免 NMS 端到端极速推理特性。")

    print(f"\n=== 初始化相机: {cam_type} ===")
    
    if "realsense" in cam_type:
        try:
            import pyrealsense2 as rs
        except ImportError:
            print("[错误] 未安装 pyrealsense2，请执行: pip install pyrealsense2")
            sys.exit(1)
            
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, color_w, color_h, rs.format.z16, fps)
        
        try:
            profile = pipeline.start(config)
        except RuntimeError as e:
            print(f"\n[致命错误] 无法启动 RealSense 相机！")
            print(f"可能原因：1. 相机未连接数据线；2. USB 接口松动；3. 被其他程序占用。")
            print(f"底层报错信息: {e}")
            sys.exit(1)
            
        align = rs.align(rs.stream.color)
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        print(f"[相机就绪] D435i (分辨率:{intrinsics.width}x{intrinsics.height})")
        
    elif "orbbec" in cam_type:
        try:
            from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
        except ImportError:
            print("[错误] 未安装 pyorbbecsdk")
            sys.exit(1)

        try:
            from pyorbbecsdk import Context, OBLogSeverity
            Context().set_logger_severity(OBLogSeverity.FATAL)
        except Exception:
            pass

        try:
            pipeline = Pipeline()
        except Exception as e:
            print(f"\n[致命错误] 找不到 Orbbec 设备！")
            print(f"可能原因：1. 相机未连接；2. USB 接口松动或未识别；3. udev 规则未配置。")
            print(f"底层报错信息: {e}")
            sys.exit(1)

        config = Config()
        
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile = profile_list.get_video_stream_profile(color_w, color_h, OBFormat.MJPG, fps)
        except Exception:
            try:
                color_profile = profile_list.get_video_stream_profile(color_w, color_h, OBFormat.RGB, fps)
            except Exception:
                color_profile = profile_list.get_default_video_stream_profile()
                
        config.enable_stream(color_profile)
        
        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            depth_profile = depth_profile_list.get_video_stream_profile(color_w, color_h, OBFormat.Y16, fps)
        except Exception:
            depth_profile = depth_profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        config.set_align_mode(OBAlignMode.HW_MODE)
        
        try:
            pipeline.start(config)
        except Exception as e:
            print(f"\n[致命错误] 无法启动 Orbbec Gemini 相机流！")
            print(f"可能原因：1. 接口带宽不足(如插在USB 2.0)；2. 设备忙(Return Code: -6)。")
            print(f"请尝试物理重新插拔相机，或执行: pkill -9 -f python")
            print(f"底层报错信息: {e}")
            sys.exit(1)
            
        camera_param = pipeline.get_camera_param()
        fx, fy = camera_param.rgb_intrinsic.fx, camera_param.rgb_intrinsic.fy
        cx, cy = camera_param.rgb_intrinsic.cx, camera_param.rgb_intrinsic.cy
        print(f"[相机就绪] Gemini 2 (fx:{fx:.2f}, cx:{cx:.2f})")
    else:
        print(f"[错误] 不支持的相机类型: {cam_type}")
        sys.exit(1)

    print("\n[操作提示] 按鼠标左键进行坐标点测，按 [Q] 退出程序")
    window_name = f"Unified 3D Vision ({cam_type})"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        while True:
            color_image, depth_frame, depth_map = None, None, None
            
            if "realsense" in cam_type:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                cf, df = aligned_frames.get_color_frame(), aligned_frames.get_depth_frame()
                if not cf or not df: continue
                color_image = np.asanyarray(cf.get_data())
                depth_frame = df 
                
            elif "orbbec" in cam_type:
                frames = pipeline.wait_for_frames(500)
                if frames is None: continue
                cf, df = frames.get_color_frame(), frames.get_depth_frame()
                if not cf or not df: continue
                
                w_ob, h_ob, fmt_ob = cf.get_width(), cf.get_height(), cf.get_format()
                raw_color = np.ascontiguousarray(np.asanyarray(cf.get_data()), dtype=np.uint8)
                try:
                    if fmt_ob == OBFormat.MJPG:
                        color_image = cv2.imdecode(raw_color, cv2.IMREAD_COLOR)
                    elif fmt_ob == OBFormat.RGB:
                        color_image = cv2.cvtColor(raw_color.reshape((h_ob, w_ob, 3)), cv2.COLOR_RGB2BGR)
                    else:
                        color_image = raw_color.reshape((h_ob, w_ob, 3))
                except: continue
                
                if color_image is None: continue
                depth_map = np.frombuffer(df.get_data(), dtype=np.uint16).reshape((h_ob, w_ob))

            results = model.predict(color_image, verbose=False, device=device)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id, conf = int(box.cls[0]), float(box.conf[0])
                    class_name = model.names[cls_id]
                    
                    u, v = (x1 + x2) // 2, (y1 + y2) // 2
                    x_m, y_m, z_m = 0.0, 0.0, 0.0
                    valid_depth = False
                    
                    if "realsense" in cam_type:
                        z_m = get_center_depth_rs(depth_frame, u, v, 5)
                        if z_m > 0:
                            pt = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], z_m)
                            x_m, y_m, z_m = pt[0], pt[1], pt[2]
                            valid_depth = True
                    elif "orbbec" in cam_type:
                        z_mm = get_center_depth_ob(depth_map, u, v, 5)
                        if z_mm > 0:
                            z_m = z_mm / 1000.0
                            x_m, y_m = (u - cx) * z_m / fx, (v - cy) * z_m / fy
                            valid_depth = True

                    if valid_depth:
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(color_image, (u, v), 5, (0, 0, 255), -1)
                        text_label = f"{class_name} {conf:.2f}"
                        coord_label = f"X:{x_m:.2f} Y:{y_m:.2f} Z:{z_m:.2f} (m)"
                        cv2.rectangle(color_image, (x1, y1 - 40), (x1 + max(len(text_label), len(coord_label))*10, y1), (0, 0, 0), -1)
                        cv2.putText(color_image, text_label, (x1 + 5, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(color_image, coord_label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(color_image, f"{class_name} (No Depth)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            cu, cv = clicked_point["u"], clicked_point["v"]
            if cu != -1 and cv != -1:
                cx_m, cy_m, cz_m = 0.0, 0.0, 0.0
                valid_test = False
                
                if "realsense" in cam_type:
                    if 0 <= cu < intrinsics.width and 0 <= cv < intrinsics.height:
                        cz_m = get_center_depth_rs(depth_frame, cu, cv)
                        if cz_m > 0:
                            pt = rs.rs2_deproject_pixel_to_point(intrinsics, [cu, cv], cz_m)
                            cx_m, cy_m, cz_m = pt[0], pt[1], pt[2]
                            valid_test = True
                elif "orbbec" in cam_type:
                    h, w = depth_map.shape
                    if 0 <= cu < w and 0 <= cv < h:
                        cz_mm = get_center_depth_ob(depth_map, cu, cv)
                        if cz_mm > 0:
                            cz_m = cz_mm / 1000.0
                            cx_m, cy_m = (cu - cx) * cz_m / fx, (cv - cy) * cz_m / fy
                            valid_test = True

                if valid_test:
                    cv2.drawMarker(color_image, (cu, cv), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
                    test_label = f"TEST -> X:{cx_m:.3f} Y:{cy_m:.3f} Z:{cz_m:.3f} m"
                    cv2.rectangle(color_image, (cu + 5, cv - 25), (cu + 320, cv + 5), (0, 0, 0), -1)
                    cv2.putText(color_image, test_label, (cu + 10, cv - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            cv2.putText(color_image, f"{cam_type.upper()} | {model_name.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(window_name, color_image)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]: break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break

    finally:
        if 'pipeline' in locals():
            try: pipeline.stop()
            except: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()