import os
import cv2
import numpy as np
import open3d as o3d
import json
# 注意：我们没有真实时间戳，所以用虚拟时间（0.1s 间隔）
# from scipy.spatial.transform import Rotation as R
# -----------------------------
# 配置路径
# -----------------------------
dataset_root = r'E:\Users\lenovo\Desktop\aligned_capture_20251210_172555\aligned_capture_20251210_172555'
# results_dir = os.path.join(dataset_root, "results") 
results_dir = r'/home/zhengwu/Desktop/concept-graphs/Datasets/Replica/room0/results'
# intrinsic_file = os.path.join(dataset_root, "camera_intrinsic.txt")

# 自动获取所有 frame 和 depth 文件（按数字排序）
rgb_files = sorted([f for f in os.listdir(results_dir) if f.startswith("frame") and f.endswith(".jpg")])
depth_files = sorted([f for f in os.listdir(results_dir) if f.startswith("depth") and f.endswith(".png")])

# 确保数量一致
assert len(rgb_files) == len(depth_files), "RGB 和 Depth 图像数量不匹配！"
num_frames = len(rgb_files)

print(f"共找到 {num_frames} 帧 RGB-D 图像。")

# -----------------------------
# 读取相机内参
# -----------------------------
def load_intrinsic(intrinsic_path):
    with open(intrinsic_path, 'r') as f:
        content = f.read().strip()
    try:
        # 尝试 JSON 格式
        data = json.loads(content)
        fx = data["fx"]
        fy = data["fy"]
        cx = data["cx"]
        cy = data["cy"]
    except json.JSONDecodeError:
        # 尝试空格分隔的纯文本格式: fx fy cx cy
        values = list(map(float, content.split()))
        assert len(values) == 4, "内参文件应包含 fx, fy, cx, cy 四个值"
        fx, fy, cx, cy = values
    return fx, fy, cx, cy

fx, fy, cx, cy = 600.0, 600.0 , 599.5 , 339.5
print(f"相机内参: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

# 获取图像尺寸（从第一张图读取）
first_rgb = cv2.imread(os.path.join(results_dir, rgb_files[0]))
height, width = first_rgb.shape[:2]

intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width, height, fx, fy, cx, cy
)

# -----------------------------
# 初始化轨迹
# -----------------------------
trajectory = [np.eye(4)]  # 第一帧设为世界原点
prev_rgbd = None

# 辅助函数：提取帧序号（如 "frame123.jpg" → 123）
def get_frame_id(filename):
    return int(''.join(filter(str.isdigit, filename)))

# 确保按数字顺序排列（虽然 sorted 通常已正确）
rgb_files.sort(key=get_frame_id)
depth_files.sort(key=get_frame_id)

# -----------------------------
# 逐帧处理
# -----------------------------
for i in range(num_frames):
    rgb_path = os.path.join(results_dir, rgb_files[i])
    depth_path = os.path.join(results_dir, depth_files[i])
    
    color = cv2.imread(rgb_path)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # 16-bit
    
    if color is None or depth_raw is None:
        print(f"警告：跳过第 {i} 帧（图像读取失败）")
        continue
        
    depth = depth_raw.astype(np.float32)  # 转为 float 供 Open3D 使用

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color),
        o3d.geometry.Image(depth),
        depth_scale=6553.5,      # mm → m
        depth_trunc=10.0,        # 忽略 10 米以外的深度（可调）
        convert_rgb_to_intensity=False
    )

    if i == 0:
        prev_rgbd = rgbd
        continue

    # 计算当前帧相对于上一帧的变换
    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    success, trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
        prev_rgbd, rgbd, intrinsic,
        np.eye(4), jacobian,
        o3d.pipelines.odometry.OdometryOption()
    )
    if success:
        current_pose = trajectory[-1] @ trans
        trajectory.append(current_pose)
        prev_rgbd = rgbd
        print(f"Frame {i}: 成功跟踪")
    else:
        print(f"Frame {i}: 跟踪失败！复用上一帧位姿")
        trajectory.append(trajectory[-1])  # 保持上一位置

# -----------------------------
# 保存为 TUM 格式轨迹 (timestamp tx ty tz qx qy qz qw)
# -----------------------------


output_file = os.path.join("poses.txt")

with open(output_file, 'w') as f:
    for pose in trajectory:
        # 将 4x4 矩阵展平为 1D 数组（行优先）
        flattened = pose.flatten()  # shape: (16,)
        # 转为字符串，保留6位小数（可调）
        line = ' '.join([f"{x:.18f}" for x in flattened])
        f.write(line + '\n')

print(f"✅ 已将 {len(trajectory)} 个位姿写入 {output_file}")