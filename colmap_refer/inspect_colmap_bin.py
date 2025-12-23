import os
import struct
import numpy as np
import argparse

# COLMAP 相机模型 (COLMAP Camera Models)
CAMERA_MODELS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE"
}

def read_cameras_bin(path):
    print(f"\n--- Analyzing {path} ---")
    cameras = {}
    with open(path, "rb") as fid:
        # 读取相机数量 (Read number of cameras)
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        print(f"Number of cameras: {num_cameras}")
        
        for _ in range(num_cameras):
            # 读取相机属性: camera_id (int), model_id (int), width (uint64), height (uint64)
            # 4 + 4 + 8 + 8 = 24 bytes
            camera_properties = struct.unpack("<iiQQ", fid.read(24))
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            
            model_name = CAMERA_MODELS.get(model_id, "UNKNOWN")
            
            # 根据 model_id 读取参数 (Read parameters based on model_id)
            # 为了简单起见，我们读取剩余的字节直到下一个已知结构或 EOF
            # PINHOLE (id=1) 有 4 个参数。
            if model_id == 1: # PINHOLE
                params = struct.unpack("<dddd", fid.read(32))
            elif model_id == 0: # SIMPLE_PINHOLE
                params = struct.unpack("<ddd", fid.read(24))
            else:
                # 检查时的回退方案：读取一些 double 并希望一切顺利，或者跳过严格解析
                # 在真正的读取器中我们需要严格的映射。目前我们假设是 PINHOLE。
                print(f"Warning: Model ID {model_id} ({model_name}) parsing not fully implemented in this quick inspector.")
                params = [] 

            cameras[camera_id] = {
                "model": model_name,
                "width": width,
                "height": height,
                "params": params
            }
            
            # 打印前几个相机 (Print first few cameras)
            if len(cameras) <= 3:
                print(f"Camera {camera_id}: Model={model_name}, Size={width}x{height}, Params={params}")

    return cameras

def read_images_bin(path):
    print(f"\n--- Analyzing {path} ---")
    images = {}
    with open(path, "rb") as fid:
        # 读取图像数量 (Read number of images)
        num_images = struct.unpack("<Q", fid.read(8))[0]
        print(f"Number of images: {num_images}")
        
        for i in range(num_images):
            # 读取图像属性: image_id (int), qw, qx, qy, qz, tx, ty, tz, camera_id (int)
            # 4 + 4*8 + 3*8 + 4 = 64 bytes
            binary_image_properties = fid.read(64)
            image_properties = struct.unpack("<idddddddi", binary_image_properties)
            image_id = image_properties[0]
            q = np.array(image_properties[1:5])
            t = np.array(image_properties[5:8])
            camera_id = image_properties[8]
            
            name = ""
            current_char = fid.read(1)
            while current_char != b"\0":
                name += current_char.decode("utf-8")
                current_char = fid.read(1)
                
            # 读取 2D 点数量 (Read number of 2D points)
            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            
            # 读取 2D 点: x, y, point3D_id
            # 每个点是 (double, double, uint64) -> 8 + 8 + 8 = 24 bytes
            # 为了检查速度，我们跳过读取所有点的内容，直接 seek
            fid.seek(24 * num_points2D, 1)
            
            images[image_id] = {
                "name": name,
                "camera_id": camera_id,
                "t": t,
                "num_points2D": num_points2D
            }
            
            if i < 3:
                print(f"Image {image_id}: Name={name}, Camera={camera_id}, Pos={t}, NumPoints2D={num_points2D}")
                
    return images

def read_points3d_bin(path):
    print(f"\n--- Analyzing {path} ---")
    points3D = {}
    with open(path, "rb") as fid:
        # 读取 3D 点数量 (Read number of 3D points)
        num_points = struct.unpack("<Q", fid.read(8))[0]
        print(f"Number of 3D points: {num_points}")
        
        # 读取前几个点以展示结构 (Read first few points to show structure)
        # 注意：这里我们不能只 seek 跳过，因为我们需要知道每个点的 track_length 才能跳到下一个点
        # 如果只想读前几个，读完前几个后就 break 即可，不需要读完整个文件
        
        # 为了避免读取整个大文件，我们只读取前 5 个点，然后停止
        for i in range(num_points):
            if i >= 5:
                break
                
            # 读取点属性: point3D_id (uint64), x, y, z (double), r, g, b (uint8), error (double)
            # 8 + 3*8 + 3*1 + 8 = 43 bytes
            binary_point_properties = fid.read(43)
            point_properties = struct.unpack("<QdddBBBd", binary_point_properties)
            point3D_id = point_properties[0]
            xyz = np.array(point_properties[1:4])
            rgb = np.array(point_properties[4:7])
            error = point_properties[7]
            
            # 读取轨迹长度 (Read track length)
            track_length = struct.unpack("<Q", fid.read(8))[0]
            # 轨迹元素: image_id (uint32), point2D_idx (uint32) -> 4 + 4 = 8 bytes
            fid.seek(8 * track_length, 1)
            
            print(f"Point {point3D_id}: XYZ={xyz}, RGB={rgb}, Error={error}, TrackLen={track_length}")
            
    return points3D

def main():
    base_path = "/home/zhengwu/Desktop/concept-graphs/colmap_refer/sparse/0"
    
    cameras_path = os.path.join(base_path, "cameras.bin")
    if os.path.exists(cameras_path):
        read_cameras_bin(cameras_path)
    else:
        print(f"File not found: {cameras_path}")
        
    images_path = os.path.join(base_path, "images.bin")
    if os.path.exists(images_path):
        read_images_bin(images_path)
    else:
        print(f"File not found: {images_path}")
        
    points_path = os.path.join(base_path, "points3D.bin")
    if os.path.exists(points_path):
        read_points3d_bin(points_path)
    else:
        print(f"File not found: {points_path}")

if __name__ == "__main__":
    main()
