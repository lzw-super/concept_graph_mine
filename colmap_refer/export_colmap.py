
import os
import struct
import numpy as np
import open3d as o3d
import yaml
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation

# COLMAP camera model IDs
# We use PINHOLE model which has ID 1 and params [f_x, f_y, c_x, c_y]
CAMERA_MODEL_ID = 1
CAMERA_MODEL_NAME = "PINHOLE"

def read_replica_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    params = config['camera_params']
    return {
        'width': params['image_width'],
        'height': params['image_height'],
        'fx': params['fx'],
        'fy': params['fy'],
        'cx': params['cx'],
        'cy': params['cy']
    }

def read_trajectory(traj_path):
    """
    Read Replica trajectory file.
    Format is 4x4 transformation matrix flattened row-wise.
    T_wc (Camera to World)
    """
    poses = []
    with open(traj_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pose = np.array([float(x) for x in line.strip().split()]).reshape(4, 4)
            poses.append(pose)
    return poses

def write_cameras_bin(cameras, path):
    """
    Write cameras.bin file.
    Format:
    num_cameras (uint64)
    camera_id (uint32), model_id (int32), width (uint64), height (uint64), params (double[])
    """
    with open(path, "wb") as fid:
        # 使用小端模式写入相机数量
        fid.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            camera_id = camera["id"]
            model_id = camera["model_id"]
            width = camera["width"]
            height = camera["height"]
            params = camera["params"]
            
            # 使用小端模式写入相机属性
            fid.write(struct.pack("<iiQQ", camera_id, model_id, width, height))
            for param in params:
                fid.write(struct.pack("<d", param))

def write_images_bin(images, path):
    """
    Write images.bin file.
    Format:
    num_images (uint64)
    image_id (uint32), qw (double), qx (double), qy (double), qz (double), tx (double), ty (double), tz (double), camera_id (uint32), name (string + \0), num_points2D (uint64), points2D (double[][2], point3D_ids (uint64))
    """
    with open(path, "wb") as fid:
        # 使用小端模式写入图像数量
        fid.write(struct.pack("<Q", len(images)))
        for image in images:
            image_id = image["id"]
            q = image["q"] # w, x, y, z
            t = image["t"]
            camera_id = image["camera_id"]
            name = image["name"].encode("utf-8")
            
            # 使用小端模式写入图像属性
            fid.write(struct.pack("<idddddddi", image_id, q[0], q[1], q[2], q[3], t[0], t[1], t[2], camera_id))
            fid.write(name + b"\0")
            
            # Write empty 2D points as we don't have feature matching
            # 使用小端模式写入 2D 点数量 (0)
            fid.write(struct.pack("<Q", 0))

def write_points3d_bin(points, path):
    """
    Write points3D.bin file.
    Format:
    num_points (uint64)
    point3D_id (uint64), x (double), y (double), z (double), r (uint8), g (uint8), b (uint8), error (double), track[] (track_length (uint64), image_id (uint32), point2D_idx (uint32))
    """
    with open(path, "wb") as fid:
        # 使用小端模式写入点数量
        fid.write(struct.pack("<Q", len(points)))
        for pt in points:
            point3D_id = pt["id"]
            xyz = pt["xyz"]
            rgb = pt["rgb"]
            error = pt["error"]
            
            # 使用小端模式写入点属性
            fid.write(struct.pack("<QdddBBBd", point3D_id, xyz[0], xyz[1], xyz[2], rgb[0], rgb[1], rgb[2], error))
            # Write empty track
            # 使用小端模式写入轨迹长度 (0)
            fid.write(struct.pack("<Q", 0))

def main():
    # Paths
    traj_path = "/home/zhengwu/Desktop/concept-graphs/Datasets/Replica/room0/traj.txt"
    config_path = "/home/zhengwu/Desktop/concept-graphs/conceptgraph/dataset/dataconfigs/replica/replica.yaml"
    ply_path = "/home/zhengwu/Desktop/concept-graphs/Datasets/Replica/room0/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_post.ply"
    
    # Output directory
    output_dir = "/home/zhengwu/Desktop/concept-graphs/colmap_refer/sparse_mine/0"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Reading configuration...")
    cam_params = read_replica_config(config_path)
    
    print("Reading trajectory...")
    poses = read_trajectory(traj_path)
    
    print("Reading point cloud...")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    
    # 1. Generate cameras.bin
    print("Generating cameras.bin...")
    # Replica uses same intrinsics for all frames, so only 1 camera
    cameras = [{
        "id": 1,
        "model_id": CAMERA_MODEL_ID,
        "width": cam_params['width'],
        "height": cam_params['height'],
        "params": [
            cam_params['fx'],
            cam_params['fy'],
            cam_params['cx'],
            cam_params['cy']
        ]
    }]
    write_cameras_bin(cameras, os.path.join(output_dir, "cameras.bin"))
    
    # 2. Generate images.bin
    print("Generating images.bin...")
    images = []
    # COLMAP expects T_cw (World to Camera), but Replica gives T_wc (Camera to World)
    # We need to invert the poses
    for i, pose_wc in enumerate(tqdm(poses)):
        # Invert pose to get World to Camera
        pose_cw = np.linalg.inv(pose_wc)
        
        R = pose_cw[:3, :3]
        t = pose_cw[:3, 3]
        
        # Convert rotation matrix to quaternion (w, x, y, z)
        rot = Rotation.from_matrix(R)
        q = rot.as_quat() # returns [x, y, z, w]
        q = np.roll(q, 1) # convert to [w, x, y, z]
        
        # Assuming image names follow format "frame000000.jpg" or similar
        # Based on Replica structure, it's usually "results/frame000000.jpg"
        image_name = f"results/frame{i:06d}.jpg"
        
        images.append({
            "id": i + 1, # 1-based index
            "q": q,
            "t": t,
            "camera_id": 1,
            "name": image_name
        })
    write_images_bin(images, os.path.join(output_dir, "images.bin"))
    
    # 3. Generate points3D.bin
    print("Generating points3D.bin...")
    points3d = []
    for i in tqdm(range(len(points))):
        points3d.append({
            "id": i + 1, # 1-based index
            "xyz": points[i],
            "rgb": colors[i],
            "error": 0.0 # Default error
        })
    write_points3d_bin(points3d, os.path.join(output_dir, "points3D.bin"))
    
    print(f"Done! COLMAP files generated in {output_dir}")

if __name__ == "__main__":
    main()
