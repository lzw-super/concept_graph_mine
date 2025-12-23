
import gzip
import pickle
import numpy as np
import open3d as o3d
import os
import argparse

def main():
    # 设置输入文件路径
    input_path = "/home/zhengwu/Desktop/concept-graphs/Datasets/Replica/room0/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_post.pkl.gz"
    
    # 设置输出文件路径
    output_path = input_path.replace(".pkl.gz", ".ply")
    
    print(f"Loading data from {input_path}...")
    
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    try:
        with gzip.open(input_path, "rb") as f:
            results = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    all_points = []
    all_colors = []

    # 处理前景对象
    if 'objects' in results and results['objects'] is not None:
        print(f"Processing {len(results['objects'])} foreground objects...")
        for i, obj in enumerate(results['objects']):
            if 'pcd_np' in obj:
                points = obj['pcd_np']
                all_points.append(points)
                
                if 'pcd_color_np' in obj:
                    colors = obj['pcd_color_np']
                    all_colors.append(colors)
                else:
                    # 如果没有颜色，使用默认颜色（例如白色）
                    all_colors.append(np.ones_like(points) * 0.5)
            else:
                print(f"Warning: Object {i} has no point cloud data ('pcd_np')")

    # 处理背景对象
    if 'bg_objects' in results and results['bg_objects'] is not None:
        print(f"Processing {len(results['bg_objects'])} background objects...")
        for i, obj in enumerate(results['bg_objects']):
            if 'pcd_np' in obj:
                points = obj['pcd_np']
                all_points.append(points)
                
                if 'pcd_color_np' in obj:
                    colors = obj['pcd_color_np']
                    all_colors.append(colors)
                else:
                    all_colors.append(np.ones_like(points) * 0.5)

    if not all_points:
        print("No point cloud data found in the file.")
        return

    # 合并所有点和颜色
    print("Concatenating points and colors...")
    final_points = np.concatenate(all_points, axis=0)
    final_colors = np.concatenate(all_colors, axis=0)

    print(f"Total points: {final_points.shape[0]}")

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    pcd.colors = o3d.utility.Vector3dVector(final_colors)

    # 保存为PLY文件
    print(f"Saving point cloud to {output_path}...")
    o3d.io.write_point_cloud(output_path, pcd)
    print("Done!")

if __name__ == "__main__":
    main()
