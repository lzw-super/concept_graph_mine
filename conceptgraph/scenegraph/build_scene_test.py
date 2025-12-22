import gc  # 垃圾回收模块，用于释放内存
import gzip  # 压缩文件操作模块
import json  # JSON文件处理模块
import os  # 操作系统接口
import pickle as pkl  # Python对象序列化模块
import time  # 时间模块
from dataclasses import dataclass  # 数据类装饰器
from pathlib import Path  # 路径处理模块
from types import SimpleNamespace  # 简单命名空间类型
from typing import List, Literal, Union  # 类型提示
from textwrap import wrap  # 文本换行函数
from conceptgraph.utils.general_utils import prjson  # 自定义JSON打印函数

import cv2  # OpenCV库，用于图像处理
import matplotlib  # 可视化库
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt  # 绘图模块

import numpy as np  # 数值计算库
import rich  # 富文本终端输出库
import torch  # PyTorch深度学习框架
import tyro  # 命令行参数解析库
from PIL import Image  # Python图像处理库
from scipy.sparse import csr_matrix  # 稀疏矩阵库
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree  # 图算法库
from tqdm import tqdm, trange  # 进度条显示库
from transformers import logging as hf_logging  # Hugging Face日志控制

from conceptgraph.slam.slam_classes import MapObjectList
def load_scene_map(map_file, scene_map):
    """
    从gzip压缩的pickle文件加载场景地图。
    
    这个函数需要处理不同格式的地图文件，因为根据生成方式的不同（cfslam_pipeline_batch.py或merge_duplicate_objects.py），
    文件格式可能会有所不同。
    
    参数:
        args: 程序参数，包含mapfile路径
        scene_map: MapObjectList对象，用于存储加载的场景地图
    
    函数检查反序列化对象的结构，以确定正确的加载方式。支持两种预期格式：
    1. 包含"objects"键的字典
    2. 列表或字典（根据实际需求调整）
    """
    print(map_file)  # 打印地图文件路径
    with gzip.open(map_file, "rb") as f:  # 打开压缩文件
        loaded_data = pkl.load(f)  # 反序列化对象
        
        # 注释掉的路径示例
        # /home/zhengwu/Desktop/concept-graphs/Datasets/Replica/room0/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_post.pkl.gz
        
        # 检查加载数据的类型以决定如何处理
        if isinstance(loaded_data, dict) and "objects" in loaded_data:
            # 如果是包含"objects"键的字典，加载该键的值
            scene_map.load_serializable(loaded_data["objects"])
        elif isinstance(loaded_data, list) or isinstance(loaded_data, dict):  
            # 如果是列表或字典，直接加载
            scene_map.load_serializable(loaded_data)
        else:
            # 如果格式不符合预期，抛出错误
            raise ValueError("Unexpected data format in map file.")
        
        # 打印加载的对象数量
        print(f"Loaded {len(scene_map)} objects")

def crop_image_and_mask(image: Image, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
    """
    同时裁剪图像和掩码，并可选择添加填充。
    
    这个函数将图像和掩码一起裁剪，以避免分开裁剪时可能出现的形状不匹配问题。
    
    参数:
        image: PIL图像对象
        mask: 表示对象区域的二维numpy数组掩码
        x1, y1, x2, y2: 边界框坐标（左上角和右下角）
        padding: 边界框周围的填充像素数，默认为0
    
    返回:
        image_crop: 裁剪后的PIL图像
        mask_crop: 裁剪后的掩码
        如果裁剪后图像和掩码形状不匹配，则返回None, None
    """
    # 将PIL图像转换为numpy数组
    image = np.array(image)
    
    # 验证初始尺寸是否匹配
    if image.shape[:2] != mask.shape:
        raise ValueError("Initial shape mismatch: Image shape {} != Mask shape {}".format(image.shape, mask.shape))
        
    # 定义裁剪坐标，考虑填充并确保不超出图像边界
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)  # image.shape[1]是图像宽度
    y2 = min(image.shape[0], y2 + padding)  # image.shape[0]是图像高度
    
    # 将坐标四舍五入为整数
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

    # 裁剪图像和掩码
    image_crop = image[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    # 验证裁剪后的尺寸是否匹配
    if image_crop.shape[:2] != mask_crop.shape:
        print("Cropped shape mismatch: Image crop shape {} != Mask crop shape {}".format(image_crop.shape, mask_crop.shape))
        return None, None
    
    # 将裁剪后的图像转换回PIL图像
    image_crop = Image.fromarray(image_crop)
    
    return image_crop, mask_crop

map_file = '/home/zhengwu/Desktop/concept-graphs/Datasets/Replica/room0/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_post.pkl.gz'
scene_map = MapObjectList()
load_scene_map(map_file, scene_map)  
build_out = '/home/zhengwu/Desktop/concept-graphs/conceptgraph/scenegraph/build_see' 
if not os.path.exists(build_out):
    os.makedirs(build_out, exist_ok=True)
for idx_obj, obj in tqdm(enumerate(scene_map), total=len(scene_map)): 
    build_out_idx = os.path.join(build_out, f'{idx_obj:04d}')
    if not os.path.exists(build_out_idx):
        os.makedirs(build_out_idx, exist_ok=True)
    conf = obj["conf"]  # 获取检测置信度列表
    conf = np.array(conf)
    # 按置信度降序排列索引
    idx_most_conf = np.argsort(conf)[::-1]

    # 初始化存储特征、描述和置信度的列表
    features = []  # 图像特征列表
    captions = []  # 对象描述列表
    low_confidences = []  # 低置信度标记列表
    
    # 调试用列表
    image_list = []  # 图像列表
    caption_list = []  # 描述列表
    confidences_list = []  # 置信度列表
    low_confidences_list = []  # 低置信度标记列表
    mask_list = []  # 掩码列表
    
    # 跳过检测数量少于2的对象
    if len(idx_most_conf) < 2:
        continue 
        
    # 选择置信度最高的检测，最多args.max_detections_per_object个
    idx_most_conf = idx_most_conf[:10] 
    lzw = 0  # 计数变量，最多处理8个检测

    # 处理每个高置信度检测
    for idx_det in tqdm(idx_most_conf): 
        # if lzw > 7:  # 限制处理的检测数量为8个
        #     continue
            
        # 打开并转换为RGB图像
        image = Image.open(obj["color_path"][idx_det]).convert("RGB")
        xyxy = obj["xyxy"][idx_det]  # 获取边界框坐标
        class_id = obj["class_id"][idx_det]  # 获取类别ID 
        class_name = obj["class_name"][idx_det]  # 获取类别名称
        mask = obj["mask"][idx_det]  # 获取对象掩码
        
        # 裁剪参数设置
        padding = 10  # 边界框周围的填充像素数
        x1, y1, x2, y2 = xyxy  # 解析边界框坐标
        
        # 裁剪图像和掩码
        image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding) 
        _w, _h = image_crop.size
        if _w * _h < 70 * 70:
            print("small object. Skipping LLaVA captioning...")
            continue
        image_crop_path = os.path.join(build_out_idx, f'{class_name}_{idx_det:04d}.jpg')
        image_crop.save(image_crop_path)
        # image_crop.show()

        print('\n')