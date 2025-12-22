'''\n# 该脚本用于在RGB-D数据集上提取Grounded SAM的分割结果\n# Grounded SAM (Grounded Segment Anything Model) 是一种结合了目标检测与分割能力的模型\n# 该脚本将使用Grounded DINO或YOLO-World进行目标检测，然后使用SAM进行精确分割\n# 支持多种SAM变体：原始SAM、MobileSAM、LightHQSAM和FastSAM\n# 支持多种类别设置方式：场景特定类别、通用类别、最小类别、Tag2Text自动标记等\n# 最终结果将保存到场景文件夹下的相应目录中\n
The script is used to extract Grounded SAM results on a posed RGB-D dataset. 
The results will be dumped to a folder under the scene folder. 
'''

# 基础库导入
import os  # 操作系统接口
import argparse  # 命令行参数解析
from pathlib import Path  # 路径处理
import re  # 正则表达式
from typing import Any, List  # 类型提示
from PIL import Image  # 图像处理库
import cv2  # OpenCV计算机视觉库
import json  # JSON数据处理
import imageio  # 视频和图像IO
import matplotlib  # 可视化库
matplotlib.use("TkAgg")  # 设置matplotlib后端
from matplotlib import pyplot as plt  # 绘图工具
import numpy as np  # 数值计算库
import pickle  # Python对象序列化
import gzip  # 数据压缩
import open_clip  # OpenCLIP库，用于特征提取

# PyTorch相关导入
import torch  # PyTorch深度学习框架
import torchvision  # PyTorch的计算机视觉扩展
from torch.utils.data import Dataset  # 数据集基类
import supervision as sv  # 目标检测与分割结果处理库
from tqdm import trange  # 进度条显示

# 自定义模块导入
from conceptgraph.dataset.datasets_common import get_dataset  # 获取数据集的函数
from conceptgraph.utils.vis import vis_result_fast, vis_result_slow_caption  # 可视化结果的函数
from conceptgraph.utils.model_utils import compute_clip_features  # 计算CLIP特征的函数
import torch.nn.functional as F  # PyTorch的神经网络函数模块

# 尝试导入GroundedSAM相关模块
try: 
    from groundingdino.util.inference import Model  # GroundingDINO模型导入
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator  # SAM模型相关导入
except ImportError as e:
    print("Import Error: Please install Grounded Segment Anything following the instructions in README.")
    raise e  # 抛出导入错误，提示用户安装必要依赖

# 设置脚本中使用的路径
# 假设所有检查点文件已按照原始GSA仓库的说明下载
if "GSA_PATH" in os.environ:
    GSA_PATH = os.environ["GSA_PATH"]  # 从环境变量获取GSA仓库路径
else:
    raise ValueError("Please set the GSA_PATH environment variable to the path of the GSA repo. ")
    
import sys
TAG2TEXT_PATH = os.path.join(GSA_PATH, "")  # Tag2Text模块路径
EFFICIENTSAM_PATH = os.path.join(GSA_PATH, "EfficientSAM")  # EfficientSAM模块路径
# 将必要的路径添加到系统路径中，以便导入相应的模块
sys.path.append(GSA_PATH) # 这是文件中后续导入所必需的
sys.path.append(TAG2TEXT_PATH) # 这是Tag2Text文件中一些导入所必需的
sys.path.append(EFFICIENTSAM_PATH) # 这是EfficientSAM相关导入所必需的

import torchvision.transforms as TS  # PyTorch的图像预处理模块
try:
    # 尝试导入RAM（Recognize Anything Model）和Tag2Text模型相关模块
    from ram.models import ram  # RAM模型
    from ram.models import tag2text  # Tag2Text模型
    from ram import inference_tag2text, inference_ram  # 推理相关函数
except ImportError as e:
    print("RAM sub-package not found. Please check your GSA_PATH. ")
    raise e  # 抛出导入错误，提示用户检查GSA路径

# 禁用PyTorch梯度计算，因为推理过程不需要反向传播
torch.set_grad_enabled(False)
    
# GroundingDINO配置和检查点路径
GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")

# Segment-Anything检查点配置
SAM_ENCODER_VERSION = "vit_h"  # 使用ViT-H骨干网络版本的SAM
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")

# Tag2Text和RAM模型检查点路径
TAG2TEXT_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./tag2text_swin_14m.pth")
RAM_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./ram_swin_large_14m.pth")

# 通用前景类别列表，用于通用对象检测
FOREGROUND_GENERIC_CLASSES = [
    "item", "furniture", "object", "electronics", "wall decoration", "door"
]

# 最小类别列表，只包含一个通用类别
FOREGROUND_MINIMAL_CLASSES = [
    "item"
]

def get_parser() -> argparse.ArgumentParser:
    """创建并配置命令行参数解析器
    
    返回:
        argparse.ArgumentParser: 配置好的命令行参数解析器
    """
    parser = argparse.ArgumentParser()
    
    # 数据集相关参数
    parser.add_argument(
        "--dataset_root", type=Path, required=True,
        help="数据集根目录路径"
    )
    parser.add_argument(
        "--dataset_config", type=str, required=True,
        help="数据集配置文件路径，根据运行环境可能需要修改"
    )
    parser.add_argument("--scene_id", type=str, default="train_3",
                       help="场景ID")
    
    # 处理范围参数
    parser.add_argument("--start", type=int, default=0,
                       help="处理起始帧索引")
    parser.add_argument("--end", type=int, default=-1,
                       help="处理结束帧索引，-1表示处理到最后")
    parser.add_argument("--stride", type=int, default=1,
                       help="帧处理步长")

    # 图像尺寸参数
    parser.add_argument("--desired_height", type=int, default=480,
                       help="输出图像的目标高度")
    parser.add_argument("--desired_width", type=int, default=640,
                       help="输出图像的目标宽度")

    # 检测阈值参数
    parser.add_argument("--box_threshold", type=float, default=0.25,
                       help="边界框检测的置信度阈值")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                       help="文本提示的置信度阈值")
    parser.add_argument("--nms_threshold", type=float, default=0.5,
                       help="非最大抑制的IoU阈值")

    # 类别设置参数
    parser.add_argument("--class_set", type=str, default="scene", 
                        choices=["scene", "generic", "minimal", "tag2text", "ram", "none"], 
                        help="类别集合类型：scene(场景特定)、generic(通用)、minimal(最小)、tag2text、ram或none(无检测直接分割)")
    parser.add_argument("--detector", type=str, default="dino", 
                        choices=["yolo", "dino"], 
                        help="目标检测器类型：yolo(YOLO-World)或dino(GroundingDINO)")
    parser.add_argument("--add_bg_classes", action="store_true", 
                        help="是否添加背景类别(墙、地板、天花板)")
    parser.add_argument("--accumu_classes", action="store_true",
                        help="是否在帧间累积类别集合")

    # SAM模型变体参数
    parser.add_argument("--sam_variant", type=str, default="sam",
                        choices=['fastsam', 'mobilesam', "lighthqsam"],
                        help="使用的SAM模型变体")
    
    # 输出相关参数
    parser.add_argument("--save_video", action="store_true",
                        help="是否保存结果视频")
    
    # 设备设置参数
    parser.add_argument("--device", type=str, default="cuda",
                        help="运行设备：cuda或cpu")
    
    # 可视化参数
    parser.add_argument("--use_slow_vis", action="store_true", 
                        help="是否使用慢速可视化(仅在使用ram/tag2text时有效)")
    
    # 实验后缀参数
    parser.add_argument("--exp_suffix", type=str, default=None,
                        help="结果保存文件夹的后缀名")
    
    return parser





# 使用检测到的边界框提示SAM进行分割
def get_sam_segmentation_from_xyxy(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """使用检测到的边界框作为提示，让SAM生成精确分割掩码
    
    参数:
        sam_predictor: 预初始化的SAM预测器
        image: 输入图像，形状为(H, W, 3)，RGB格式，范围[0, 255]
        xyxy: 边界框坐标数组，形状为(N, 4)，格式为[x1, y1, x2, y2]
    
    返回:
        np.ndarray: 分割掩码数组，形状为(N, H, W)，每个掩码对应一个边界框
    """
    sam_predictor.set_image(image)  # 设置输入图像，提取图像特征
    result_masks = []  # 存储结果掩码的列表
    
    # 对每个边界框生成分割掩码
    for box in xyxy:
        # 使用SAM预测器生成掩码，开启多掩码输出以获得更好质量
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        # 选择得分最高的掩码
        index = np.argmax(scores)
        result_masks.append(masks[index])
    
    return np.array(result_masks)  # 返回掩码数组


def get_sam_predictor(variant: str, device: str | int) -> SamPredictor:
    """初始化指定变体的SAM预测器
    
    参数:
        variant: SAM模型变体名称，可以是"sam"、"mobilesam"、"lighthqsam"等
        device: 运行设备，可以是"cuda"、"cpu"或设备索引
    
    返回:
        SamPredictor: 初始化好的SAM预测器
    
    异常:
        NotImplementedError: 当指定了不支持的模型变体时抛出
    """
    if variant == "sam":
        # 初始化原始SAM模型
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)  # 将模型移至指定设备
        sam_predictor = SamPredictor(sam)  # 创建预测器
        return sam_predictor
    
    if variant == "mobilesam":
        # 导入并初始化MobileSAM模型
        from MobileSAM.setup_mobile_sam import setup_model
        MOBILE_SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./mobile_sam.pt")
        checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)  # 加载预训练权重
        mobile_sam = setup_model()  # 设置MobileSAM模型
        mobile_sam.load_state_dict(checkpoint, strict=True)  # 加载权重
        mobile_sam.to(device=device)  # 将模型移至指定设备
        
        sam_predictor = SamPredictor(mobile_sam)  # 创建预测器
        return sam_predictor

    elif variant == "lighthqsam":
        # 导入并初始化LightHQSAM模型
        from LightHQSAM.setup_light_hqsam import setup_model
        HQSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_hq_vit_tiny.pth")
        checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)  # 加载预训练权重
        light_hqsam = setup_model()  # 设置LightHQSAM模型
        light_hqsam.load_state_dict(checkpoint, strict=True)  # 加载权重
        light_hqsam.to(device=device)  # 将模型移至指定设备
        
        sam_predictor = SamPredictor(light_hqsam)  # 创建预测器
        return sam_predictor
        
    elif variant == "fastsam":
        raise NotImplementedError("FastSAM预测器尚未实现")
    else:
        raise NotImplementedError(f"不支持的SAM变体: {variant}")
    


# The SAM based on automatic mask generation, without bbox prompting
# def get_sam_segmentation_dense(
#     variant:str, model: Any, image: np.ndarray
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     '''
#     The SAM based on automatic mask generation, without bbox prompting
    
#     Args:
#         model: The mask generator or the YOLO model
#         image: )H, W, 3), in RGB color space, in range [0, 255]
        
#     Returns:
#         mask: (N, H, W)
#         xyxy: (N, 4)
#         conf: (N,)
#     '''
#     if variant == "sam":
#         results = model.generate(image)
#         mask = []
#         xyxy = []
#         conf = []
#         for r in results:
#             mask.append(r["segmentation"])
#             r_xyxy = r["bbox"].copy()
#             # Convert from xyhw format to xyxy format
#             r_xyxy[2] += r_xyxy[0]
#             r_xyxy[3] += r_xyxy[1]
#             xyxy.append(r_xyxy)
#             conf.append(r["predicted_iou"])
#         mask = np.array(mask)
#         xyxy = np.array(xyxy)
#         conf = np.array(conf)
#         return mask, xyxy, conf
#     elif variant == "fastsam":
#         # The arguments are directly copied from the GSA repo
#         results = model(
#             image,
#             imgsz=1024,
#             device="cuda",
#             retina_masks=True,
#             iou=0.9,
#             conf=0.4,
#             max_det=100,
#         )
#         raise NotImplementedError
#     else:
#         raise NotImplementedError
def get_sam_segmentation_dense(
    variant: str, model: Any, image: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    使用SAM进行基于自动掩码生成的密集分割，无需边界框提示
    
    参数:
        variant: SAM模型变体类型，支持'sam'、'lighthqsam'和'fastsam'
        model: 掩码生成器(SAM模型)或YOLO模型(FastSAM)
        image: 输入图像，形状为(H, W, 3)，RGB颜色空间，值范围[0, 255]
        
    返回:
        mask: 分割掩码数组，形状为(N, H, W)，N为掩码数量
        xyxy: 边界框坐标数组，形状为(N, 4)，格式为[x1, y1, x2, y2]
        conf: 置信度分数数组，形状为(N,)，表示每个掩码的置信度
    '''
    # 处理原始SAM和LightHQSAM变体
    if variant in ["sam", "lighthqsam"]:  # 同时支持原始SAM和LightHQSAM模型
        # 调用模型的generate方法生成掩码结果
        results = model.generate(image)
        mask = []  # 存储分割掩码
        xyxy = []  # 存储边界框坐标
        conf = []  # 存储置信度分数
        
        # 处理每个生成的掩码结果
        for r in results:
            mask.append(r["segmentation"])  # 获取分割掩码
            r_xyxy = r["bbox"].copy()  # 获取边界框
            # 将边界框从xyhw格式(x1, y1, width, height)转换为xyxy格式(x1, y1, x2, y2)
            r_xyxy[2] += r_xyxy[0]  # x1 + width -> x2
            r_xyxy[3] += r_xyxy[1]  # y1 + height -> y2
            xyxy.append(r_xyxy)
            conf.append(r["predicted_iou"])  # 使用预测的IOU作为置信度分数
            
        # 转换为numpy数组格式
        mask = np.array(mask)
        xyxy = np.array(xyxy)
        conf = np.array(conf)
        return mask, xyxy, conf
    
    # 处理FastSAM变体
    elif variant == "fastsam":
        # FastSAM模型推理，基于YOLO架构
        results = model(
            image,
            imgsz=1024,  # 输入图像大小
            device="cuda",  # 运行设备
            retina_masks=True,  # 启用高分辨率掩码
            iou=0.9,  # NMS的IoU阈值
            conf=0.4,  # 置信度阈值
            max_det=100,  # 最大检测数量
        )
        
        # 解析FastSAM的输出结果
        result = results[0]  # 通常只处理第一个结果
        masks = result.masks.data.cpu().numpy().astype(bool)  # 获取掩码并转换格式
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # 获取边界框坐标
        confidences = result.boxes.conf.cpu().numpy()  # 获取置信度分数
        
        return masks, boxes_xyxy, confidences
    
    # 处理不支持的模型变体
    else:
        raise NotImplementedError(f"不支持的SAM变体: {variant}")

# def get_sam_mask_generator(variant:str, device: str | int) -> SamAutomaticMaskGenerator:
#     if variant == "sam":
#         sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
#         sam.to(device)
#         mask_generator = SamAutomaticMaskGenerator(
#             model=sam,
#             points_per_side=12,
#             points_per_batch=144,
#             pred_iou_thresh=0.88,
#             stability_score_thresh=0.95,
#             crop_n_layers=0,
#             min_mask_region_area=100,
#         )
#         return mask_generator
#     elif variant == "fastsam":
#         raise NotImplementedError
#         # from ultralytics import YOLO
#         # from FastSAM.tools import *
#         # FASTSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/FastSAM-x.pt")
#         # model = YOLO(args.model_path)
#         # return model
#     else:
#         raise NotImplementedError
import os
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from EfficientSAM.MobileSAM.setup_mobile_sam import setup_model as setup_mobile_sam
from EfficientSAM.LightHQSAM.setup_light_hqsam import setup_model as setup_light_hqsam
from ultralytics import YOLO  # 用于FastSAM

# 假设已定义以下配置路径（可根据实际情况调整）
SAM_ENCODER_VERSION = "vit_h"  # 或其他SAM版本
SAM_CHECKPOINT_PATH = os.path.join(os.getenv("GSA_PATH"), "sam_vit_h_4b8939.pth")
FASTSAM_CHECKPOINT_PATH = os.path.join(os.getenv("GSA_PATH"), "EfficientSAM/FastSAM-x.pt")
MOBILESAM_CHECKPOINT_PATH = os.path.join(os.getenv("GSA_PATH"), "EfficientSAM/mobile_sam.pt")
LIGHTHQSAM_CHECKPOINT_PATH = os.path.join(os.getenv("GSA_PATH"), "sam_hq_vit_tiny.pth")


def get_sam_mask_generator(variant: str, device: str | int) -> SamAutomaticMaskGenerator:
    """初始化指定变体的SAM掩码生成器
    
    参数:
        variant: SAM模型变体名称，支持"sam"、"mobilesam"、"lighthqsam"和"fastsam"
        device: 运行设备，可以是"cuda"、"cpu"或设备索引
    
    返回:
        SamAutomaticMaskGenerator或兼容接口的掩码生成器
    
    异常:
        NotImplementedError: 当指定了不支持的模型变体时抛出
    """
    # 通用的掩码生成器参数（保持与SAM一致）
    common_kwargs = {
        "points_per_side": 12,  # 每边采样点数量，影响生成的掩码数量和密度
        "points_per_batch": 32,  # 每批处理的点数量
        "pred_iou_thresh": 0.88,  # 预测IOU阈值，用于过滤低质量掩码
        "stability_score_thresh": 0.95,  # 稳定性分数阈值
        "crop_n_layers": 0,  # 裁剪层数，0表示不裁剪
        "min_mask_region_area": 100,  # 最小掩码区域面积，过滤过小的掩码
    }

    if variant == "sam":
        # 标准SAM模型
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)  # 将模型移至指定设备
        # 创建并返回标准SAM自动掩码生成器
        return SamAutomaticMaskGenerator(model=sam, **common_kwargs)

    elif variant == "mobilesam":
        # MobileSAM模型（基于setup_mobile_sam.py）- 轻量级版本
        mobile_sam = setup_mobile_sam()  # 初始化MobileSAM模型
        # 加载预训练权重并映射到指定设备
        mobile_sam.load_state_dict(torch.load(MOBILESAM_CHECKPOINT_PATH, map_location=device))
        mobile_sam.to(device)  # 将模型移至指定设备
        mobile_sam.eval()  # 设置为评估模式
        # 创建并返回MobileSAM自动掩码生成器
        return SamAutomaticMaskGenerator(model=mobile_sam, **common_kwargs)

    elif variant == "lighthqsam":
        # Light-HQSAM模型（基于setup_light_hqsam.py）- 高质量轻量级版本
        light_hqsam = setup_light_hqsam()  # 初始化LightHQSAM模型
        # 加载预训练权重并映射到指定设备
        light_hqsam.load_state_dict(torch.load(LIGHTHQSAM_CHECKPOINT_PATH, map_location=device))
        light_hqsam.to(device)  # 将模型移至指定设备
        light_hqsam.eval()  # 设置为评估模式
        # 创建并返回LightHQSAM自动掩码生成器
        return SamAutomaticMaskGenerator(model=light_hqsam, **common_kwargs)

    elif variant == "fastsam":
        # FastSAM模型（基于YOLO接口）- 高效版本
        # 注意：FastSAM的输出格式与SAM略有差异，此处做适配处理
        class FastSAMMaskGenerator:
            """FastSAM掩码生成器适配器，将FastSAM输出转换为SAM兼容格式"""
            def __init__(self, model,** kwargs):
                self.model = model  # YOLO模型实例
                self.kwargs = kwargs  # 额外的推理参数

            def generate(self, image):
                """生成掩码并转换为SAM兼容格式
                
                参数:
                    image: 输入图像
                
                返回:
                    list[dict]: 包含掩码信息的字典列表，格式与SAM一致
                """
                # 调用FastSAM的推理接口
                results = self.model(image, **self.kwargs)
                masks = []
                # 处理每个检测结果
                for result in results:
                    # 遍历每个掩码
                    for mask in result.masks.data:
                        mask_np = mask.cpu().numpy().astype(bool)  # 转换为numpy布尔数组
                        # 构造与SAM一致的输出字段（仅保留必要字段）
                        masks.append({
                            "segmentation": mask_np,  # 分割掩码
                            "area": int(mask_np.sum()),  # 掩码面积
                            "bbox": [0, 0, mask_np.shape[1], mask_np.shape[0]],  # 简化的边界框
                            "predicted_iou": 0.9,  # 预测IOU占位符
                            "stability_score": 0.9,  # 稳定性分数占位符
                        })
                return masks

        # 初始化YOLO模型作为FastSAM
        model = YOLO(FASTSAM_CHECKPOINT_PATH)
        model.to(device)  # 将模型移至指定设备
        # 返回适配后的FastSAM掩码生成器，设置特定参数
        return FastSAMMaskGenerator(model, conf=0.4, iou=0.7)  # FastSAM专用参数

    else:
        # 不支持的变体
        raise NotImplementedError(f"不支持的SAM变体: {variant}")

def process_tag_classes(text_prompt:str, add_classes:List[str]=[], remove_classes:List[str]=[]) -> list[str]:
    """处理Tag2Text生成的文本提示，将其转换为类别列表
    
    此函数处理Tag2Text模型生成的文本标签，将逗号分隔的标签字符串转换为
    类别列表，并支持添加额外类别和移除不需要的类别。
    
    参数:
        text_prompt: Tag2Text生成的文本提示，格式为逗号分隔的标签字符串
        add_classes: 需要额外添加的类别列表
        remove_classes: 需要移除的类别关键词列表
    
    返回:
        list[str]: 处理后的类别列表
    """
    # 按逗号分割文本提示，获取初步的类别列表
    classes = text_prompt.split(',')
    # 去除每个类别的前后空白字符
    classes = [obj_class.strip() for obj_class in classes]
    # 过滤掉空字符串类别
    classes = [obj_class for obj_class in classes if obj_class != '']
    
    # 添加额外指定的类别（如果不存在）
    for c in add_classes:
        if c not in classes:
            classes.append(c)
    
    # 移除包含指定关键词的类别（不区分大小写）
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
    
    return classes  # 返回处理后的类别列表


def process_ai2thor_classes(classes: List[str], add_classes:List[str]=[], remove_classes:List[str]=[]) -> List[str]:
    """处理AI2Thor环境中的类别名称，主要是拆分驼峰命名
    
    AI2Thor环境中的物体类别通常使用驼峰命名法（如'BaseballBat'），
    此函数将它们拆分为更自然的多个单词，并进行统一处理。
    
    参数:
        classes: AI2Thor环境中的类别列表
        add_classes: 需要额外添加的类别列表
        remove_classes: 需要移除的类别关键词列表
    
    返回:
        List[str]: 处理后的类别列表
    """
    # 去重，确保每个类别只出现一次
    classes = list(set(classes))
    
    # 添加额外指定的类别
    for c in add_classes:
        classes.append(c)
        
    # 移除包含指定关键词的类别（不区分大小写）
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]

    # 特殊处理TV类名，将其改为Tv以符合后续的分割逻辑
    classes = [obj_class.replace("TV", "Tv") for obj_class in classes]
    # 使用正则表达式按大写字母分割类名，例如'BaseballBat' -> ['Baseball', 'Bat']
    classes = [re.findall('[A-Z][^A-Z]*', obj_class) for obj_class in classes]
    # 将分割后的各部分用空格连接，形成更自然的类名
    classes = [" ".join(obj_class) for obj_class in classes]
    
    return classes
    
    
def main(args: argparse.Namespace):
    """主函数，执行完整的Grounded SAM分割流程
    
    此函数是脚本的核心，它初始化各种模型，加载数据集，
    执行检测和分割，并保存结果。
    
    参数:
        args: 命令行参数命名空间，包含各种配置选项
    """
    ### 初始化Grounding DINO模型（用于开放世界物体检测） ###
    grounding_dino_model = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,  # 模型配置文件路径
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,  # 模型权重路径
        device=args.device  # 运行设备
    )

    ### 初始化SAM模型（用于精确分割） ###
    # 根据class_set参数选择不同的初始化方式
    if args.class_set == "none":
        # 当没有指定类别集时，使用自动掩码生成器进行密集分割
        mask_generator = get_sam_mask_generator(args.sam_variant, args.device)
    else:
        # 当有指定类别集时，使用预测器（需要边界框提示）
        sam_predictor = get_sam_predictor(args.sam_variant, args.device)
    
    ### 初始化CLIP模型（用于生成文本特征或类别特征） ###
    # 创建CLIP模型和图像预处理函数
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14",  # ViT-H/14骨干网络
        "laion2b_s32b_b79k"  # 在LAION-2B数据集上预训练的权重
    )
    clip_model = clip_model.to(args.device)  # 将模型移至指定设备
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")  # 获取对应的分词器
    
    # 初始化数据集
    dataset = get_dataset(
        dataconfig=args.dataset_config,  # 数据集配置
        start=args.start,  # 起始帧索引
        end=args.end,  # 结束帧索引
        stride=args.stride,  # 采样步长
        basedir=args.dataset_root,  # 数据集根目录
        sequence=args.scene_id,  # 场景ID
        desired_height=args.desired_height,  # 期望的图像高度
        desired_width=args.desired_width,  # 期望的图像宽度,
        device="cpu",
        dtype=torch.float,
    )

    # 初始化全局类别集合，用于跟踪所有处理过的类别
    global_classes = set()
    
    # 初始化YOLO-World模型（如果指定使用它作为检测器）
    if args.detector == "yolo":
        from ultralytics import YOLO
        yolo_model_w_classes = YOLO('yolov8l-world.pt')  # 加载YOLOv8l-world模型，也可选择yolov8m-world.pt等
    
    # 根据不同的class_set参数设置检测类别
    if args.class_set == "scene":
        # 从场景元数据中加载类别信息
        obj_meta_path = args.dataset_root / args.scene_id / "obj_meta.json"
        with open(obj_meta_path, "r") as f:
            obj_meta = json.load(f)  # 加载场景中的物体元数据
        # 从元数据中提取物体类别并进行处理
        classes = process_ai2thor_classes(
            [obj["objectType"] for obj in obj_meta],  # 提取所有物体类型
            add_classes=[],  # 不添加额外类别
            remove_classes=['wall', 'floor', 'room', 'ceiling']  # 移除不需要的背景类
        )
    elif args.class_set == "generic":
        # 使用预定义的通用前景类别列表
        classes = FOREGROUND_GENERIC_CLASSES
    elif args.class_set == "minimal":
        # 使用最小化的类别列表（只有一个通用类别）
        classes = FOREGROUND_MINIMAL_CLASSES
    elif args.class_set in ["tag2text", "ram"]:
        ### 初始化Tag2Text或RAM模型（用于动态生成图像的类别标签）###
        
        if args.class_set == "tag2text":
            # 类别集将由Tag2Text针对每张图像动态生成
            # 过滤掉难以定位的属性和动作类别
            delete_tag_index = []
            # 这些索引对应的是属性和动作类别
            for i in range(3012, 3429):
                delete_tag_index.append(i)

            specified_tags='None'  # 不指定特定标签，让模型自由生成
            # 加载Tag2Text模型
            tagging_model = tag2text.tag2text_caption(
                pretrained=TAG2TEXT_CHECKPOINT_PATH,  # 预训练权重路径
                image_size=384,  # 输入图像大小
                vit='swin_b',  # 使用Swin-B骨干网络
                delete_tag_index=delete_tag_index  # 需要删除的标签索引
            )
            # 设置标签阈值
            # 降低阈值以获取更多标签
            tagging_model.threshold = 0.64 
        elif args.class_set == "ram":
            # 加载RAM模型
            tagging_model = ram(
                pretrained=RAM_CHECKPOINT_PATH,  # 预训练权重路径
                image_size=384,  # 输入图像大小
                vit='swin_l'  # 使用Swin-L骨干网络
            )
            
        # 设置为评估模式并移至指定设备
        tagging_model = tagging_model.eval().to(args.device)
        
        # 初始化Tag2Text和RAM模型使用的图像预处理变换
        tagging_transform = TS.Compose([
            TS.Resize((384, 384)),  # 调整图像大小
            TS.ToTensor(),  # 转换为张量
            TS.Normalize(  # 标准化
                mean=[0.485, 0.456, 0.406],  # ImageNet均值
                std=[0.229, 0.224, 0.225]  # ImageNet标准差
            ),
        ])
        
        # 初始类别设为None，将由模型动态生成
        classes = None
    elif args.class_set == "none":
        # 当不指定类别集时，使用通用的'item'类别
        classes = ['item']
    else:
        # 未知的类别集参数
        raise ValueError("Unknown args.class_set: ", args.class_set)

    # 打印类别信息
    if args.class_set not in ["ram", "tag2text"]:
        print("There are total", len(classes), "classes to detect. ")
    elif args.class_set == "none":
        print("Skipping tagging and detection models. ")
    else:
        print(f"{args.class_set} will be used to detect classes. ")
        
    # 设置保存文件名
    save_name = f"{args.class_set}"
    # 如果使用非标准SAM变体，添加变体名称（向后兼容）
    if args.sam_variant != "sam": 
        save_name += f"_{args.sam_variant}"
    # 如果指定了实验后缀，添加到保存名中
    if args.exp_suffix:
        save_name += f"_{args.exp_suffix}"
    
    # 视频保存设置
    if args.save_video:
        # 设置视频保存路径
        video_save_path = args.dataset_root / args.scene_id / f"gsa_vis_{save_name}.mp4"
        frames = []  # 存储视频帧的列表
    
    # 主循环：遍历数据集中的所有图像
    for idx in trange(len(dataset)):
        ### 相关路径和图像加载 ###
        color_path = dataset.color_paths[idx]  # 获取当前图像路径

        color_path = Path(color_path)  # 转换为Path对象便于路径操作
        
        # 设置可视化结果保存路径
        vis_save_path = args.dataset_root / args.scene_id / f"gsa_vis_{save_name}" / color_path.name
        # 设置检测结果保存路径（使用pkl.gz格式压缩）
        detections_save_path = args.dataset_root / args.scene_id / f"gsa_detections_{save_name}" / color_path.name
        detections_save_path = detections_save_path.with_suffix(".pkl.gz")
        
        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(vis_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(detections_save_path), exist_ok=True)
        
        # OpenCV无法直接读取Path对象，需要转换为字符串
        color_path = str(color_path)
        vis_save_path = str(vis_save_path)
        detections_save_path = str(detections_save_path)
        
        # 使用OpenCV读取图像（默认读取为BGR格式）
        image = cv2.imread(color_path) 
        # 转换为RGB格式（后续处理需要）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        # 转换为PIL图像格式（方便进行变换）
        image_pil = Image.fromarray(image_rgb)
        
        ### Tag2Text/RAM 标签生成 ###
        if args.class_set in ["ram", "tag2text"]:
            # 调整图像大小并应用预处理变换
            raw_image = image_pil.resize((384, 384))
            raw_image = tagging_transform(raw_image).unsqueeze(0).to(args.device)
            
            # 使用RAM模型进行标签生成
            if args.class_set == "ram":
                res = inference_ram(raw_image, tagging_model)
                caption="NA"  # RAM模型不生成完整描述
            # 使用Tag2Text模型进行标签生成
            elif args.class_set == "tag2text":
                res = inference_tag2text.inference(raw_image, tagging_model, specified_tags)
                caption = res[2]  # 获取生成的图像描述文本

            # 目前使用","分隔符比"."更适合检测单个标签
            text_prompt = res[0].replace(' |', ',')
            
            # 添加"other item"类别来捕获不在Tag2Text生成的标签中的对象
            # 移除房间相关类别，否则可能会包含整个图像区域
            # 暂时隐藏背景类别如"wall"和"floor"
            add_classes = ["other item"]  # 需要额外添加的类别
            remove_classes = [  # 需要移除的类别列表
                "room", "kitchen", "office", "house", "home", "building", "corner",
                "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
                "apartment", "image", "city", "blue", "skylight", "hallway", 
                "bureau", "modern", "salon", "doorway", "wall lamp", "wood floor"
            ]
            bg_classes = ["wall", "floor", "ceiling"]  # 定义背景类别

            # 根据参数决定是否包含背景类别
            if args.add_bg_classes:
                add_classes += bg_classes  # 添加背景类别
            else:
                remove_classes += bg_classes  # 移除背景类别

            # 处理标签文本，得到最终的类别列表
            classes = process_tag_classes(
                text_prompt,  # 原始标签文本
                add_classes=add_classes,  # 要添加的额外类别
                remove_classes=remove_classes,  # 要移除的类别
            )
            
        # 将当前帧的类别添加到全局类别集合中
        global_classes.update(classes)
        
        if args.accumu_classes:
            # 如果启用累积类别，则使用迄今为止所有见过的类别
            classes = list(global_classes)
            
        ### 检测和分割处理 ###
        if args.class_set == "none":
            # 当不指定类别集时，直接使用SAM的密集采样模式获取分割
            mask, xyxy, conf = get_sam_segmentation_dense(
                args.sam_variant, mask_generator, image_rgb)
            # 创建detections对象，包含分割结果的所有信息
            detections = sv.Detections(
                xyxy=xyxy,  # 边界框坐标
                confidence=conf,  # 置信度分数
                class_id=np.zeros_like(conf).astype(int),  # 类别ID（默认为0，因为使用通用类别）
                mask=mask,  # 分割掩码
            )
            # 计算CLIP特征用于后续处理
            image_crops, image_feats, text_feats = compute_clip_features(
                image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, args.device)

            ### 可视化结果 ###
            # 快速可视化检测和分割结果
            annotated_image, labels = vis_result_fast(
                image, detections, classes, instance_random_color=True)  # 使用随机颜色显示不同实例
            
            # 保存可视化结果图像
            cv2.imwrite(vis_save_path, annotated_image)
        else:
            if args.detector == "dino":
                # 使用GroundingDINO进行检测，SAM进行分割
                detections = grounding_dino_model.predict_with_classes(
                    image=image,  # 注意：此函数需要BGR格式图像
                    classes=classes,  # 要检测的类别列表
                    box_threshold=args.box_threshold,  # 边界框置信度阈值
                    text_threshold=args.text_threshold,  # 文本置信度阈值
                )
            
                if len(detections.class_id) > 0:  # 如果检测到目标
                    ### 非最大抑制（NMS）###
                    # 应用NMS来消除重叠的检测框
                    nms_idx = torchvision.ops.nms(
                        torch.from_numpy(detections.xyxy),  # 转换为PyTorch张量
                        torch.from_numpy(detections.confidence),  # 置信度分数
                        args.nms_threshold  # NMS阈值
                    ).numpy().tolist()  # 转回numpy数组再转为列表

                    # 根据NMS结果过滤检测框
                    detections.xyxy = detections.xyxy[nms_idx]
                    detections.confidence = detections.confidence[nms_idx]
                    detections.class_id = detections.class_id[nms_idx]
                    
                    # 移除无效的检测结果（class_id=-1表示未分类的目标）
                    valid_idx = detections.class_id != -1
                    detections.xyxy = detections.xyxy[valid_idx]
                    detections.confidence = detections.confidence[valid_idx]
                    detections.class_id = detections.class_id[valid_idx]

                    # # Somehow some detections will have class_id=-None, remove them
                    # valid_idx = [i for i, val in enumerate(detections.class_id) if val is not None]
                    # detections.xyxy = detections.xyxy[valid_idx]
                    # detections.confidence = detections.confidence[valid_idx]
                    # detections.class_id = [detections.class_id[i] for i in valid_idx]
            elif args.detector == "yolo":
                # 使用YOLO-World进行检测
                # 设置检测类别
                yolo_model_w_classes.set_classes(classes)
                # 执行YOLO-World检测
                yolo_results_w_classes = yolo_model_w_classes.predict(color_path)

                # 保存YOLO检测原始结果
                yolo_results_w_classes[0].save(vis_save_path[:-4] + "_yolo_out.jpg")
                # 提取边界框坐标（tensor格式）
                xyxy_tensor = yolo_results_w_classes[0].boxes.xyxy 
                # 转换为numpy数组
                xyxy_np = xyxy_tensor.cpu().numpy()
                # 提取置信度分数
                confidences = yolo_results_w_classes[0].boxes.conf.cpu().numpy()
                
                # 创建detections对象
                detections = sv.Detections(
                    xyxy=xyxy_np,  # 边界框坐标
                    confidence=confidences,  # 置信度
                    class_id=yolo_results_w_classes[0].boxes.cls.cpu().numpy().astype(int),  # 类别ID
                    mask=None,  # 初始没有分割掩码，后面会通过SAM生成
                )
                
            if len(detections.class_id) > 0:  # 如果有检测到目标
                
                ### 使用SAM进行分割 ###
                # 基于检测到的边界框生成分割掩码
                detections.mask = get_sam_segmentation_from_xyxy(
                    sam_predictor=sam_predictor,  # SAM预测器
                    image=image_rgb,  # RGB格式图像
                    xyxy=detections.xyxy  # 检测到的边界框
                )

                # 计算并保存检测结果的CLIP特征  
                image_crops, image_feats, text_feats = compute_clip_features(
                    image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, args.device)
            else:
                # 如果没有检测到目标，初始化空特征列表
                image_crops, image_feats, text_feats = [], [], []
            
            ### 可视化结果 ###
            # 生成可视化结果图像和标签
            annotated_image, labels = vis_result_fast(image, detections, classes)
            
            # 保存带注释的图像
            if args.class_set in ["ram", "tag2text"] and args.use_slow_vis:
                # 如果使用标签生成模型且启用了慢可视化，添加额外的注释信息
                annotated_image_caption = vis_result_slow_caption(
                    image_rgb, detections.mask, detections.xyxy, labels, caption, text_prompt)
                Image.fromarray(annotated_image_caption).save(vis_save_path)
            else:
                # 普通保存方式
                cv2.imwrite(vis_save_path, annotated_image)
        
        # 如果需要保存视频，将当前帧添加到帧列表
        if args.save_video:
            frames.append(annotated_image)
        
        # 将检测结果转换为字典格式（所有元素都使用numpy数组）
        results = {
            "xyxy": detections.xyxy,  # 边界框坐标
            "confidence": detections.confidence,  # 置信度
            "class_id": detections.class_id,  # 类别ID
            "mask": detections.mask,  # 分割掩码
            "classes": classes,  # 类别名称列表
            "image_crops": image_crops,  # 裁剪出的图像区域
            "image_feats": image_feats,  # 图像特征
            "text_feats": text_feats,  # 文本特征
        }
        
        # 如果使用标签生成模型，添加额外的标签信息
        if args.class_set in ["ram", "tag2text"]:
            results["tagging_caption"] = caption  # 生成的图像描述
            results["tagging_text_prompt"] = text_prompt  # 提取的标签文本
        
        # 使用pickle保存检测结果
        # 使用gzip进行压缩，可以将文件大小减少约500倍
        with gzip.open(detections_save_path, "wb") as f:
            pickle.dump(results, f)
    
    # 保存全局类别集合
    # 主循环结束后，将所有检测到的类别保存到JSON文件中
    with open(args.dataset_root / args.scene_id / f"gsa_classes_{save_name}.json", "w") as f:
        json.dump(list(global_classes), f)
            
    # 保存视频结果（如果启用）
    if args.save_video:
        # 使用imageio保存收集的帧为MP4视频，帧率设为10fps
        imageio.mimsave(video_save_path, frames, fps=10)
        # 打印视频保存路径
        print(f"Video saved to {video_save_path}")
        

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)