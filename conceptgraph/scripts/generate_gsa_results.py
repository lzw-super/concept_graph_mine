'''"""""
# generate_gsa_results.py
# 功能：在RGB-D数据集上提取Grounded SAM (Segment Anything Model)的检测和分割结果
# 描述：该脚本用于处理姿态已知的RGB-D数据集，使用Grounded SAM模型提取场景中的物体检测、分割和特征信息
#      结果将被保存到场景文件夹下的相应目录中
"""""'''

import os
import argparse
from pathlib import Path
import re
from typing import Any, List
from PIL import Image
import cv2
import json
import imageio
import matplotlib
matplotlib.use('Agg')  # 必须在 pyplot 导入前设置！
import matplotlib.pyplot as plt
import numpy as np
import pickle
import gzip
import open_clip  # 用于加载CLIP模型

import torch
import torchvision
from torch.utils.data import Dataset
import supervision as sv  # 用于处理检测结果
from tqdm import trange  # 用于显示进度条

# 从项目中导入必要的模块
from conceptgraph.dataset.datasets_common import get_dataset  # 获取数据集
from conceptgraph.utils.vis import vis_result_fast, vis_result_slow_caption  # 可视化工具
from conceptgraph.utils.model_utils import compute_clip_features  # 计算CLIP特征
import torch.nn.functional as F  # PyTorch的函数式API

# 尝试导入Grounded SAM相关模块
try:
    from groundingdino.util.inference import Model  # Grounding DINO模型
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator  # SAM模型
except ImportError as e:
    print("Import Error: Please install Grounded Segment Anything following the instructions in README.")
    raise e

# # 设置脚本中使用的路径
# # 假设所有检查点文件都已按照原始GSA仓库的说明下载
# if "GSA_PATH" in os.environ:
#     GSA_PATH = os.environ["GSA_PATH"]  # 从环境变量获取GSA路径
# else:
#     raise ValueError("Please set the GSA_PATH environment variable to the path of the GSA repo. ")
GSA_PATH = '/home/zhengwu/Desktop/concept-graphs/Grounded-Segment-Anything'
import sys
TAG2TEXT_PATH = os.path.join(GSA_PATH, "")  # Tag2Text模块路径
EFFICIENTSAM_PATH = os.path.join(GSA_PATH, "EfficientSAM")  # EfficientSAM模块路径
sys.path.append(GSA_PATH)  # 将GSA路径添加到Python路径
sys.path.append(TAG2TEXT_PATH)  # 将Tag2Text路径添加到Python路径
sys.path.append(EFFICIENTSAM_PATH)  # 将EfficientSAM路径添加到Python路径

import torchvision.transforms as TS  # 用于图像变换
try:
    from ram.models import ram  # RAM模型
    from ram.models import tag2text  # Tag2Text模型
    from ram import inference_tag2text, inference_ram  # 推理函数
except ImportError as e:
    print("RAM sub-package not found. Please check your GSA_PATH. ")
    raise e

# 禁用PyTorch梯度计算（推理时不需要）
torch.set_grad_enabled(False)
    
# GroundingDINO配置和检查点路径
GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")

# Segment-Anything检查点
SAM_ENCODER_VERSION = "vit_h"  # SAM编码器版本
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")  # SAM检查点路径

# Tag2Text检查点
TAG2TEXT_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./tag2text_swin_14m.pth")
RAM_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./ram_swin_large_14m.pth")

# 预定义的前景类别列表
FOREGROUND_GENERIC_CLASSES = [
    "item", "furniture", "object", "electronics", "wall decoration", "door"
]

FOREGROUND_MINIMAL_CLASSES = [
    "item"
]

# 设置命令行参数解析器
def get_parser() -> argparse.ArgumentParser:
    """""
    创建并配置命令行参数解析器
    
    Returns:
        argparse.ArgumentParser: 配置好的参数解析器
    """""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", type=Path,default='yourdataset',
        help="数据集根目录路径"
    )
    parser.add_argument(
        "--dataset_config", type=str, default="yourdataconfig",
        help="数据集配置文件路径（可能需要根据运行脚本的位置进行更改）"
    )
    
    parser.add_argument("--scene_id", type=str, default="train_3",
                       help="场景ID，默认为'train_3'")
    
    parser.add_argument("--start", type=int, default=0,
                       help="处理的起始帧索引")
    parser.add_argument("--end", type=int, default=-1,
                       help="处理的结束帧索引，-1表示处理所有帧")
    parser.add_argument("--stride", type=int, default=1,
                       help="帧处理步长")

    parser.add_argument("--desired_height", type=int, default=480,
                       help="处理后图像的期望高度")
    parser.add_argument("--desired_width", type=int, default=640,
                       help="处理后图像的期望宽度")

    parser.add_argument("--box_threshold", type=float, default=0.4,
                       help="检测框置信度阈值")
    parser.add_argument("--text_threshold", type=float, default=0.4,
                       help="文本置信度阈值")
    parser.add_argument("--nms_threshold", type=float, default=0.5,
                       help="非极大值抑制阈值")

    parser.add_argument("--class_set", type=str, default="scene", 
                        choices=["scene", "generic", "minimal", "tag2text", "ram", "none"], 
                        help="使用的类别集类型。如果为'none'，则不使用标签和检测，直接使用SAM进行密集采样分割。")
    parser.add_argument("--detector", type=str, default="dino", 
                        choices=["yolo", "dino"], 
                        help="当给定类别时，使用YOLO-World还是GroundingDINO进行目标检测")
    parser.add_argument("--add_bg_classes", action="store_true", 
                        help="如果设置，则将背景类别（wall, floor, ceiling）添加到类别集中")
    parser.add_argument("--accumu_classes", action="store_true",
                        help="如果设置，类别集将在处理帧时累积")

    parser.add_argument("--sam_variant", type=str, default="sam",
                        choices=['fastsam', 'mobilesam', "lighthqsam"],
                        help="使用的SAM变体")
     
    parser.add_argument("--save_video", action="store_true",
                        help="是否保存处理结果为视频")
    
    parser.add_argument("--device", type=str, default="cuda",
                        help="运行模型的设备（'cuda'或'cpu'）")
    
    parser.add_argument("--use_slow_vis", action="store_true", 
                        help="如果设置，使用较慢但更详细的可视化。仅在使用ram/tag2text时有效。")
    
    parser.add_argument("--exp_suffix", type=str, default=None,
                        help="保存结果的文件夹后缀")
    
    return parser

# 使用检测框提示SAM进行分割
def get_sam_segmentation_from_xyxy(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """""
    使用SAM模型基于检测框生成分割掩码
    
    Args:
        sam_predictor: SAM预测器对象
        image: RGB格式的输入图像 (H, W, 3)
        xyxy: 检测框坐标数组，形状为 (N, 4)，格式为 [x1, y1, x2, y2]
        
    Returns:
        np.ndarray: 分割掩码数组，形状为 (N, H, W)
    """""
    sam_predictor.set_image(image)  # 设置输入图像
    result_masks = []  # 存储结果掩码
    for box in xyxy:  # 遍历每个检测框
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True  # 输出多个掩码选项
        )
        index = np.argmax(scores)  # 选择置信度最高的掩码
        result_masks.append(masks[index])  # 添加到结果列表
    return np.array(result_masks)  # 转换为数组返回

# 获取SAM预测器
def get_sam_predictor(variant: str, device: str | int) -> SamPredictor:
    """""
    获取指定变体的SAM预测器
    
    Args:
        variant: SAM变体名称 ("sam", "mobilesam", "lighthqsam", "fastsam")
        device: 运行设备 (str或int)
        
    Returns:
        SamPredictor: 配置好的SAM预测器
    """""
    if variant == "sam":  # 标准SAM模型
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        sam_predictor = SamPredictor(sam)
        return sam_predictor
    
    if variant == "mobilesam":  # MobileSAM模型
        from MobileSAM.setup_mobile_sam import setup_model
        MOBILE_SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./mobile_sam.pt")
        checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
        mobile_sam = setup_model()
        mobile_sam.load_state_dict(checkpoint, strict=True)
        mobile_sam.to(device=device)
        
        sam_predictor = SamPredictor(mobile_sam)
        return sam_predictor
 
    elif variant == "lighthqsam":  # LightHQSAM模型
        from LightHQSAM.setup_light_hqsam import setup_model
        HQSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_hq_vit_tiny.pth")
        checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
        light_hqsam = setup_model()
        light_hqsam.load_state_dict(checkpoint, strict=True)
        light_hqsam.to(device=device)
        
        sam_predictor = SamPredictor(light_hqsam)
        return sam_predictor
        
    elif variant == "fastsam":  # FastSAM模型（未实现）
        raise NotImplementedError
    else:
        raise NotImplementedError

# 使用SAM进行密集分割（不使用检测框提示）
def get_sam_segmentation_dense(
    variant: str, model: Any, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """""
    使用SAM基于自动掩码生成进行密集分割，无需检测框提示
    
    Args:
        variant: SAM变体名称
        model: 掩码生成器或YOLO模型
        image: RGB格式的输入图像 (H, W, 3)，范围[0, 255]
        
    Returns:
        mask: 分割掩码数组，形状为 (N, H, W)
        xyxy: 边界框坐标数组，形状为 (N, 4)
        conf: 置信度数组，形状为 (N,)
    """""
    if variant in ["sam", "lighthqsam"]:  # 支持标准SAM和LightHQSAM
        results = model.generate(image)  # 生成掩码
        mask = []  # 存储掩码
        xyxy = []  # 存储边界框
        conf = []  # 存储置信度
        for r in results:
            mask.append(r["segmentation"])  # 添加分割掩码
            r_xyxy = r["bbox"].copy()  # 获取边界框
            # 转换从xyhw格式到xyxy格式
            r_xyxy[2] += r_xyxy[0]  # x1 + w -> x2
            r_xyxy[3] += r_xyxy[1]  # y1 + h -> y2
            xyxy.append(r_xyxy)
            conf.append(r["predicted_iou"])  # 添加预测的IOU作为置信度
        mask = np.array(mask)
        xyxy = np.array(xyxy)
        conf = np.array(conf)
        return mask, xyxy, conf
    
    elif variant == "fastsam":  # FastSAM处理逻辑
        results = model(
            image,
            imgsz=1024,
            device="cuda",
            retina_masks=True,
            iou=0.9,
            conf=0.4,
            max_det=100,
        )
        # 解析FastSAM结果
        result = results[0]
        masks = result.masks.data.cpu().numpy().astype(bool)
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        return masks, boxes_xyxy, confidences
    
    else:
        raise NotImplementedError(f"Unsupported variant: {variant}")

# 导入额外的模块用于支持不同的SAM变体
import os
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from EfficientSAM.MobileSAM.setup_mobile_sam import setup_model as setup_mobile_sam
from EfficientSAM.LightHQSAM.setup_light_hqsam import setup_model as setup_light_hqsam
from ultralytics import YOLO  # 用于FastSAM

# 定义模型配置路径
SAM_ENCODER_VERSION = "vit_h"  # SAM编码器版本
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "sam_vit_h_4b8939.pth")
FASTSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "FastSAM-x.pt")
MOBILESAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "mobile_sam.pt")
LIGHTHQSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "sam_hq_vit_tiny.pth")

# 获取SAM掩码生成器
def get_sam_mask_generator(variant: str, device: str | int) -> SamAutomaticMaskGenerator:
    """""
    获取指定变体的SAM掩码生成器
    
    Args:
        variant: SAM变体名称 ("sam", "mobilesam", "lighthqsam", "fastsam")
        device: 运行设备 (str或int)
        
    Returns:
        SamAutomaticMaskGenerator: 配置好的掩码生成器
    """""
    # 通用的掩码生成器参数
    common_kwargs = {
        "points_per_side": 12,  # 每边采样点数量
        "points_per_batch": 32,  # 每批处理的点数量
        "pred_iou_thresh": 0.88,  # 预测IOU阈值
        "stability_score_thresh": 0.95,  # 稳定性分数阈值
        "crop_n_layers": 0,  # 裁剪层数
        "min_mask_region_area": 100,  # 最小掩码区域面积
    }

    if variant == "sam":  # 标准SAM模型
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        return SamAutomaticMaskGenerator(model=sam, **common_kwargs)

    elif variant == "mobilesam":  # MobileSAM模型
        mobile_sam = setup_mobile_sam()
        mobile_sam.load_state_dict(torch.load(MOBILESAM_CHECKPOINT_PATH, map_location=device))
        mobile_sam.to(device)
        mobile_sam.eval()
        return SamAutomaticMaskGenerator(model=mobile_sam,** common_kwargs)

    elif variant == "lighthqsam":  # Light-HQSAM模型
        light_hqsam = setup_light_hqsam()
        light_hqsam.load_state_dict(torch.load(LIGHTHQSAM_CHECKPOINT_PATH, map_location=device))
        light_hqsam.to(device)
        light_hqsam.eval()
        return SamAutomaticMaskGenerator(model=light_hqsam, **common_kwargs)

    elif variant == "fastsam":  # FastSAM模型
        # 定义一个适配器类使FastSAM输出与SAM兼容
        class FastSAMMaskGenerator:
            def __init__(self, model,** kwargs):
                self.model = model
                self.kwargs = kwargs

            def generate(self, image):
                # 调用FastSAM推理接口并转换输出格式
                results = self.model(image, **self.kwargs)
                masks = []
                for result in results:
                    for mask in result.masks.data:
                        mask_np = mask.cpu().numpy().astype(bool)
                        # 构造与SAM一致的输出字段（仅保留必要字段）
                        masks.append({
                            "segmentation": mask_np,
                            "area": int(mask_np.sum()),
                            "bbox": [0, 0, mask_np.shape[1], mask_np.shape[0]],  # 简化处理
                            "predicted_iou": 0.9,  # 占位符
                            "stability_score": 0.9,  # 占位符
                        })
                return masks

        model = YOLO(FASTSAM_CHECKPOINT_PATH)
        model.to(device)
        return FastSAMMaskGenerator(model, conf=0.4, iou=0.7)  # FastSAM专用参数

    else:
        raise NotImplementedError(f"不支持的SAM变体: {variant}")

# 处理Tag2Text生成的类别
def process_tag_classes(text_prompt:str, add_classes:List[str]=[], remove_classes:List[str]=[]) -> list[str]:
    """""
    将Tag2Text生成的文本提示转换为类别列表
    
    Args:
        text_prompt: Tag2Text生成的文本提示
        add_classes: 要添加的额外类别列表
        remove_classes: 要移除的类别列表
        
    Returns:
        list[str]: 处理后的类别列表
    """""
    classes = text_prompt.split(',')  # 按逗号分割
    classes = [obj_class.strip() for obj_class in classes]  # 去除首尾空格
    classes = [obj_class for obj_class in classes if obj_class != '']  # 移除空字符串
    
    # 添加额外类别
    for c in add_classes:
        if c not in classes:
            classes.append(c)
    
    # 移除不需要的类别
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
    
    return classes

# 处理AI2Thor数据集的类别
def process_ai2thor_classes(classes: List[str], add_classes:List[str]=[], remove_classes:List[str]=[]) -> List[str]:
    """""
    对AI2Thor场景中的objectTypes进行预处理
    
    Args:
        classes: AI2Thor对象类型列表
        add_classes: 要添加的额外类别
        remove_classes: 要移除的类别
        
    Returns:
        List[str]: 处理后的类别列表
    """""
    classes = list(set(classes))  # 去重
    
    # 添加额外类别
    for c in add_classes:
        classes.append(c)
        
    # 移除不需要的类别
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]

    # 处理驼峰命名法的类别，将其拆分为单词
    classes = [obj_class.replace("TV", "Tv") for obj_class in classes]  # 特殊处理TV
    classes = [re.findall('[A-Z][^A-Z]*', obj_class) for obj_class in classes]  # 按大写字母分割
    classes = [" ".join(obj_class) for obj_class in classes]  # 用空格连接
    
    return classes
    
# 主函数
def main(args: argparse.Namespace):
    """""
    主函数，处理数据集并生成Grounded SAM结果
    
    Args:
        args: 命令行参数命名空间
    """""
    ### 初始化Grounding DINO模型 ### 
    grounding_dino_model = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH, 
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, 
        device=args.device
    )

    ### 初始化SAM模型 ###
    if args.class_set == "none":  # 无类别集时使用掩码生成器
        mask_generator = get_sam_mask_generator(args.sam_variant, args.device)
    else:  # 有类别集时使用预测器
        sam_predictor = get_sam_predictor(args.sam_variant, args.device)
    
    ### 初始化CLIP模型 ###
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"  # 使用ViT-H-14模型和laion2b_s32b_b79k数据集
    )
    clip_model = clip_model.to(args.device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    
    # 初始化数据集
    dataset = get_dataset(
        dataconfig=args.dataset_config,
        start=args.start,
        end=args.end,
        stride=args.stride,
        basedir=args.dataset_root,
        sequence=args.scene_id,
        desired_height=args.desired_height,
        desired_width=args.desired_width,
        device="cpu",
        dtype=torch.float,
    )

    global_classes = set()  # 存储所有帧中出现的类别
    
    # 初始化YOLO-World模型（如果使用）
    if args.detector == "yolo":
        from ultralytics import YOLO
        yolo_model_w_classes = YOLO('yolov8l-world.pt')  # 加载YOLO-World模型
    
    # 根据类别集类型初始化类别
    if args.class_set == "scene":  # 使用场景特定类别
        # 加载对象元信息
        obj_meta_path = args.dataset_root / args.scene_id / "obj_meta.json"
        with open(obj_meta_path, "r") as f:
            obj_meta = json.load(f)
        # 获取场景中的对象类别列表
        classes = process_ai2thor_classes(
            [obj["objectType"] for obj in obj_meta],
            add_classes=[],
            remove_classes=['wall', 'floor', 'room', 'ceiling']  # 移除背景类
        )
    elif args.class_set == "generic":  # 使用通用类别
        classes = FOREGROUND_GENERIC_CLASSES
    elif args.class_set == "minimal":  # 使用最小类别集
        classes = FOREGROUND_MINIMAL_CLASSES
    elif args.class_set in ["tag2text", "ram"]:  # 使用自动标签生成
        ### 初始化Tag2Text或RAM模型 ###
        
        if args.class_set == "tag2text":
            # 过滤掉难以定位的属性和动作类别
            delete_tag_index = []
            for i in range(3012, 3429):
                delete_tag_index.append(i)

            specified_tags='None'
            # 加载Tag2Text模型
            tagging_model = tag2text.tag2text_caption(pretrained=TAG2TEXT_CHECKPOINT_PATH,
                                                    image_size=384,
                                                    vit='swin_b',
                                                    delete_tag_index=delete_tag_index)
            # 降低阈值以获得更多标签
            tagging_model.threshold = 0.64 
        elif args.class_set == "ram":
            # 加载RAM模型
            tagging_model = ram(pretrained=RAM_CHECKPOINT_PATH,
                                         image_size=384,
                                         vit='swin_l')
            
        tagging_model = tagging_model.eval().to(args.device)
        
        # 初始化图像变换
        tagging_transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(), 
            TS.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        
        classes = None  # 将在每个图像上动态生成
    elif args.class_set == "none":  # 无类别集
        classes = ['item']
    else:
        raise ValueError("Unknown args.class_set: ", args.class_set)

    if args.class_set not in ["ram", "tag2text"]:
        print("There are total", len(classes), "classes to detect. ")
    elif args.class_set == "none":
        print("Skipping tagging and detection models. ")
    else:
        print(f"{args.class_set} will be used to detect classes. ")
        
    # 设置保存文件名
    save_name = f"{args.class_set}"
    if args.sam_variant != "sam":  # 向后兼容
        save_name += f"_{args.sam_variant}"
    if args.exp_suffix:
        save_name += f"_{args.exp_suffix}" 
    save_name +=f"_{args.box_threshold}_{args.text_threshold}"
    
    # 初始化视频保存相关变量
    if args.save_video:
        video_save_path = args.dataset_root / args.scene_id / f"gsa_vis_{save_name}.mp4"
        frames = []
    
    for idx in trange(len(dataset)):
        ### 相关路径和加载图像 ###
        color_path = dataset.color_paths[idx]  # 获取彩色图像路径

        color_path = Path(color_path)
        
        # 设置保存路径
        vis_save_path = args.dataset_root / args.scene_id / f"gsa_vis_{save_name}" / color_path.name
        detections_save_path = args.dataset_root / args.scene_id / f"gsa_detections_{save_name}" / color_path.name
        detections_save_path = detections_save_path.with_suffix(".pkl.gz")  # 使用gzip压缩
        
        # 创建保存目录
        os.makedirs(os.path.dirname(vis_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(detections_save_path), exist_ok=True)
        
        # 将Path对象转换为字符串（OpenCV不能直接读取Path对象）
        color_path = str(color_path)
        vis_save_path = str(vis_save_path)
        detections_save_path = str(detections_save_path)
        
        # 读取图像
        image = cv2.imread(color_path)  # OpenCV读取的是BGR格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        image_pil = Image.fromarray(image_rgb)  # 转换为PIL图像
        
        ### Tag2Text或RAM处理 ###
        if args.class_set in ["ram", "tag2text"]:
            # 预处理图像用于标签生成
            raw_image = image_pil.resize((384, 384))
            raw_image = tagging_transform(raw_image).unsqueeze(0).float().to(args.device)
            
            # 执行标签生成
            if args.class_set == "ram":
                res = inference_ram(raw_image, tagging_model)
                caption="NA"
            elif args.class_set == "tag2text":
                res = inference_tag2text.inference(raw_image, tagging_model, specified_tags)
                caption=res[2]

            # 将标签分隔符从' |' 替换为 ','
            text_prompt=res[0].replace(' |', ',') 
            print(f'第{idx+1}张图出现的类: {text_prompt}')
            
            # 添加"other item"以捕获不在tag2text描述中的对象
            # 移除"xxx room"等，否则会包含整个图像
            # 暂时隐藏"wall"和"floor"
            add_classes = ["other item"]
            remove_classes = [
                "room", "kitchen", "office", "house", "home", "building", "corner",
                "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
                "apartment", "image", "city", "blue", "skylight", "hallway", 
                "bureau", "modern", "salon", "doorway", "wall lamp", "wood floor"
            ]
            bg_classes = ["wall", "floor", "ceiling"]  # 背景类别

            # 根据参数决定是否添加背景类别
            if args.add_bg_classes:
                add_classes += bg_classes
            else:
                remove_classes += bg_classes

            # 处理标签为类别列表
            classes = process_tag_classes(
                text_prompt, 
                add_classes = add_classes,
                remove_classes = remove_classes,
            )
            
        # 将当前帧的类别添加到全局类别集
        global_classes.update(classes)
        
        # 如果设置了累积类别，则使用所有见过的类别
        if args.accumu_classes:
            classes = list(global_classes)
            
        ### 检测和分割 ###
        if args.class_set == "none":  # 无类别集模式
            # 直接使用SAM进行密集采样分割
            mask, xyxy, conf = get_sam_segmentation_dense(
                args.sam_variant, mask_generator, image_rgb)
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros_like(conf).astype(int),  # 所有对象标记为同一类
                mask=mask,
            )
            # 计算CLIP特征
            image_crops, image_feats, text_feats = compute_clip_features(
                image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, args.device)

            ### 可视化结果 ###
            annotated_image, labels = vis_result_fast(
                image, detections, classes, instance_random_color=True)
            
            cv2.imwrite(vis_save_path, annotated_image)  # 保存可视化结果
        else:  # 有类别集模式
            if args.detector == "dino":  # 使用Grounding DINO检测
                # 使用GroundingDINO进行检测，SAM进行分割
                detections = grounding_dino_model.predict_with_classes(
                    image=image,  # 该函数期望BGR格式图像
                    classes=classes,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                )
            
                if len(detections.class_id) > 0:  # 如果有检测结果
                    ### 非极大值抑制 ###
                    nms_idx = torchvision.ops.nms(
                        torch.from_numpy(detections.xyxy), 
                        torch.from_numpy(detections.confidence), 
                        args.nms_threshold
                    ).numpy().tolist()

                    # 应用NMS结果
                    detections.xyxy = detections.xyxy[nms_idx]
                    detections.confidence = detections.confidence[nms_idx]
                    detections.class_id = detections.class_id[nms_idx]
                    
                    # 移除class_id为-1的检测结果
                    valid_idx = detections.class_id != -1
                    detections.xyxy = detections.xyxy[valid_idx]
                    detections.confidence = detections.confidence[valid_idx]
                    detections.class_id = detections.class_id[valid_idx]

            elif args.detector == "yolo":  # 使用YOLO-World检测
                # 设置YOLO类别
                yolo_model_w_classes.set_classes(classes)
                yolo_results_w_classes = yolo_model_w_classes.predict(color_path)

                # 保存YOLO检测结果
                yolo_results_w_classes[0].save(vis_save_path[:-4] + "_yolo_out.jpg")
                # 提取检测框和置信度
                xyxy_tensor = yolo_results_w_classes[0].boxes.xyxy 
                xyxy_np = xyxy_tensor.cpu().numpy()
                confidences = yolo_results_w_classes[0].boxes.conf.cpu().numpy()
                
                # 创建检测结果对象
                detections = sv.Detections(
                    xyxy=xyxy_np,
                    confidence=confidences,
                    class_id=yolo_results_w_classes[0].boxes.cls.cpu().numpy().astype(int),
                    mask=None,
                )
                
            if len(detections.class_id) > 0:  # 如果有检测结果
                # 使用SAM进行分割
                detections.mask = get_sam_segmentation_from_xyxy(
                    sam_predictor=sam_predictor,
                    image=image_rgb,
                    xyxy=detections.xyxy
                )

                # 计算并保存检测结果的CLIP特征
                image_crops, image_feats, text_feats = compute_clip_features(
                    image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, args.device)
            else:
                image_crops, image_feats, text_feats = [], [], []  # 无检测结果时初始化为空列表
            
            # 可视化结果
            annotated_image, labels = vis_result_fast(image, detections, classes)
            
            # 保存可视化结果
            if args.class_set in ["ram", "tag2text"] and args.use_slow_vis:
                # 使用包含标题的详细可视化
                annotated_image_caption = vis_result_slow_caption(
                    image_rgb, detections.mask, detections.xyxy, labels, caption, text_prompt)
                Image.fromarray(annotated_image_caption).save(vis_save_path)
            else:
                cv2.imwrite(vis_save_path, annotated_image)
        
        # 如果需要保存视频，添加当前帧
        if args.save_video:
            frames.append(annotated_image)
        
        # 将检测结果转换为字典格式
        results = {
            "xyxy": detections.xyxy,
            "confidence": detections.confidence,
            "class_id": detections.class_id,
            "mask": detections.mask,
            "classes": classes,
            "image_crops": image_crops,
            "image_feats": image_feats,
            "text_feats": text_feats,
        }
        
        # 如果使用tag2text或ram，添加额外信息
        if args.class_set in ["ram", "tag2text"]:
            results["tagging_caption"] = caption
            results["tagging_text_prompt"] = text_prompt
        
        # 使用pickle保存检测结果，并用gzip压缩
        with gzip.open(detections_save_path, "wb") as f:
            pickle.dump(results, f)
    
    # 保存全局类别列表
    with open(args.dataset_root / args.scene_id / f"gsa_classes_{save_name}.json", "w") as f:
        json.dump(list(global_classes), f)
            
    # 保存视频（如果需要）
    if args.save_video:
        imageio.mimsave(video_save_path, frames, fps=10)
        print(f"Video saved to {video_save_path}")
        
# 程序入口
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args() 
    args.dataset_root = Path('/home/zhengwu/Desktop/concept-graphs/Datasets/Replica' )
    args.dataset_config = '/home/zhengwu/Desktop/concept-graphs/conceptgraph/dataset/dataconfigs/replica/replica.yaml'
    args.scene_id = 'room0' 
    args.class_set = 'ram'
    args.sam_variant = 'mobilesam'

    main(args)