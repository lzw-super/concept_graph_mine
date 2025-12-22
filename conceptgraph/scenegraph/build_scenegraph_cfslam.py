"""
Build a scene graph from the segment-based map and captions from LLaVA.

这个文件实现了一个从基于分割的地图(CFSLAM)和LLaVA模型生成的描述构建场景图的系统。
主要功能包括：
1. 从CFSLAM格式的地图中提取对象
2. 使用LLaVA模型为对象生成描述
3. 使用GPT模型优化对象描述
4. 计算对象间的空间关系
5. 构建完整的场景图
6. 生成场景图JSON文件
7. 提供交互式注释功能
""" 
 
# 导入必要的库
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

# 禁用PyTorch梯度计算，节省内存
# 注释掉的导入，可能是之前使用但现在不再使用的映射工具
# from mappingutils import (
#     MapObjectList,
#     compute_3d_giou_accuracte_batch,
#     compute_3d_iou_accuracte_batch,
#     compute_iou_batch,
#     compute_overlap_matrix_faiss,
#     num_points_closer_than_threshold_batch,
# )

torch.autograd.set_grad_enabled(False)  # 禁用梯度计算
hf_logging.set_verbosity_error()  # 设置Hugging Face日志级别为错误

# 导入OpenAI API
from openai import OpenAI

# # 从环境变量获取OpenAI API密钥和基础URL
# API_KEY = os.getenv("OPENAI_API_KEY") 
# BASE_URL = os.getenv("OPENAI_API_BASE")

# # 初始化OpenAI客户端
# client = OpenAI(
#     api_key=API_KEY,
#     base_url=BASE_URL
# )
 
# 定义程序参数的数据类
@dataclass
class ProgramArgs:
    # 程序运行模式，支持五种不同的操作
    mode: Literal[
        "extract-node-captions",  # 提取节点描述
        "refine-node-captions",   # 优化节点描述
        "build-scenegraph",       # 构建场景图
        "generate-scenegraph-json",  # 生成场景图JSON文件
        "annotate-scenegraph",    # 注释场景图
    ]

    # 缓存目录路径，默认值为"saved/room0"
    cachedir: str = "saved/room0"
    
    # GPT提示路径
    prompts_path: str = "prompts/gpt_prompts.json"

    # 地图文件路径
    mapfile: str = "saved/room0/map/scene_map_cfslam.pkl.gz"

    # 使用的设备
    device: str = "cuda:0"

    # 下采样体素大小
    downsample_voxel_size: float = 0.025

    # 每个对象考虑的最大检测数量
    max_detections_per_object: int = 10

    # 抑制观察次数少于此值的对象
    min_views_per_object: int = 2

    # 要注释的对象列表（默认为所有对象）
    annot_inds: Union[List[int], None] = None

    # 掩码选项：涂黑、红色轮廓或无
    masking_option: Literal["blackout", "red_outline", "none"] = "none"

def load_scene_map(args, scene_map):
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
    print(args.mapfile)  # 打印地图文件路径
    with gzip.open(Path(args.mapfile), "rb") as f:  # 打开压缩文件
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



def crop_image_pil(image: Image, x1: int, y1: int, x2: int, y2: int, padding: int = 0) -> Image:
    """
    使用PIL库裁剪图像，并可选择添加填充。

    参数:
        image: PIL图像对象
        x1, y1, x2, y2: 边界框坐标（左上角和右下角）
        padding: 边界框周围的填充像素数，默认为0

    返回:
        image_crop: 裁剪后的PIL图像

    实现来自CFSLAM仓库
    """
    # 获取图像尺寸
    image_width, image_height = image.size
    
    # 计算带填充的裁剪坐标，确保不超出图像边界
    x1 = max(0, x1 - padding)  # 上边界不小于0
    y1 = max(0, y1 - padding)  # 左边界不小于0
    x2 = min(image_width, x2 + padding)  # 右边界不大于图像宽度
    y2 = min(image_height, y2 + padding)  # 下边界不大于图像高度

    # 执行裁剪
    image_crop = image.crop((x1, y1, x2, y2))
    return image_crop


def draw_red_outline(image, mask):
    """
    在图像中的对象周围绘制红色轮廓。
    
    参数:
        image: PIL图像对象
        mask: 表示对象区域的二维numpy数组掩码
    
    返回:
        image_pil: 添加了红色轮廓的PIL图像
    """
    # 将PIL图像转换为numpy数组
    image_np = np.array(image)

    # 定义红色轮廓的RGB值
    red_outline = [255, 0, 0]

    # 在二进制掩码中查找轮廓
    # mask.astype(np.uint8) * 255: 将掩码转换为8位无符号整数并缩放到0-255范围
    # cv2.RETR_EXTERNAL: 只检索外部轮廓
    # cv2.CHAIN_APPROX_SIMPLE: 使用简单的轮廓近似方法
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在对象周围绘制红色轮廓
    # 参数-1表示绘制所有轮廓，3表示轮廓的厚度
    cv2.drawContours(image_np, contours, -1, red_outline, 3)

    # 可选：通过膨胀绘制的轮廓为对象添加额外的填充
    kernel = np.ones((5, 5), np.uint8)  # 创建5x5的膨胀核
    image_np = cv2.dilate(image_np, kernel, iterations=1)  # 执行一次膨胀操作
    
    # 将numpy数组转换回PIL图像
    image_pil = Image.fromarray(image_np)

    return image_pil


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

def blackout_nonmasked_area(image_pil, mask):
    """
    将图像中未被掩码覆盖的区域涂黑。
    
    参数:
        image_pil: PIL图像对象
        mask: 表示对象区域的二维numpy数组掩码，True表示保留区域
    
    返回:
        black_image: 处理后的PIL图像，未被掩码覆盖的区域为黑色
    """
    # 将PIL图像转换为numpy数组
    image_np = np.array(image_pil)
    
    # 创建一个与输入图像形状相同的全黑图像
    black_image = np.zeros_like(image_np)
    
    # 在掩码为True的区域，用原始图像的像素值替换黑色图像的像素值
    black_image[mask] = image_np[mask]
    
    # 将numpy数组转换回PIL图像
    black_image = Image.fromarray(black_image)
    return black_image

def plot_images_with_captions(images, captions, confidences, low_confidences, masks, savedir, idx_obj):
    """
    调试辅助函数：绘制带有标题和叠加掩码的图像，并保存到目录。
    
    通过这种方式，您可以精确检查LLaVA模型为哪个图像生成了哪个描述，以及掩码区域和置信度分数。
    
    参数:
        images: 图像列表
        captions: 描述列表
        confidences: 置信度分数列表
        low_confidences: 低置信度标记列表
        masks: 掩码列表
        savedir: 保存目录路径
        idx_obj: 对象索引，用于文件名
    """
    # 最多绘制9张图像
    n = min(9, len(images))  
    
    # 计算子图的行数和列数
    nrows = int(np.ceil(n / 3))  # 向上取整，确保所有图像都能显示
    ncols = 3 if n > 1 else 1     # 至少显示1列，最多3列
    
    # 创建图形和子图数组
    fig, axarr = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), squeeze=False)  # 调整图形大小

    # 为每张图像绘制子图
    for i in range(n):
        # 计算当前图像在网格中的位置
        row, col = divmod(i, 3)
        ax = axarr[row][col]
        
        # 显示图像
        ax.imshow(images[i])

        # 将掩码叠加到图像上
        img_array = np.array(images[i])
        if img_array.shape[:2] != masks[i].shape:
            # 如果图像和掩码形状不匹配，显示错误消息
            ax.text(0.5, 0.5, "Plotting error: Shape mismatch between image and mask", ha='center', va='center')
        else:
            # 创建绿色掩码叠加层
            green_mask = np.zeros((*masks[i].shape, 3), dtype=np.uint8)
            green_mask[masks[i]] = [0, 255, 0]  # 在掩码为True的区域设置绿色
            ax.imshow(green_mask, alpha=0.15)  # 半透明叠加

        # 设置标题文本，包含描述和置信度
        title_text = f"Caption: {captions[i]}\nConfidence: {confidences[i]:.2f}"
        if low_confidences[i]:
            title_text += "\nLow Confidence"
        
        # 将标题文本换行，每行最多30个字符
        wrapped_title = '\n'.join(wrap(title_text, 30))
        
        ax.set_title(wrapped_title, fontsize=12)  # 设置标题，使用较小的字体以适应
        ax.axis('off')  # 关闭坐标轴显示

    # 移除任何未使用的子图
    for i in range(n, nrows * ncols):
        row, col = divmod(i, 3)
        axarr[row][col].axis('off')
    
    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(savedir / f"{idx_obj}.png")
    plt.close()  # 关闭图形以释放内存



def extract_node_captions(args):
    """
    使用LLaVA模型为场景地图中的所有节点提取描述
    
    该函数从场景地图中加载对象数据，对每个对象的检测帧进行处理，并使用LLaVA模型生成对象描述。
    处理流程包括：
    1. 加载场景地图数据
    2. 初始化LLaVA多模态模型
    3. 为每个对象选择高置信度的检测帧
    4. 裁剪并预处理图像（应用掩码选项）
    5. 使用LLaVA模型生成对象描述
    6. 保存特征描述符和对象描述到文件系统
    7. 生成调试图像以可视化结果
    
    参数:
        args: ProgramArgs对象，包含配置信息如缓存目录、最大检测数等
    """
    # from conceptgraph.llava.llava_model import LLaVaChat    # old
    from conceptgraph.llava.llava_model_sub import LLaVaChat   # new 
    # NOTE: args.mapfile is in cfslam format
    from conceptgraph.slam.slam_classes import MapObjectList

    # 加载场景地图
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)
    
    # 场景地图格式为CFSLAM，包含以下关键字段
    # keys: 'image_idx', 'mask_idx', 'color_path', 'class_id', 'num_detections',
    # 'mask', 'xyxy', 'conf', 'n_points', 'pixel_area', 'contain_number', 'clip_ft',
    # 'text_ft', 'pcd_np', 'bbox_np', 'pcd_color_np'

    # 创建LLaVA聊天对象所需的命名空间参数
    chat_args = SimpleNamespace()
    chat_args.model_path = '/home/zhengwu/Desktop/concept-graphs/llava_v1.6_vicuna' # 从环境变量获取模型路径
    # chat_args.conv_mode = "v0_mmtag"  # 对话模式，可选"multimodal"
    # chat_args.num_gpus = 2  # 使用的GPU数量

    # 创建富文本控制台用于美化输出
    console = rich.console.Console()

    # 初始化LLaVA聊天模型
    chat = LLaVaChat(chat_args.model_path)
    print("LLaVA chat initialized...")
    # query = (
    #     "请尽可能详细地描述图像中的中心物体：包括颜色、形状、材质、尺寸大致描述、"
    #     "表面纹理、可见部件与结构、可能的用途/功能、当前状态（如打开/关闭、满/空、损坏/完好）、"
    #     "与附近物体或表面的关系（如在桌面上、靠着墙、悬挂在钩子上等）。"
    #     "输出为简洁自然语言，不要包含多余前后缀。"
    # )
    query = ("Please describe the central object in the image as thoroughly as possible:Include color, shape, material, and approximate size."
            "Surface texture, visible components and structure, possible purpose/function, current state (e.g., open/closed, full/empty, damaged/intact)."
            "Relationship to nearby objects or surfaces (e.g., on a table, against a wall, hanging from a hook)."
            "Output in concise, natural language without unnecessary prefixes or suffixes."
            )

    # 创建保存特征和描述的目录
    savedir_feat = Path(args.cachedir) / "cfslam_feat_llava"  # 特征保存目录
    savedir_feat.mkdir(exist_ok=True, parents=True)  # 创建目录（如果不存在）
    savedir_captions = Path(args.cachedir) / "cfslam_captions_llava"  # 描述保存目录
    savedir_captions.mkdir(exist_ok=True, parents=True)
    savedir_debug = Path(args.cachedir) / "cfslam_captions_llava_debug"  # 调试图像保存目录
    savedir_debug.mkdir(exist_ok=True, parents=True)

    # 存储所有对象描述的列表
    caption_dict_list = [] 
    
    # 遍历场景地图中的每个对象
    for idx_obj, obj in tqdm(enumerate(scene_map), total=len(scene_map)):
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
        idx_most_conf = idx_most_conf[:args.max_detections_per_object] 
        lzw = 0  # 计数变量，最多处理8个检测

        # 处理每个高置信度检测
        for idx_det in tqdm(idx_most_conf): 
            # if lzw > 7:  # 限制处理的检测数量为8个
            #     continue
                
            # 打开并转换为RGB图像
            image = Image.open(obj["color_path"][idx_det]).convert("RGB")
            xyxy = obj["xyxy"][idx_det]  # 获取边界框坐标
            class_id = obj["class_id"][idx_det]  # 获取类别ID
            mask = obj["mask"][idx_det]  # 获取对象掩码
            
            # 裁剪参数设置
            padding = 10  # 边界框周围的填充像素数
            x1, y1, x2, y2 = xyxy  # 解析边界框坐标
            
            # 裁剪图像和掩码
            image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
            image_crop.show()
            
            # 根据masking_option参数选择不同的图像预处理方式
            if args.masking_option == "blackout":
                # 将未被掩码覆盖的区域涂黑
                image_crop_modified = blackout_nonmasked_area(image_crop, mask_crop)
            elif args.masking_option == "red_outline":
                # 在对象周围绘制红色轮廓
                image_crop_modified = draw_red_outline(image_crop, mask_crop)
            else:
                # 不进行修改
                image_crop_modified = image_crop

            # 检查裁剪后的图像大小，如果太小则跳过
            _w, _h = image_crop.size
            if _w * _h < 70 * 70:
                print("small object. Skipping LLaVA captioning...")
                low_confidences.append(True)
                continue
            else:
                low_confidences.append(False)

            
            # if getattr(chat, "is_mistral", False):
            #     chat.reset()
            #     console.print("[bold red]User:[/bold red] " + query)
            #     outputs = chat(query=query, image_features=None, image_pil=image_crop_modified)
            # else:
            #     image_tensor = chat.image_processor.preprocess(image_crop_modified, return_tensors="pt")["pixel_values"][0]
            #     image_features = chat.encode_image(image_tensor[None, ...].half().cuda())
            #     features.append(image_features.detach().cpu())
            #     chat.reset()
            #     console.print("[bold red]User:[/bold red] " + query)
            #     outputs = chat(query=query, image_features=image_features)
            
            # print(outputs)
            # 移除输出中的任何额外标记
            # outputs = outputs.split("###")[0] 
            # print('after ') 
            # print(outputs)
            print('\n')
            console.print("[bold red]User:[/bold red] " + query) 
            image_sizes = [image_crop_modified.size]
            image_tensor = chat.preprocess_image([image_crop_modified]).to("cuda", dtype=torch.float16)
            outputs = chat(query=query, image_features=image_tensor, image_sizes=image_sizes).replace("<s>", "").replace("</s>", "").strip() 
            # console.print("[bold green]LLaVA:[/bold green] " + outputs) 
            
            captions.append(outputs)
        
            # 为调试保存相关信息
            conf_value = conf[idx_det]
            image_list.append(image_crop)
            caption_list.append(outputs)
            confidences_list.append(conf_value)
            low_confidences_list.append(low_confidences[-1])
            mask_list.append(mask_crop)  # 添加裁剪后的掩码
            lzw += 1
            
        # 保存当前对象的描述信息
        caption_dict_list.append(
            {
                "id": idx_obj,
                "captions": captions,
                "low_confidences": low_confidences,
            }
        )

        _features_tensor = torch.cat(features, dim=0) if len(features) > 0 else torch.empty(0)
        torch.save(_features_tensor, savedir_feat / f"{idx_obj}.pt")
        
        # 生成并保存调试图像
        if len(image_list) > 0:
            plot_images_with_captions(image_list, caption_list, confidences_list, low_confidences_list, mask_list, savedir_debug, idx_obj)
    
    # 处理生成的描述，移除不必要的前缀
    for item in caption_dict_list:
        for idx_lzw, caption in enumerate(item["captions"]):
            # 移除常见的冗余前缀
            text = "The central object in the image is "
            if text in caption:
                item["captions"][idx_lzw] = caption.replace(text, "")
                
    # 保存处理后的描述到JSON文件
    with open(Path(args.cachedir) / "cfslam_llava_captions.json", "w", encoding="utf-8") as f:
        json.dump(caption_dict_list, f, indent=4, sort_keys=False)


def save_json_to_file(json_str, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_str, f, indent=4, sort_keys=False)


def refine_node_captions(args):
    """
    使用GPT模型优化节点描述
    
    该函数从保存的LLaVA描述文件加载对象描述，结合场景地图信息，
    使用GPT模型对描述进行优化，以生成更准确、更简洁的对象标签和摘要。
    处理结果保存到指定目录中的JSON文件。
    
    参数:
        args: ProgramArgs对象，包含配置信息如缓存目录、提示文件路径等
        
    处理流程:
    1. 从指定文件加载LLaVA生成的对象描述
    2. 加载场景地图数据
    3. 加载GPT提示模板
    4. 对每个对象构建提示并调用GPT模型
    5. 保存模型响应到指定目录
    6. 记录处理成功和失败的响应数量
    """
    # NOTE: args.mapfile is in cfslam format
    from conceptgraph.slam.slam_classes import MapObjectList
    from conceptgraph.scenegraph.GPTPrompt import GPTPrompt

    # 加载每个分割对象的描述
    caption_file = Path(args.cachedir) / "cfslam_llava_captions.json"
    captions = None
    with open(caption_file, "r") as f:
        captions = json.load(f)
    # loaddir_captions = Path(args.cachedir) / "cfslam_captions_llava"
    # captions = []
    # for idx_obj in range(len(os.listdir(loaddir_captions))):
    #     with open(loaddir_captions / f"{idx_obj}.pkl", "rb") as f:
    #         captions.append(pkl.load(f))

    # 加载场景地图
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)
    
    # 加载GPT提示模板
    gpt_messages = GPTPrompt().get_json()

    TIMEOUT = 15000  # Timeout in seconds

    responses_savedir = Path(args.cachedir) / "cfslam_gpt-4_responses"
    responses_savedir.mkdir(exist_ok=True, parents=True)

    responses = []
    unsucessful_responses = 0

    # loop over every object
    for _i in trange(len(captions)):
        if len(captions[_i]) == 0:
            continue
        
        # Prepare the object prompt 
        _dict = {}
        _caption = captions[_i]
        _bbox = scene_map[_i]["bbox"]
        # _bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(scene_map[_i]["bbox"]))
        _dict["id"] = _caption["id"]
        # _dict["bbox_extent"] = np.round(_bbox.extent, 1).tolist()
        # _dict["bbox_center"] = np.round(_bbox.center, 1).tolist()
        _dict["captions"] = _caption["captions"]
        # _dict["low_confidences"] = _caption["low_confidences"]
        # Convert to printable string
        
        # Make and format the full prompt
        preds = json.dumps(_dict, indent=0)

        start_time = time.time()
    
        curr_chat_messages = gpt_messages[:]
        curr_chat_messages.append({"role": "user", "content": preds})
        chat_completion = client.chat.completions.create(
            # model="gpt-3.5-turbo",
            model= 'qwen-plus',   #"gpt-4",
            messages=curr_chat_messages,
            timeout=TIMEOUT  # Timeout in seconds
        )
        elapsed_time = time.time() - start_time
        if elapsed_time > TIMEOUT:
            print("Timed out exceeded!")
            _dict["response"] = "FAIL"
            # responses.append('{"object_tag": "FAIL"}')
            save_json_to_file(_dict, responses_savedir / f"{_caption['id']}.json")
            responses.append(json.dumps(_dict))
            unsucessful_responses += 1
            continue
        
        # count unsucessful responses
        if "invalid" in chat_completion.choices[0].message.content.strip("\n"): #chat_completion["choices"][0]["message"]["content"].strip("\n"):
            unsucessful_responses += 1
            
        # print output
        prjson([{"role": "user", "content": preds}])
        print(chat_completion.choices[0].message.content)#print(chat_completion["choices"][0]["message"]["content"])
        print(f"Unsucessful responses so far: {unsucessful_responses}")
        _dict["response"] = chat_completion.choices[0].message.content #chat_completion["choices"][0]["message"]["content"].strip("\n")
        
        # save the response
        responses.append(json.dumps(_dict))
        save_json_to_file(_dict, responses_savedir / f"{_caption['id']}.json")
        # responses.append(chat_completion["choices"][0]["message"]["content"].strip("\n"))

    # tags = []
    # for response in responses:
    #     try:
    #         parsed = json.loads(response)
    #         tags.append(parsed["object_tag"])
    #     except:
    #         tags.append("FAIL")

    # Save the responses to a text file
    # with open(Path(args.cachedir) / "gpt-3.5-turbo_responses.txt", "w") as f:
    #     for response in responses:
    #         f.write(response + "\n")
    with open(Path(args.cachedir) / "cfslam_gpt-4_responses.pkl", "wb") as f:
        pkl.dump(responses, f)


def extract_object_tag_from_json_str(json_str):
    """
    从GPT-4响应的JSON字符串中提取对象标签
    
    该函数手动解析JSON格式的字符串，定位并提取其中的object_tag字段值。
    它通过按空格分割字符串并逐词扫描的方式查找对象标签，这比使用json.loads
    更能容忍某些格式问题。
    
    参数:
        json_str: 包含对象信息的JSON字符串，通常来自GPT-4的响应
        
    返回:
        str: 提取的对象标签，如果未找到则返回空字符串
        
    实现细节:
    1. 将JSON字符串按空格分割成词列表
    2. 逐词扫描，分为三个阶段：
       - 阶段1: 寻找JSON开始的大括号
       - 阶段2: 寻找"object_tag:"字段
       - 阶段3: 提取object_tag的值直到遇到逗号或右大括号
    3. 返回提取并清理后的对象标签
    """
    # 初始化解析状态变量
    start_str_found = False  # 是否已找到JSON开始的大括号
    is_object_tag = False    # 是否已找到object_tag字段
    object_tag_complete = False  # 对象标签是否已完全提取
    object_tag = ""          # 存储提取的对象标签
    
    # 将JSON字符串按空格分割成词列表
    r = json_str.strip().split()
    
    # 逐词扫描字符串以提取对象标签
    for _idx, _r in enumerate(r):
        if not start_str_found:
            # 阶段1: 寻找JSON的开始大括号
            if _r == "{":
                start_str_found = True
                continue
            else:
                continue
        # 阶段2: 寻找object_tag字段
        if not is_object_tag:
            if _r == '"object_tag":':
                is_object_tag = True
                continue
            else:
                continue
        # 阶段3: 提取object_tag的值
        if is_object_tag and not object_tag_complete:
            # 跳过引号
            if _r == '"':
                continue
            else:
                # 如果遇到逗号或右大括号，说明标签结束
                if _r.strip() in [",", "}"]:
                    break
                # 否则将当前词添加到对象标签中
                object_tag += f" {_r}"
                continue
    
    # 返回提取的对象标签（去除多余空格）
    return object_tag.strip()


def build_scenegraph(args):
    """构建场景图，包含对象检测、过滤、关系推理和图构建
    
    功能概述：
    1. 加载场景地图和GPT-4生成的对象描述
    2. 过滤无效对象和观察次数不足的对象
    3. 计算对象间的边界框重叠度，构建加权邻接矩阵
    4. 生成最小生成树，识别连通组件
    5. 使用Qwen-plus模型推理对象间的关系
    6. 构建场景图边关系并保存结果
    
    Args:
        args: 包含配置参数的ProgramArgs对象
    """
    # 导入必要的模块
    from conceptgraph.slam.slam_classes import MapObjectList
    from conceptgraph.slam.utils import compute_overlap_matrix

    # 加载场景地图
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)

    # 设置响应目录路径
    response_dir = Path(args.cachedir) / "cfslam_gpt-4_responses"
    responses = []  # 存储模型响应
    object_tags = []  # 存储对象标签
    also_indices_to_remove = []  # 需要移除的索引（如果JSON文件不存在）
    
    # 遍历场景地图中的所有对象
    for idx in range(len(scene_map)):
        # 检查JSON文件是否存在
        if not (response_dir / f"{idx}.json").exists():
            also_indices_to_remove.append(idx)
            continue
        
        # 读取JSON文件内容
        with open(response_dir / f"{idx}.json", "r") as f:
            _d = json.load(f)
            # 尝试解析response字段
            try:
                _d["response"] = json.loads(_d["response"])
            except json.JSONDecodeError:
                # 处理解析失败的情况
                _d["response"] = {
                    'summary': f'GPT4 json reply failed: Here is the invalid response {_d["response"]}',
                    'possible_tags': ['possible_tag_json_failed'],
                    'object_tag': 'invalid'
                }
            # 添加到响应列表和对象标签列表
            responses.append(_d)
            object_tags.append(_d["response"]["object_tag"])

    # 移除标签为"fail"或"invalid"的片段
    indices_to_remove = [i for i in range(len(responses)) if object_tags[i].lower() in ["fail", "invalid"]]
    
    # 转换为集合以便去重
    indices_to_remove = set(indices_to_remove)
    
    # 移除观察次数少于最小要求的对象
    for obj_idx in range(len(scene_map)):
        conf = scene_map[obj_idx]["conf"]
        # 检查观察次数是否小于最小要求
        if len(conf) < args.min_views_per_object:
            indices_to_remove.add(obj_idx)
    
    # 转回列表形式
    indices_to_remove = list(indices_to_remove)
    # 合并并去重需要移除的索引
    indices_to_remove = list(set(indices_to_remove + also_indices_to_remove))
    
    # 获取需要保留的片段ID
    segment_ids_to_retain = [i for i in range(len(scene_map)) if i not in indices_to_remove]
    
    # 保存需要移除的索引
    with open(Path(args.cachedir) / "cfslam_scenegraph_invalid_indices.pkl", "wb") as f:
        pkl.dump(indices_to_remove, f)
    
    # 输出移除的片段数量
    print(f"Removed {len(indices_to_remove)} segments")
    
    # 根据保留的片段ID过滤响应
    responses = [resp for resp in responses if resp['id'] in segment_ids_to_retain]

    # 提取过滤后的对象标签
    object_tags = [resp['response']['object_tag'] for resp in responses]

    # 创建修剪后的场景地图和对象标签
    pruned_scene_map = []
    pruned_object_tags = []
    for _idx, segmentidx in enumerate(segment_ids_to_retain):
        pruned_scene_map.append(scene_map[segmentidx])
        pruned_object_tags.append(object_tags[_idx])
    
    # 创建新的MapObjectList对象
    scene_map = MapObjectList(pruned_scene_map)
    object_tags = pruned_object_tags
    
    # 释放内存
    del pruned_scene_map
    gc.collect()
    
    # 获取片段数量
    num_segments = len(scene_map)

    # 将响应信息添加到场景地图中
    for i in range(num_segments):
        scene_map[i]["caption_dict"] = responses[i]

    # 保存修剪后的场景地图
    if not (Path(args.cachedir) / "map").exists():
        (Path(args.cachedir) / "map").mkdir(parents=True, exist_ok=True)
    
    with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "wb") as f:
        pkl.dump(scene_map.to_serializable(), f)

    # 计算边界框重叠度
    print("Computing bounding box overlaps...")
    bbox_overlaps = compute_overlap_matrix(args, scene_map)

    # 构建基于相似度分数的加权邻接矩阵
    weights = []  # 边权重
    rows = []     # 行索引
    cols = []     # 列索引
    
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            if i == j:
                continue
            # 如果重叠度大于阈值，则添加边
            if bbox_overlaps[i, j] > 0.01:
                weights.append(bbox_overlaps[i, j])
                rows.append(i)
                cols.append(j)
                # 添加反向边（无向图）
                weights.append(bbox_overlaps[i, j])
                rows.append(j)
                cols.append(i)

    # 创建稀疏邻接矩阵
    adjacency_matrix = csr_matrix((weights, (rows, cols)), shape=(num_segments, num_segments))

    # 找到加权邻接矩阵的最小生成树
    mst = minimum_spanning_tree(adjacency_matrix)

    # 找到最小生成树中的连通组件
    _, labels = connected_components(mst)

    # 收集每个连通组件
    components = []
    _total = 0
    if len(labels) != 0:
        for label in range(labels.max() + 1):
            # 获取当前标签对应的所有索引
            indices = np.where(labels == label)[0]
            _total += len(indices.tolist())
            components.append(indices.tolist())

    # 保存连通组件
    with open(Path(args.cachedir) / "cfslam_scenegraph_components.pkl", "wb") as f:
        pkl.dump(components, f)

    # 初始化存储每个连通组件最小生成树的列表
    minimum_spanning_trees = []
    relations = []  # 存储对象关系
    
    if len(labels) != 0:
        # 遍历每个连通组件
        for label in range(labels.max() + 1):
            component_indices = np.where(labels == label)[0]
            # 提取连通组件的子图
            subgraph = adjacency_matrix[component_indices][:, component_indices]
            # 找到连通组件子图的最小生成树
            _mst = minimum_spanning_tree(subgraph)
            # 添加到列表中
            minimum_spanning_trees.append(_mst)

        TIMEOUT = 1500  # 超时时间（秒）

        # 如果关系文件不存在，则生成关系
        if not (Path(args.cachedir) / "cfslam_object_relations.json").exists():
            relation_queries = []  # 存储关系查询
            
            # 遍历每个连通组件
            for componentidx, component in enumerate(components):
                if len(component) <= 1:
                    continue
                
                # 遍历组件中最小生成树的每条边
                for u, v in zip(
                    minimum_spanning_trees[componentidx].nonzero()[0], minimum_spanning_trees[componentidx].nonzero()[1]
                ):
                    # 获取实际的片段索引
                    segmentidx1 = component[u]
                    segmentidx2 = component[v]
                    # 获取边界框
                    _bbox1 = scene_map[segmentidx1]["bbox"]
                    _bbox2 = scene_map[segmentidx2]["bbox"]

                    # 构建输入字典
                    resp1 = scene_map[segmentidx1]["caption_dict"]["response"]
                    resp2 = scene_map[segmentidx2]["caption_dict"]["response"]
                    input_dict = {
                        "object1": {
                            "id": segmentidx1,
                            "bbox_extent": np.round(_bbox1.extent, 1).tolist(),
                            "bbox_center": np.round(_bbox1.center, 1).tolist(),
                            "object_tag": object_tags[segmentidx1],
                            "summary": resp1.get("summary", ""),
                            "object_attitude": resp1.get("object_attitude", ""),
                            "possible_tags": resp1.get("possible_tags", []),
                            "possible_attitude": resp1.get("possible_attitude", []),
                        },
                        "object2": {
                            "id": segmentidx2,
                            "bbox_extent": np.round(_bbox2.extent, 1).tolist(),
                            "bbox_center": np.round(_bbox2.center, 1).tolist(),
                            "object_tag": object_tags[segmentidx2],
                            "summary": resp2.get("summary", ""),
                            "object_attitude": resp2.get("object_attitude", ""),
                            "possible_tags": resp2.get("possible_tags", []),
                            "possible_attitude": resp2.get("possible_attitude", []),
                        },
                    }
                    
                    print(f"{input_dict['object1']['object_tag']}, {input_dict['object2']['object_tag']}")

                    relation_queries.append(input_dict)

                    # 转换为JSON字符串
                    input_json_str = json.dumps(input_dict)

                    # 定义提示词模板
                    DEFAULT_PROMPT = """
                    输入是一段JSON，其中包含两个对象"object1"与"object2"的关键信息：
                    - bbox_extent: 该对象的三维包围盒尺寸
                    - bbox_center: 该对象的三维包围盒中心
                    - object_tag: 对该对象的简短标签
                    - summary/object_attitude/possible_tags/possible_attitude: 来自上一步语义总结的描述与属性

                    请仅输出一个JSON字符串（不要输出任何其他文本），包含两个键："object_relation"与"reason"。

                    任务：基于三维空间位置(bbox_center与bbox_extent)以及对象语义(summary、object_tag、object_attitude)，判断两者之间最合理的关系。
                    可选关系（严格按如下原文之一）：
                    - "a on b" / "b on a"
                    - "a in b" / "b in a"
                    - "a above b" / "b above a"
                    - "a below b" / "b below a"
                    - "a left of b" / "b left of a"
                    - "a right of b" / "b right of a"
                    - "a in front of b" / "b in front of a"
                    - "a behind b" / "b behind a"
                    - "a near b" / "b near a"
                    - "a overlapping b" / "b overlapping a"（包围盒有明显交叠）
                    - "a attached to b" / "b attached to a"
                    - "a hanging on b" / "b hanging on a"
                    - "a leaning on b" / "b leaning on a"
                    - "none of these"（若以上都不适用）

                    判定建议：
                    - 以z轴中心与高度推断"above/below"；以x轴中心推断"left/right"；以y轴中心推断"in front/behind"；
                    - 若一个对象的中心位于另一对象包围盒范围内且尺寸关系合理，可判定为"in"或"contains"（统一用"a in b"/"b in a"）；
                    - 若两者中心距离相对两者尺寸较小，判定为"near"；
                    - 若两个包围盒有明显交叠，判定为"overlapping"；
                    - 结合summary与object_attitude判断常见语义关系（如桌面上的杯子→"a on b"，挂钩上的外套→"a hanging on b"）。

                    请先给出"reason"，用一到两句阐明你为何选择该关系（引用关键数值或语义线索），再给出"object_relation"。
                    输出必须为有效JSON，仅含"reason"与"object_relation"两个字段。
                    """

                    # 记录开始时间
                    start_time = time.time()

                    # 调用Qwen-plus模型
                    chat_completion = client.chat.completions.create(
                        model="qwen-plus",
                        messages=[{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + input_json_str}],
                        timeout=TIMEOUT,  # 超时时间
                    )
                    
                    # 计算耗时
                    elapsed_time = time.time() - start_time
                    output_dict = input_dict
                    
                    # 处理超时情况
                    if elapsed_time > TIMEOUT:
                        print("Timed out exceeded!")
                        output_dict["object_relation"] = "FAIL"
                        output_dict["reason"] = "FAIL"
                    else:
                        try:
                            # 尝试解析输出为JSON
                            chat_output_json = json.loads(chat_completion.choices[0].message.content)
                            # 如果解析成功，添加到输出字典
                            output_dict["object_relation"] = chat_output_json["object_relation"]
                            output_dict["reason"] = chat_output_json["reason"]
                        except:
                            # 处理解析失败的情况
                            output_dict["object_relation"] = "FAIL"
                            output_dict["reason"] = "FAIL"
                    
                    # 添加到关系列表
                    relations.append(output_dict)

            # 保存查询JSON到文件
            print("Saving query JSON to file...")
            with open(Path(args.cachedir) / "cfslam_object_relation_queries.json", "w") as f:
                json.dump(relation_queries, f, indent=4)

            # 保存对象关系到文件
            print("Saving object relations to file...")
            with open(Path(args.cachedir) / "cfslam_object_relations.json", "w") as f:
                json.dump(relations, f, indent=4)
        else:
            # 如果关系文件已存在，直接加载
            relations = json.load(open(Path(args.cachedir) / "cfslam_object_relations.json", "r"))

    # 初始化场景图边列表
    scenegraph_edges = []

    _idx = 0
    # 遍历每个连通组件
    for componentidx, component in enumerate(components):
        if len(component) <= 1:
            continue
        # 遍历组件中最小生成树的每条边
        for u, v in zip(
            minimum_spanning_trees[componentidx].nonzero()[0], minimum_spanning_trees[componentidx].nonzero()[1]
        ):
            segmentidx1 = component[u]
            segmentidx2 = component[v]
            # 如果关系不是"none of these"，则添加到场景图边列表
            if relations[_idx]["object_relation"] not in ["none of these", "none", "no_relation"]:
                scenegraph_edges.append((segmentidx1, segmentidx2, relations[_idx]["object_relation"]))
            _idx += 1
    
    # 输出场景图信息
    print(f"Created 3D scenegraph with {num_segments} nodes and {len(scenegraph_edges)} edges")

    # 保存场景图边
    with open(Path(args.cachedir) / "cfslam_scenegraph_edges.pkl", "wb") as f:
        pkl.dump(scenegraph_edges, f)


def generate_scenegraph_json(args):
    """
    生成场景图JSON文件，汇总场景中所有对象的信息
    
    参数:
        args: 包含配置参数的对象，必须有cachedir属性指定缓存目录路径
    
    功能:
        1. 加载剪枝后的场景地图数据
        2. 提取每个对象的关键信息（ID、边界框、标签、描述等）
        3. 将这些信息组织成JSON格式并保存到文件中
    """
    # 导入地图对象列表类，用于加载场景地图数据
    from conceptgraph.slam.slam_classes import MapObjectList
    
    # 初始化场景描述列表，用于存储所有对象的信息
    scene_desc = []
    print("Generating scene graph JSON file...")

    # 加载剪枝后的场景地图
    scene_map = MapObjectList()
    # 从缓存文件中读取序列化的场景地图数据
    with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "rb") as f:
        scene_map.load_serializable(pkl.load(f))
    print(f"Loaded scene map with {len(scene_map)} objects")

    # 遍历场景地图中的每个对象
    for i, segment in enumerate(scene_map):
        # 为每个对象创建一个字典，包含其关键信息
        _d = {
            "id": segment["caption_dict"]["id"],  # 对象唯一标识符
            "bbox_extent": np.round(segment['bbox'].extent, 1).tolist(),  # 边界框大小（四舍五入到1位小数）
            "bbox_center": np.round(segment['bbox'].center, 1).tolist(),  # 边界框中心点坐标
            "possible_tags": segment["caption_dict"]["response"]["possible_tags"],  # 可能的标签列表
            "possible_attitude": segment["caption_dict"]["response"].get("possible_attitude", []),  # 可能的属性列表
            "object_tag": segment["caption_dict"]["response"]["object_tag"],  # 主要对象标签
            "object_attitude": segment["caption_dict"]["response"].get("object_attitude", ""),  # 最可能的属性
            "caption": segment["caption_dict"]["response"]["summary"],  # 对象描述摘要
        }
        # 将对象信息添加到场景描述列表中
        scene_desc.append(_d)
    
    # 将场景描述保存为JSON文件，便于后续处理和可视化
    with open(Path(args.cachedir) / "scene_graph.json", "w") as f:
        json.dump(scene_desc, f, indent=4)  # 使用缩进格式化JSON文件，便于阅读


def display_images(image_list):
    """
    在网格布局中显示多张图像
    
    参数:
        image_list: 图像对象列表，每个元素应该是可以通过matplotlib显示的图像格式（如numpy数组或PIL图像）
    
    功能:
        1. 计算合适的网格布局（行数和列数）
        2. 创建子图并在每个子图中显示一张图像
        3. 关闭所有轴标签以专注于图像内容
        4. 调整布局并显示图像
    """
    # 获取图像列表的长度
    num_images = len(image_list)
    # 设置网格的列数，这里固定为2列（可以根据需要调整）
    cols = 2  # 子图网格的列数（可以根据需要更改）
    # 计算需要的行数，使用向上取整的方法确保所有图像都能显示
    rows = (num_images + cols - 1) // cols

    # 创建指定行数和列数的子图网格，设置整体图大小
    _, axes = plt.subplots(rows, cols, figsize=(10, 5))

    # 遍历所有子图轴对象
    for i, ax in enumerate(axes.flat):
        if i < num_images:  # 如果当前索引在图像列表范围内
            img = image_list[i]  # 获取当前图像
            ax.imshow(img)  # 在子图中显示图像
            ax.axis("off")  # 关闭坐标轴，使图像显示更清晰
        else:  # 如果是多余的子图位置（当图像总数不能完全填满网格时）
            ax.axis("off")  # 同样关闭坐标轴，保持一致的外观

    # 调整子图布局，避免图像和标签重叠
    plt.tight_layout()
    # 显示所有图像
    plt.show()


def annotate_scenegraph(args):
    """
    交互式地为场景图中的对象添加注释（标签、颜色、材质等）
    
    参数:
        args: 包含配置参数的对象，必须有cachedir属性指定缓存目录路径，
              可选有annot_inds属性指定要注释的对象索引列表
    
    功能:
        1. 加载剪枝后的场景地图数据
        2. 确定要注释的对象范围
        3. 对每个对象：
           - 显示该对象最置信度的5张裁剪图像
           - 显示之前的注释（如果有）
           - 提示用户输入新的注释信息（标签、颜色、材质）
           - 保存注释
        4. 所有注释完成后保存到JSON文件
    """
    # 导入地图对象列表类，用于加载场景地图数据
    from conceptgraph.slam.slam_classes import MapObjectList

    # 加载剪枝后的场景地图
    scene_map = MapObjectList()
    with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "rb") as f:
        scene_map.load_serializable(pkl.load(f))

    # 初始化要注释的对象索引列表
    annot_inds = None
    if args.annot_inds is not None:
        annot_inds = args.annot_inds
    
    # 如果指定了要注释的对象索引，需要加载已有的注释文件
    annots = []
    if annot_inds is not None:
        annots = json.load(open(Path(args.cachedir) / "annotated_scenegraph.json", "r"))

    # 如果未指定要注释的对象索引，则注释所有对象
    if annot_inds is None:
        annot_inds = list(range(len(scene_map)))

    # 遍历要注释的每个对象
    for idx in annot_inds:
        print(f"Object {idx} out of {len(annot_inds)}...")
        obj = scene_map[idx]  # 获取当前对象

        # 检查是否存在之前的注释
        prev_annot = None
        if len(annots) >= idx + 1:
            prev_annot = annots[idx]

        # 创建新的注释字典
        annot = {}
        annot["id"] = idx  # 设置注释的对象ID

        # 获取对象的置信度数组并排序，找出最可靠的检测结果
        conf = obj["conf"]
        conf = np.array(conf)
        idx_most_conf = np.argsort(conf)[::-1]  # 按置信度从高到低排序
        print(obj.keys())  # 打印对象的所有属性键，用于调试

        # 初始化图像列表，用于存储要显示的对象图像
        imgs = []

        # 遍历置信度最高的几个检测结果
        for idx_det in idx_most_conf:
            # 加载原始彩色图像
            image = Image.open(obj["color_path"][idx_det]).convert("RGB")
            # 获取边界框坐标
            xyxy = obj["xyxy"][idx_det]
            # 获取对应的掩码
            mask = obj["mask"][idx_det]

            # 设置裁剪的填充像素数，确保对象周围有一定的上下文
            padding = 10
            x1, y1, x2, y2 = xyxy
            # 裁剪图像和掩码
            image_crop = crop_image_pil(image, x1, y1, x2, y2, padding=padding)
            mask_crop = crop_image_pil(Image.fromarray(mask), x1, y1, x2, y2, padding=padding)
            # 将掩码转换为numpy数组并添加通道维度
            mask_crop = np.array(mask_crop)[..., None]
            # 将掩码中为0的部分设置为一个小值，使得背景半透明
            mask_crop[mask_crop == 0] = 0.05
            # 将图像与掩码相乘，突出显示对象部分
            image_crop = np.array(image_crop) * mask_crop
            imgs.append(image_crop)
            # 最多显示5张图像
            if len(imgs) >= 5:
                break

        # 显示处理后的图像
        display_images(imgs)
        plt.close("all")  # 关闭所有图像窗口，避免内存泄漏

        # 提示用户为对象添加注释
        if prev_annot is not None:  # 如果有之前的注释，显示出来供参考
            print("Previous annotation:")
            print(prev_annot)
        
        # 获取用户输入的注释信息
        annot["object_tags"] = input("Enter object tags (comma-separated): ")  # 对象标签
        annot["colors"] = input("Enter colors (comma-separated): ")  # 颜色信息
        annot["materials"] = input("Enter materials (comma-separated): ")  # 材质信息

        # 更新或添加注释
        if prev_annot is not None:
            annots[idx] = annot  # 更新已有的注释
        else:
            annots.append(annot)  # 添加新的注释

        # 询问用户是否继续注释
        go_on = input("Continue? (y/n): ")
        if go_on == "n":  # 如果用户选择不继续，退出循环
            break

    # 保存所有注释到JSON文件
    with open(Path(args.cachedir) / "annotated_scenegraph.json", "w") as f:
        json.dump(annots, f, indent=4)  # 使用缩进格式化JSON文件，便于阅读


def main():
    # Process command-line args (if any)
    args = tyro.cli(ProgramArgs)
    
    # print using masking option
    print(f"args.masking_option: {args.masking_option}")

    if args.mode == "extract-node-captions":
        extract_node_captions(args)
    elif args.mode == "refine-node-captions":
        refine_node_captions(args)
    elif args.mode == "build-scenegraph":
        build_scenegraph(args)
    elif args.mode == "generate-scenegraph-json":
        generate_scenegraph_json(args)
    elif args.mode == "annotate-scenegraph":
        annotate_scenegraph(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
