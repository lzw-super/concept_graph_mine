#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型工具模块

此模块提供了与SAM (Segment Anything Model)和CLIP模型相关的工具函数，主要用于：
1. SAM模型的初始化与配置
2. 基于边界框的图像分割预测
3. CLIP特征提取与向量相似性计算
4. 批量和非批量特征提取的统计比较

该模块是Concept Graphs项目的核心工具组件，支持图像分割、特征提取等关键功能。
"""
import os
from conceptgraph.utils.general_utils import measure_time  # 导入时间测量工具
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator  # SAM模型相关导入
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cosine  # 用于计算余弦相似度

def get_sam_predictor(cfg) -> SamPredictor:
    """
    初始化并返回SAM (Segment Anything Model) 预测器
    
    根据配置文件选择不同的SAM变体并加载相应模型
    
    参数:
        cfg: 配置对象，包含以下必要参数：
            - sam_variant: SAM模型变体类型 ("sam", "mobilesam", "lighthqsam" 等)
            - sam_encoder_version: SAM编码器版本
            - sam_checkpoint_path: 标准SAM模型的权重路径
            - mobile_sam_path: MobileSAM模型的权重路径
            - device: 运行设备 (cuda/cpu)
    
    返回:
        SamPredictor: 配置好的SAM预测器实例
    
    异常:
        NotImplementedError: 当指定了未实现的SAM变体时抛出
    """
    if cfg.sam_variant == "sam":
        # 初始化标准SAM模型
        sam = sam_model_registry[cfg.sam_encoder_version](checkpoint=cfg.sam_checkpoint_path)
        sam.to(cfg.device)
        sam_predictor = SamPredictor(sam)
        return sam_predictor
    
    if cfg.sam_variant == "mobilesam":
        # 初始化MobileSAM模型 (轻量级版本)
        from MobileSAM.setup_mobile_sam import setup_model
        checkpoint = torch.load(cfg.mobile_sam_path)
        mobile_sam = setup_model()
        mobile_sam.load_state_dict(checkpoint, strict=True)
        mobile_sam.to(device=cfg.device)
        
        sam_predictor = SamPredictor(mobile_sam)
        return sam_predictor

    elif cfg.sam_variant == "lighthqsam":
        # 初始化LightHQSAM模型 (高质量轻量级版本)
        from LightHQSAM.setup_light_hqsam import setup_model
        # 注意：这里使用了未定义的GSA_PATH变量，可能需要在实际使用中修改
        HQSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/sam_hq_vit_tiny.pth")
        checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
        light_hqsam = setup_model()
        light_hqsam.load_state_dict(checkpoint, strict=True)
        light_hqsam.to(device=cfg.device)
        
        sam_predictor = SamPredictor(light_hqsam)
        return sam_predictor
        
    elif cfg.sam_variant == "fastsam":
        # FastSAM变体暂未实现
        raise NotImplementedError
    else:
        # 未知的SAM变体
        raise NotImplementedError

# 使用检测到的边界框批量提示SAM
def get_sam_segmentation_from_xyxy_batched(sam_predictor: SamPredictor, image: np.ndarray, xyxy_tensor: torch.Tensor) -> torch.Tensor:
    """
    批量处理基于边界框的SAM分割
    
    使用PyTorch张量形式的边界框批量提取图像分割掩码
    
    参数:
        sam_predictor: 预配置的SAM预测器实例
        image: 输入图像 (numpy数组)
        xyxy_tensor: 边界框坐标的PyTorch张量，格式为 [x_min, y_min, x_max, y_max]
    
    返回:
        torch.Tensor: 分割掩码张量，形状为 [num_boxes, H, W]
    """
    # 设置输入图像到预测器
    sam_predictor.set_image(image)
    
    # 应用SAM的变换将边界框转换为模型可接受的格式
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(xyxy_tensor, image.shape[:2])
    
    # 使用转换后的边界框进行批量预测
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,  # 不使用点提示
        point_labels=None,  # 不使用点标签
        boxes=transformed_boxes,  # 使用边界框作为提示
        multimask_output=False,  # 每个提示只返回一个掩码
    )
    
    # 移除多余的维度并返回
    return masks.squeeze()

# 使用检测到的边界框提示SAM
def get_sam_segmentation_from_xyxy(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """
    非批量处理基于边界框的SAM分割
    
    对每个边界框单独进行处理，选择置信度最高的掩码
    
    参数:
        sam_predictor: 预配置的SAM预测器实例
        image: 输入图像 (numpy数组)
        xyxy: 边界框坐标的numpy数组，格式为 [x_min, y_min, x_max, y_max]
    
    返回:
        np.ndarray: 分割掩码数组，形状为 [num_boxes, H, W]
    """
    # 设置输入图像到预测器
    sam_predictor.set_image(image)
    
    result_masks = []
    # 逐一对每个边界框进行处理
    for box in xyxy:
        # 对每个边界框生成多个掩码并选择置信度最高的一个
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True  # 为每个提示生成多个掩码变体
        )
        # 选择置信度最高的掩码
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)
    
def compute_clip_features(image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
    """
    非批量计算检测对象的CLIP特征
    
    从图像中裁剪检测到的对象区域，并提取其CLIP视觉特征和对应类别的文本特征
    
    参数:
        image: 输入图像 (numpy数组)
        detections: 包含检测结果的对象，需具有xyxy和class_id属性
        clip_model: 预加载的CLIP模型
        clip_preprocess: CLIP图像预处理函数
        clip_tokenizer: CLIP文本标记化函数
        classes: 类别名称列表，索引对应于class_id
        device: 运行设备 (cuda/cpu)
    
    返回:
        tuple: (裁剪的图像列表, 图像特征数组, 文本特征数组)
    """
    # 保存图像副本，防止原始图像被修改
    backup_image = image.copy()
    
    # 转换为PIL图像以便裁剪操作
    image = Image.fromarray(image)
    
    # 设置裁剪填充大小，默认20像素
    padding = 20  # 可根据需要调整填充量
    
    # 存储裁剪的图像和特征
    image_crops = []
    image_feats = []
    text_feats = []

    # 逐个处理每个检测结果
    for idx in range(len(detections.xyxy)):
        # 获取边界框坐标
        x_min, y_min, x_max, y_max = detections.xyxy[idx]

        # 检查并调整填充，避免超出图像边界
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        # 应用调整后的填充
        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        # 裁剪图像区域
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        
        # 对裁剪图像进行预处理并提取CLIP特征
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0).to("cuda")

        # 提取并归一化图像特征
        crop_feat = clip_model.encode_image(preprocessed_image)
        crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
        
        # 获取对应的类别文本并提取文本特征
        class_id = detections.class_id[idx]
        tokenized_text = clip_tokenizer([classes[class_id]]).to("cuda")
        text_feat = clip_model.encode_text(tokenized_text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        
        # 将特征从CUDA移至CPU并转换为numpy数组
        crop_feat = crop_feat.cpu().numpy()
        text_feat = text_feat.cpu().numpy()

        # 存储结果
        image_crops.append(cropped_image)
        image_feats.append(crop_feat)
        text_feats.append(text_feat)
        
    # 将特征列表转换为numpy矩阵
    image_feats = np.concatenate(image_feats, axis=0)
    text_feats = np.concatenate(text_feats, axis=0)

    return image_crops, image_feats, text_feats

def compute_clip_features_batched(image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
    """
    批量计算检测对象的CLIP特征
    
    优化版本，使用批量处理方式提取CLIP特征，提高计算效率
    
    参数:
        image: 输入图像 (numpy数组)
        detections: 包含检测结果的对象，需具有xyxy和class_id属性
        clip_model: 预加载的CLIP模型
        clip_preprocess: CLIP图像预处理函数
        clip_tokenizer: CLIP文本标记化函数
        classes: 类别名称列表，索引对应于class_id
        device: 运行设备 (cuda/cpu)
    
    返回:
        tuple: (裁剪的图像列表, 图像特征数组, 文本特征数组)
    """
    # 转换为PIL图像以便裁剪操作
    image = Image.fromarray(image)
    padding = 20  # 调整填充量
    
    # 存储裁剪的图像和预处理后的数据
    image_crops = []
    preprocessed_images = []
    text_tokens = []
    
    # 准备批量处理数据
    for idx in range(len(detections.xyxy)):
        # 获取边界框坐标并调整填充
        x_min, y_min, x_max, y_max = detections.xyxy[idx]
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        # 裁剪并预处理图像
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)
        preprocessed_images.append(preprocessed_image)

        # 收集类别文本
        class_id = detections.class_id[idx]
        text_tokens.append(classes[class_id])
        image_crops.append(cropped_image)
    
    # 将列表转换为批处理张量
    preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device)
    text_tokens_batch = clip_tokenizer(text_tokens).to(device)
    
    # 批量推理，避免梯度计算以提高效率
    with torch.no_grad():
        image_features = clip_model.encode_image(preprocessed_images_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        text_features = clip_model.encode_text(text_tokens_batch)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 转换为numpy格式
    image_feats = image_features.cpu().numpy()
    text_feats = text_features.cpu().numpy()

    return image_crops, image_feats, text_feats


def compute_ft_vector_closeness_statistics(unbatched, batched):
    """
    计算非批量和批量特征向量之间的相似性统计信息
    
    用于评估批量处理与非批量处理之间的特征一致性
    
    参数:
        unbatched: 非批量处理的特征向量列表
        batched: 批量处理的特征向量列表
    
    输出:
        打印多种相似性度量：平均绝对差、最大绝对差、平均相对差、余弦相似度
    """
    # 初始化存储统计信息的列表
    mad = []  # 平均绝对差 (Mean Absolute Difference)
    max_diff = []  # 最大绝对差 (Maximum Absolute Difference)
    mrd = []  # 平均相对差 (Mean Relative Difference)
    cosine_sim = []  # 余弦相似度 (Cosine Similarity)

    # 逐个比较特征向量
    for i in range(len(unbatched)):
        diff = np.abs(unbatched[i] - batched[i])
        mad.append(np.mean(diff))
        max_diff.append(np.max(diff))
        # 添加一个小值以避免除以零
        mrd.append(np.mean(diff / (np.abs(batched[i]) + 1e-8)))
        # 使用1 - 余弦距离来计算相似度
        cosine_sim.append(1 - cosine(unbatched[i].flatten(), batched[i].flatten()))

    # 将列表转换为numpy数组以便统计计算
    mad = np.array(mad)
    max_diff = np.array(max_diff)
    mrd = np.array(mrd)
    cosine_sim = np.array(cosine_sim)

    # 打印统计信息
    print(f"平均绝对差: {np.mean(mad)}")
    print(f"最大绝对差: {np.max(max_diff)}")
    print(f"平均相对差: {np.mean(mrd)}")
    print(f"平均余弦相似度: {np.mean(cosine_sim)}")
    print(f"最小余弦相似度: {np.min(cosine_sim)}")