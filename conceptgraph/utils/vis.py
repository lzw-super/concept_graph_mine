#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化工具模块

此模块提供了丰富的2D和3D可视化工具，主要功能包括：
1. 3D场景实时渲染与可视化
2. 点云、边界框、轨迹的绘制
3. 相机视锥体生成与显示
4. 图像分割掩码和检测框的标注
5. 视频生成与保存

该模块是Concept Graphs项目的核心可视化组件，支持从2D图像分析到3D场景建模的全流程可视化需求。
"""
import copy
from typing import Iterable
import dataclasses
from PIL import Image
import cv2

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 必须在 pyplot 导入前设置！
import matplotlib.pyplot as plt
import torch
import open3d as o3d

import supervision as sv
from supervision.draw.color import Color, ColorPalette
from conceptgraph.slam.slam_classes import MapObjectList

class OnlineObjectRenderer():
    """
    在线3D对象渲染器
    
    Open3D可视化代码的模块化重构版本，用于实时渲染3D场景中的对象、轨迹和相机姿态。
    支持基于点云、边界框的可视化，以及轨迹跟踪和实时更新。
    """
    def __init__(
        self, 
        view_param: str | dict,  # 视图参数，可以是配置文件路径或参数字典
        base_objects: MapObjectList | None = None,  # 基础对象列表（点云和边界框）
        gray_map: bool = False  # 是否使用灰度地图
    ) -> None:
        # 如果提供了基础对象，将其可视化
        if base_objects is not None:
            self.n_base_objects = len(base_objects)

            # 深拷贝基础对象的点云和边界框用于可视化
            base_pcds_vis = copy.deepcopy(base_objects.get_values("pcd"))
            base_bboxes_vis = copy.deepcopy(base_objects.get_values("bbox"))
            # 对点云进行体素下采样以提高渲染性能
            for i in range(self.n_base_objects):
                base_pcds_vis[i] = base_pcds_vis[i].voxel_down_sample(voxel_size=0.08)
                if gray_map:
                    base_pcds_vis[i].paint_uniform_color([0.5, 0.5, 0.5])  # 设置为灰色
            # 设置边界框颜色
            for i in range(self.n_base_objects):
                base_bboxes_vis[i].color = [0.5, 0.5, 0.5]
            
            self.base_pcds_vis = base_pcds_vis
            self.base_bboxes_vis = base_bboxes_vis
        else:
            self.n_base_objects = 0
        
        # 初始化轨迹存储列表
        self.est_traj = []  # 估计轨迹
        self.gt_traj = []  # 真实轨迹
        
        # 使用turbo色彩映射
        self.cmap = matplotlib.colormaps.get_cmap("turbo")

        # 加载相机参数
        if isinstance(view_param, str):
            self.view_param = o3d.io.read_pinhole_camera_parameters(view_param)
        else:
            self.view_param = view_param
            
        # 设置窗口尺寸
        self.window_height = self.view_param.intrinsic.height
        self.window_width = self.view_param.intrinsic.width
        
        # 创建可视化器
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            width = self.window_width,
            height = self.window_height,
        )
        
        # 获取视图控制器并设置参数
        self.vis_ctrl = self.vis.get_view_control()
        self.vis_ctrl.convert_from_pinhole_camera_parameters(self.view_param)
        
    def filter_base_by_mask(self, mask: Iterable[bool]):
        """
        根据掩码过滤基础对象
        
        参数:
            mask: 布尔值迭代器，指定哪些基础对象应保留
        """
        assert len(mask) == self.n_base_objects
        self.base_pcds_vis = [pcd for pcd, m in zip(self.base_pcds_vis, mask) if m]
        self.base_bboxes_vis = [bbox for bbox, m in zip(self.base_bboxes_vis, mask) if m]
        self.n_base_objects = len(self.base_pcds_vis)
    
    def step(
        self,
        image: Image.Image,  # 当前帧图像
        pcds: list[o3d.geometry.PointCloud] | None = None,  # 要可视化的点云列表
        pcd_colors: np.ndarray | None = None,  # 点云颜色数组
        est_pose: np.ndarray | None = None,  # 估计的相机位姿
        gt_pose: np.ndarray | None = None,  # 真实的相机位姿
        base_objects_color: dict | None = None,  # 基础对象的颜色字典
        new_objects: MapObjectList = None,  # 新检测到的对象列表
        paint_new_objects: bool = True,  # 是否为新对象设置特殊颜色
        return_vis_handle: bool = False,  # 是否返回可视化句柄
    ):
        """
        执行一次渲染步骤，更新可视化场景
        
        参数:
            如上所示
            
        返回:
            tuple: (渲染图像, 可视化句柄或None)
        """
        # 清除所有几何体
        self.vis.clear_geometries()
        
        # 添加估计的相机位姿和轨迹
        if est_pose is not None:
            self.est_traj.append(est_pose)
            # 创建相机视锥体
            est_camera_frustum = better_camera_frustum(
                est_pose, image.height, image.width, scale=0.5, color=[1., 0, 0]  # 红色
            )
            self.vis.add_geometry(est_camera_frustum)
            # 绘制轨迹线
            if len(self.est_traj) > 1:
                est_traj_lineset = poses2lineset(np.stack(self.est_traj), color=[1., 0, 0])
                self.vis.add_geometry(est_traj_lineset)
            
        # 添加真实的相机位姿和轨迹
        if gt_pose is not None:
            self.gt_traj.append(gt_pose)
            gt_camera_frustum = better_camera_frustum(
                gt_pose, image.height, image.width, scale=0.5, color=[0, 1., 0]  # 绿色
            )
            self.vis.add_geometry(gt_camera_frustum)
            if len(self.gt_traj) > 1:
                gt_traj_lineset = poses2lineset(np.stack(self.gt_traj), color=[0, 1., 0])
                self.vis.add_geometry(gt_traj_lineset)
    
        # 添加基础对象
        if self.n_base_objects > 0:
            # 如果提供了颜色字典，使用指定颜色
            if base_objects_color is not None:
                for obj_id in range(self.n_base_objects):
                    color = base_objects_color[obj_id]
                    self.base_pcds_vis[obj_id].paint_uniform_color(color)
                    self.base_bboxes_vis[obj_id].color = color
            
            # 添加所有基础几何体
            for geom in self.base_pcds_vis + self.base_bboxes_vis:
                self.vis.add_geometry(geom)
            
        # 显示额外的点云
        if pcds is not None:
            for i in range(len(pcds)):
                # 应用位姿变换
                pcds[i].transform(est_pose)
                # 设置颜色
                if pcd_colors is not None:
                    pcds[i].paint_uniform_color(pcd_colors[i][:3])
                self.vis.add_geometry(pcds[i])
            
        # 显示新检测到的对象
        if new_objects is not None:
            for obj in new_objects:
                pcd = copy.deepcopy(obj['pcd'])
                bbox = copy.deepcopy(obj['bbox'])
                # 默认蓝色
                bbox.color = [0.0, 0.0, 1.0]
                if paint_new_objects:
                    # 如果需要特殊标记，设置为绿色
                    pcd.paint_uniform_color([0.0, 1.0, 0.0])
                    bbox.color = [0.0, 1.0, 0.0]
                
                self.vis.add_geometry(pcd)
                self.vis.add_geometry(bbox)
        
        # 应用视图参数
        self.vis_ctrl.convert_from_pinhole_camera_parameters(self.view_param)
        
        # 更新事件和渲染器
        self.vis.poll_events()
        self.vis.update_renderer()
        
        # 捕获渲染图像
        rendered_image = self.vis.capture_screen_float_buffer(False)
        rendered_image = np.asarray(rendered_image)
        
        # 根据需要返回可视化句柄
        if return_vis_handle:
            return rendered_image, self.vis
        else:
            return rendered_image, None

def get_random_colors(num_colors):
    """
    生成随机颜色用于可视化
    
    参数:
        num_colors (int): 要生成的颜色数量
        
    返回:
        colors (np.ndarray): (num_colors, 3) 颜色数组，RGB格式，范围[0, 1]
    """
    colors = []
    for i in range(num_colors):
        colors.append(np.random.rand(3))
    colors = np.array(colors)
    return colors

def show_mask(mask, ax, random_color=False):
    """
    在matplotlib轴上显示分割掩码
    
    参数:
        mask: 分割掩码
        ax: matplotlib轴对象
        random_color: 是否使用随机颜色
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])  # 默认蓝色半透明
    h, w = mask.shape[-2:]
    # 将掩码转换为彩色图像
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    """
    显示带标签的点
    
    参数:
        coords: 点坐标数组
        labels: 点标签（0或1）
        ax: matplotlib轴对象
        marker_size: 标记大小
    """
    pos_points = coords[labels==1]  # 正样本点（标签为1）
    neg_points = coords[labels==0]  # 负样本点（标签为0）
    # 绘制正样本为绿色星号
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    # 绘制负样本为红色星号
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax, label=None):
    """
    显示边界框
    
    参数:
        box: 边界框坐标 [x0, y0, x1, y1]
        ax: matplotlib轴对象
        label: 边界框标签文本
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    # 添加矩形框
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
    
    # 添加标签文本
    if label is not None:
        ax.text(x0, y0, label)
        
def vis_result_fast(
    image: np.ndarray,  # 输入图像
    detections: sv.Detections,  # 检测结果
    classes: list[str],  # 类别名称列表
    color: Color | ColorPalette = ColorPalette.DEFAULT,  # 颜色或调色板
    instance_random_color: bool = False,  # 是否为每个实例使用随机颜色
    draw_bbox: bool = True,  # 是否绘制边界框
) -> np.ndarray:
    """
    快速标注图像中的检测结果
    
    该函数使用supervision库快速标注图像，但由于保持原始图像分辨率，可能会显得模糊。
    
    参数:
        如上所示
        
    返回:
        tuple: (标注后的图像, 标签列表)
    """
    # 创建边界框标注器
    box_annotator = sv.BoxAnnotator(
        color = color,
        # text_scale=0.3,
        # text_thickness=1,
        # text_padding=2,
    )
    # 创建掩码标注器
    mask_annotator = sv.MaskAnnotator(
        color = color
    )

    # 生成标签
    if hasattr(detections, 'confidence') and hasattr(detections, 'class_id'):
        confidences = detections.confidence
        class_ids = detections.class_id
        if confidences is not None:
            # 如果有置信度，在标签中包含置信度
            labels = [
                f"{classes[class_id]} {confidence:0.2f}"
                for confidence, class_id in zip(confidences, class_ids)
            ]
        else:
            # 只有类别名称
            labels = [f"{classes[class_id]}" for class_id in class_ids]
    else:
        print("Detections object does not have 'confidence' or 'class_id' attributes or one of them is missing.")

    # 实例随机颜色处理
    if instance_random_color:
        # 创建检测结果的浅拷贝
        detections = dataclasses.replace(detections)
        # 为每个实例分配唯一ID，这样可以应用不同的颜色
        detections.class_id = np.arange(len(detections))
        
    # 先标注掩码
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    
    # 然后标注边界框
    if draw_bbox:
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    return annotated_image, labels

def vis_result_slow_caption(image, masks, boxes_filt, pred_phrases, caption, text_prompt):
    """
    带标题和提示文本的详细图像标注
    
    该函数虽然处理速度较慢，但输出更清晰可读，支持添加标题和提示文本。
    
    参数:
        image: 输入图像 (np.ndarray, HxWx3)
        masks: 分割掩码列表
        boxes_filt: 过滤后的边界框
        pred_phrases: 预测的文本短语
        caption: 图像标题
        text_prompt: 文本提示
        
    返回:
        np.ndarray: 标注后的 RGB 图像数组 (HxWx3, uint8)
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # 显示所有掩码
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    
    # 显示所有边界框和标签
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box, plt.gca(), label)

    # 添加标题和提示文本
    plt.title('Tagging-Caption: ' + caption + '\n' + 'Tagging-classes: ' + text_prompt + '\n')
    plt.axis('off')
    
    # 获取当前图形并调整布局
    fig = plt.gcf()
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    # === 关键修复：替换 tostring_rgb() ===
    # 新版 Matplotlib 使用 buffer_rgba()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    vis_image = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)  # RGBA
    
    # 转为 RGB（丢弃 Alpha 通道）
    vis_image = vis_image[:, :, :3]  # 取 R, G, B 三通道
    
    plt.close()  # 释放内存
    
    return vis_image
def vis_sam_mask(anns):
    """
    可视化SAM模型生成的分割掩码
    
    参数:
        anns: SAM掩码注释列表
        
    返回:
        np.ndarray: 带有叠加掩码的图像
    """
    # 按面积降序排序掩码
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    # 创建透明画布
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0  # 设置透明度为0
    # 为每个掩码应用随机颜色
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])  # 随机颜色+半透明
        img[m] = color_mask  # 在掩码区域应用颜色
        
    return img

def poses2lineset(poses, color=[0, 0, 1]):
    """
    从位姿序列创建Open3D线集
    
    参数:
        poses: 位姿数组，形状为(N, 4, 4)
        color: 线的颜色，RGB格式 [0, 1]
        
    返回:
        o3d.geometry.LineSet: 表示轨迹的线集
    """
    N = poses.shape[0]
    lineset = o3d.geometry.LineSet()
    # 设置点集（位姿的平移部分）
    lineset.points = o3d.utility.Vector3dVector(poses[:, :3, 3])
    # 创建线段连接相邻的点
    lineset.lines = o3d.utility.Vector2iVector(
        np.array([[i, i + 1] for i in range(N - 1)])
    )
    # 设置所有线段的颜色
    lineset.colors = o3d.utility.Vector3dVector([color for _ in range(len(lineset.lines))])
    return lineset

def create_camera_frustum(
    camera_pose, width=1, height=1, z_near=0.5, z_far=1, color=[0, 0, 1]
):
    """
    创建相机视锥体
    
    参数:
        camera_pose: 相机位姿矩阵 [4x4]
        width: 视锥体宽度
        height: 视锥体高度
        z_near: 近平面距离
        z_far: 远平面距离
        color: 视锥体颜色
        
    返回:
        o3d.geometry.LineSet: 表示相机视锥体的线集
    """
    # 内参矩阵近似
    K = np.array([[z_near, 0, 0], [0, z_near, 0], [0, 0, z_near + z_far]])
    # 定义视锥体近平面的四个角点和原点
    points = np.array(
        [
            [-width / 2, -height / 2, z_near],
            [width / 2, -height / 2, z_near],
            [width / 2, height / 2, z_near],
            [-width / 2, height / 2, z_near],
            [0, 0, 0],
        ]
    )
    # 应用相机位姿变换
    points_transformed = camera_pose[:3, :3] @ (K @ points.T) + camera_pose[:3, 3:4]
    points_transformed = points_transformed.T
    # 创建线集
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points_transformed)
    # 定义线段连接
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3]]
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return frustum


def better_camera_frustum(camera_pose, img_h, img_w, scale=3.0, color=[0, 0, 1]):
    """
    创建改进的相机视锥体表示
    
    基于图像尺寸创建更精确的视锥体，考虑了图像宽高比。
    
    参数:
        camera_pose: 相机位姿矩阵
        img_h: 图像高度
        img_w: 图像宽度
        scale: 视锥体缩放因子
        color: 视锥体颜色
        
    返回:
        o3d.geometry.LineSet: 表示视锥体的线集
    """
    # 将torch张量转换为numpy数组
    if isinstance(camera_pose, torch.Tensor):
        camera_pose = camera_pose.numpy()
    
    # 定义近平面和远平面距离
    near = scale * 0.1
    far = scale * 1.0
    
    # 基于近平面定义视锥体尺寸，并保持图像宽高比
    frustum_h = near
    frustum_w = frustum_h * img_w / img_h  # 根据图像宽高比设置视锥体宽度
    
    # 计算定义视锥体的8个点
    points = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                # 根据平面位置调整尺寸
                u = x * (frustum_w // 2 if z == -1 else frustum_w * far / near)
                v = y * (frustum_h // 2 if z == -1 else frustum_h * far / near)
                d = near if z == -1 else far
                point = np.array([u, v, d, 1]).reshape(-1, 1)
                # 应用相机变换
                transformed_point = (camera_pose @ point).ravel()[:3]
                points.append(transformed_point)
    
    # 创建连接8个点的线段
    lines = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], 
             [0, 4], [1, 5], [3, 7], [2, 6]]
    
    # 构建线集
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])

    return frustum


# 从https://github.com/isl-org/Open3D/pull/738复制
def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    使用轴角旋转将向量a对齐到向量b
    
    参数:
        a: 源向量
        b: 目标向量
        
    返回:
        tuple: (旋转轴, 旋转角度)，如果向量相同则返回(None, None)
    """
    if np.array_equal(a, b):
        return None, None
    # 计算旋转轴（叉积）
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)  # 归一化
    # 计算旋转角度（点积）
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """
    归一化numpy数组的点
    
    参数:
        a: 输入数组
        axis: 归一化轴
        order: 范数类型
        
    返回:
        tuple: (归一化数组, 原始范数)
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1  # 避免除以零
    return a / np.expand_dims(l2, axis), l2

def save_video_detections(exp_out_path, save_path=None, fps=30):
    """
    将文件夹中的检测结果保存为视频
    
    参数:
        exp_out_path: 包含检测结果图像的实验输出路径
        save_path: 视频保存路径，默认为exp_out_path/vis_video.mp4
        fps: 视频帧率
    """
    if save_path is None:
        save_path = exp_out_path / "vis_video.mp4"
    
    # 获取图像文件列表
    image_files = list((exp_out_path / "vis").glob("*.jpg"))
    image_files.sort()
    
    # 读取第一张图像获取尺寸
    image = Image.open(image_files[0])
    width, height = image.size
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
    
    # 将图像写入视频
    for image_file in image_files:
        image = cv2.imread(str(image_file))
        out.write(image)
    
    # 释放资源
    out.release()
    print(f"视频保存至 {save_path}")


class LineMesh(object):
    """
    线段网格类
    
    使用圆柱体网格表示线段，支持3D空间中的线条渲染。
    适用于需要粗线条表示的场景，如轨迹、连接等。
    """
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """
        创建表示为圆柱体网格序列的线

        参数:
            points {ndarray} -- 点的Numpy数组 Nx3

        可选参数:
            lines {list[list] or None} -- 表示线段的点索引对列表。如果为None，从有序点对隐式创建
            colors {list} -- 线条的颜色列表，或单色
            radius {float} -- 圆柱体半径
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []  # 存储各个圆柱体段

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        """
        从有序点创建线段列表
        
        参数:
            points: 有序的点数组
            
        返回:
            np.ndarray: 线段索引数组
        """
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        """
        创建线网格，为每个线段生成圆柱体
        """
        # 获取线段的起点和终点
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        # 计算线段向量
        line_segments = second_points - first_points
        # 归一化线段并计算长度
        line_segments_unit, line_lengths = normalized(line_segments)

        # 以Z轴作为参考
        z_axis = np.array([0, 0, 1])
        # 为每个线段创建圆柱体网格
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # 获取轴角旋转以将圆柱体与线段对齐
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # 计算平移向量（线段中点）
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # 创建圆柱体并应用变换
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            # 平移到正确位置
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            # 应用旋转
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a))
            # 设置颜色
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """
        将线添加到可视化器
        
        参数:
            vis: Open3D可视化器
        """
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder, reset_bounding_box=False)

    def remove_line(self, vis):
        """
        从可视化器移除线
        
        参数:
            vis: Open3D可视化器
        """
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder, reset_bounding_box=False)