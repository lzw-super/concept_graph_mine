# -*- coding: utf-8 -*-
"""
几何相交度量（IoU、GIoU等）计算工具集

本模块提供了一系列用于计算2D和3D边界框之间相交度量的函数，包括：
- 3D IoU（交集并集比）计算
- 2D IoU批量计算
- 3D GIoU（广义IoU）计算
- 3D边界框体积计算
- 边界框扩展和包含关系判断
- 点云包围体积计算

这些函数主要用于SLAM、目标检测、场景理解和概念图谱构建等任务中的目标匹配和跟踪。
"""
import numpy as np
import torch
import open3d as o3d

def compute_3d_iou(bbox1, bbox2, padding=0, use_iou=True):
    """
    计算两个3D边界框的IoU（交集并集比）
    
    Args:
        bbox1: Open3D的BoundingVolume对象，第一个边界框
        bbox2: Open3D的BoundingVolume对象，第二个边界框
        padding: 浮点数，边界框的填充大小，用于扩展边界框
        use_iou: 布尔值，若为True返回IoU值，否则返回最大重叠率
    
    Returns:
        float: 计算得到的IoU值或最大重叠率
    """
    # 获取第一个边界框的最小和最大坐标，并添加填充
    bbox1_min = np.asarray(bbox1.get_min_bound()) - padding
    bbox1_max = np.asarray(bbox1.get_max_bound()) + padding

    # 获取第二个边界框的最小和最大坐标，并添加填充
    bbox2_min = np.asarray(bbox2.get_min_bound()) - padding
    bbox2_max = np.asarray(bbox2.get_max_bound()) + padding

    # 计算两个边界框的重叠部分
    overlap_min = np.maximum(bbox1_min, bbox2_min)  # 重叠部分的最小坐标
    overlap_max = np.minimum(bbox1_max, bbox2_max)  # 重叠部分的最大坐标
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)  # 重叠部分的尺寸，确保非负

    # 计算重叠体积和两个边界框各自的体积
    overlap_volume = np.prod(overlap_size)  # 重叠部分的体积
    bbox1_volume = np.prod(bbox1_max - bbox1_min)  # 第一个边界框的体积
    bbox2_volume = np.prod(bbox2_max - bbox2_min)  # 第二个边界框的体积
    
    # 计算每个边界框被重叠的比例
    obj_1_overlap = overlap_volume / bbox1_volume  # 边界框1的重叠比例
    obj_2_overlap = overlap_volume / bbox2_volume  # 边界框2的重叠比例
    max_overlap = max(obj_1_overlap, obj_2_overlap)  # 最大重叠比例

    # 计算IoU = 交集体积 / (边界框1体积 + 边界框2体积 - 交集体积)
    iou = overlap_volume / (bbox1_volume + bbox2_volume - overlap_volume)

    # 根据参数决定返回IoU还是最大重叠率
    if use_iou:
        return iou
    else:
        return max_overlap

def compute_iou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    批量计算两组轴对齐3D边界框之间的IoU
    
    Args:
        bbox1: 形状为(M, V, D)的张量，例如(M, 8, 3)，表示M个边界框，每个边界框有V个顶点，每个顶点D个坐标
        bbox2: 形状为(N, V, D)的张量，例如(N, 8, 3)，表示N个边界框
    
    Returns:
        torch.Tensor: 形状为(M, N)的张量，表示bbox1中每个边界框与bbox2中每个边界框的IoU值
    """
    # 计算每个边界框的最小和最大坐标
    bbox1_min, _ = bbox1.min(dim=1)  # 形状: (M, 3)
    bbox1_max, _ = bbox1.max(dim=1)  # 形状: (M, 3)
    bbox2_min, _ = bbox2.min(dim=1)  # 形状: (N, 3)
    bbox2_max, _ = bbox2.max(dim=1)  # 形状: (N, 3)

    # 扩展维度以支持广播运算
    bbox1_min = bbox1_min.unsqueeze(1)  # 形状: (M, 1, 3)
    bbox1_max = bbox1_max.unsqueeze(1)  # 形状: (M, 1, 3)
    bbox2_min = bbox2_min.unsqueeze(0)  # 形状: (1, N, 3)
    bbox2_max = bbox2_max.unsqueeze(0)  # 形状: (1, N, 3)

    # 计算交集边界框的最小和最大坐标
    inter_min = torch.max(bbox1_min, bbox2_min)  # 形状: (M, N, 3)
    inter_max = torch.min(bbox1_max, bbox2_max)  # 形状: (M, N, 3)

    # 计算交集体积，将负值部分设为0
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)  # 形状: (M, N)

    # 计算两个边界框集合各自的体积
    bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # 形状: (M, 1)
    bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # 形状: (1, N)

    # 计算IoU，通过添加一个小的epsilon值(1e-10)避免除零错误
    iou = inter_vol / (bbox1_vol + bbox2_vol - inter_vol + 1e-10)

    return iou
    
def compute_3d_giou(bbox1, bbox2):
    """
    计算两个3D边界框的GIoU（广义IoU）
    
    GIoU是IoU的扩展，它考虑了两个边界框的空间关系，即使它们不重叠也能产生有意义的值
    
    Args:
        bbox1: Open3D的BoundingVolume对象，第一个边界框
        bbox2: Open3D的BoundingVolume对象，第二个边界框
    
    Returns:
        float: 计算得到的GIoU值
    """
    # 获取第一个边界框的最小和最大坐标
    bbox1_min = np.asarray(bbox1.get_min_bound())
    bbox1_max = np.asarray(bbox1.get_max_bound())

    # 获取第二个边界框的最小和最大坐标
    bbox2_min = np.asarray(bbox2.get_min_bound())
    bbox2_max = np.asarray(bbox2.get_max_bound())
    
    # 计算交集
    intersec_min = np.maximum(bbox1_min, bbox2_min)
    intersec_max = np.minimum(bbox1_max, bbox2_max)
    intersec_size = np.maximum(intersec_max - intersec_min, 0.0)
    intersec_volume = np.prod(intersec_size)

    # 计算并集体积
    bbox1_volume = np.prod(bbox1_max - bbox1_min)
    bbox2_volume = np.prod(bbox2_max - bbox2_min)
    union_volume = bbox1_volume + bbox2_volume - intersec_volume
    
    # 计算IoU
    iou = intersec_volume / union_volume
    
    # 计算包含两个边界框的最小外接边界框
    enclosing_min = np.minimum(bbox1_min, bbox2_min)
    enclosing_max = np.maximum(bbox1_max, bbox2_max)
    enclosing_size = np.maximum(enclosing_max - enclosing_min, 0.0)
    enclosing_volume = np.prod(enclosing_size)
    
    # 计算GIoU = IoU - (外接边界框体积 - 并集体积) / 外接边界框体积
    giou = iou - (enclosing_volume - union_volume) / enclosing_volume
    
    return giou

def compute_giou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    批量计算两组轴对齐3D边界框之间的GIoU（广义IoU）
    
    Args:
        bbox1: 形状为(M, V, D)的张量，例如(M, 8, 3)
        bbox2: 形状为(N, V, D)的张量，例如(N, 8, 3)
    
    Returns:
        torch.Tensor: 形状为(M, N)的张量，表示bbox1中每个边界框与bbox2中每个边界框的GIoU值
    """
    # 计算每个边界框的最小和最大坐标
    bbox1_min, _ = bbox1.min(dim=1)  # 形状: (M, D)
    bbox1_max, _ = bbox1.max(dim=1)  # 形状: (M, D)
    bbox2_min, _ = bbox2.min(dim=1)  # 形状: (N, D)
    bbox2_max, _ = bbox2.max(dim=1)  # 形状: (N, D)

    # 扩展维度以支持广播运算
    bbox1_min = bbox1_min.unsqueeze(1)  # 形状: (M, 1, D)
    bbox1_max = bbox1_max.unsqueeze(1)  # 形状: (M, 1, D)
    bbox2_min = bbox2_min.unsqueeze(0)  # 形状: (1, N, D)
    bbox2_max = bbox2_max.unsqueeze(0)  # 形状: (1, N, D)

    # 计算交集边界框的最小和最大坐标
    inter_min = torch.max(bbox1_min, bbox2_min)  # 形状: (M, N, D)
    inter_max = torch.min(bbox1_max, bbox2_max)  # 形状: (M, N, D)
    
    # 计算包含两个边界框的最小外接边界框坐标
    enclosing_min = torch.min(bbox1_min, bbox2_min)  # 形状: (M, N, D)
    enclosing_max = torch.max(bbox1_max, bbox2_max)  # 形状: (M, N, D)

    # 计算交集体积
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)  # 形状: (M, N)
    # 计算外接边界框体积
    enclosing_vol = torch.prod(enclosing_max - enclosing_min, dim=2)  # 形状: (M, N)

    # 计算两个边界框集合各自的体积
    bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # 形状: (M, 1)
    bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # 形状: (1, N)
    # 计算并集体积
    union_vol = bbox1_vol + bbox2_vol - inter_vol

    # 计算IoU和GIoU，添加小的epsilon值避免除零错误
    iou = inter_vol / (union_vol + 1e-10)
    giou = iou - (enclosing_vol - union_vol) / (enclosing_vol + 1e-10)

    return giou

def compute_3d_iou_accuracte_batch(bbox1, bbox2):
    """
    精确计算两组旋转的（或轴对齐的）3D边界框之间的IoU
    
    使用PyTorch3D库提供的box3d_overlap函数进行精确计算，适用于非轴对齐的边界框
    
    Args:
        bbox1: 形状为(M, 8, D)的张量，例如(M, 8, 3)
        bbox2: 形状为(N, 8, D)的张量，例如(N, 8, 3)
    
    Returns:
        torch.Tensor: 形状为(M, N)的张量，表示IoU值
    """
    # 必须预先扩展边界框，否则可能导致结果高估
    bbox1 = expand_3d_box(bbox1, 0.02)
    bbox2 = expand_3d_box(bbox2, 0.02)
    
    import pytorch3d.ops as ops

    # 重新排序边界框的顶点以符合PyTorch3D的格式要求
    bbox1 = bbox1[:, [0, 2, 5, 3, 1, 7, 4, 6]]
    bbox2 = bbox2[:, [0, 2, 5, 3, 1, 7, 4, 6]]
    
    # 使用PyTorch3D计算边界框重叠
    inter_vol, iou = ops.box3d_overlap(bbox1.float(), bbox2.float())
    
    return iou

def compute_3d_giou_accurate(obj1, obj2):
    """
    以更精确的方式计算3D GIoU
    
    该函数使用点云和边界框信息计算两个3D对象之间的GIoU，支持旋转的边界框
    
    Args:
        obj1: 字典，包含对象信息，必须包含'bbox'(边界框)和'pcd'(点云)键
        obj2: 字典，包含对象信息，必须包含'bbox'(边界框)和'pcd'(点云)键
    
    Returns:
        float: 计算得到的GIoU值
    """
    import pytorch3d.ops as ops
    
    # 获取边界框和点云信息
    bbox1 = obj1['bbox']
    bbox2 = obj2['bbox']
    pcd1 = obj1['pcd']
    pcd2 = obj2['pcd']
    
    # 获取边界框的顶点坐标
    box_points1 = np.asarray(bbox1.get_box_points())
    box_points2 = np.asarray(bbox2.get_box_points())
    
    # 重新排序顶点以符合PyTorch3D的格式要求
    # 顺序应为 [---, -+-, -++, --+, +--, ++-, +++, +-+]
    box_points1 = box_points1[[0, 2, 5, 3, 1, 7, 4, 6]]
    box_points2 = box_points2[[0, 2, 5, 3, 1, 7, 4, 6]]
    
    # 计算两个边界框的交集
    try:
        vols, ious = ops.box3d_overlap(
            torch.from_numpy(box_points1).unsqueeze(0).float(), 
            torch.from_numpy(box_points2).unsqueeze(0).float()
        )
    except ValueError as e:  # 处理共线点的异常情况
        union_volume = 0.0
        iou = 0.0
    else:
        union_volume = vols[0,0].item()
        iou = ious[0,0].item()
    
    # 合并两个点云
    pcd_union = pcd1 + pcd2

    # 使用定向边界框计算包围体积
    enclosing_box = pcd_union.get_oriented_bounding_box()
    enclosing_volume = enclosing_box.volume()
    
    # 计算GIoU
    giou = iou - (enclosing_volume - union_volume) / enclosing_volume
    
    return giou

def compute_3d_box_volume_batch(bbox: torch.Tensor) -> torch.Tensor:
    """
    批量计算一组矩形边界框的体积
    
    假设边界框顶点顺序遵循Open3D的约定，即：
    ---, +--, -+-, --+, +++, -++, +-+, ++-
    
    Args:
        bbox: 形状为(N, 8, D)的张量，表示N个边界框
    
    Returns:
        torch.Tensor: 形状为(N,)的张量，表示每个边界框的体积
    """
    # 计算边界框的三个边长（使用顶点间的欧几里得距离）
    a = torch.linalg.vector_norm(bbox[:, 0, :] - bbox[:, 1, :], ord=2, dim=1)
    b = torch.linalg.vector_norm(bbox[:, 0, :] - bbox[:, 2, :], ord=2, dim=1)
    c = torch.linalg.vector_norm(bbox[:, 0, :] - bbox[:, 3, :], ord=2, dim=1)
    
    # 体积 = 长 × 宽 × 高
    vol = a * b * c
    return vol
    
def expand_3d_box(bbox: torch.Tensor, eps=0.02) -> torch.Tensor:
    """
    扩展3D边界框的边长，确保每边至少有eps长度
    
    适用于处理非常小或扁平的边界框，避免数值不稳定问题
    
    Args:
        bbox: 形状为(N, 8, D)的张量，表示N个边界框
        eps: 浮点数，边界框每边的最小长度
    
    Returns:
        torch.Tensor: 形状为(N, 8, D)的张量，表示扩展后的边界框
    """
    # 计算每个边界框的中心点
    center = bbox.mean(dim=1)  # 形状: (N, D)

    # 计算从第一个顶点出发的三个边向量
    va = bbox[:, 1, :] - bbox[:, 0, :]  # 形状: (N, D)
    vb = bbox[:, 2, :] - bbox[:, 0, :]  # 形状: (N, D)
    vc = bbox[:, 3, :] - bbox[:, 0, :]  # 形状: (N, D)
    
    # 计算三个边的长度
    a = torch.linalg.vector_norm(va, ord=2, dim=1, keepdim=True)  # 形状: (N, 1)
    b = torch.linalg.vector_norm(vb, ord=2, dim=1, keepdim=True)  # 形状: (N, 1)
    c = torch.linalg.vector_norm(vc, ord=2, dim=1, keepdim=True)  # 形状: (N, 1)
    
    # 扩展边长小于eps的边，保持方向不变
    va = torch.where(a < eps, va / a * eps, va)  # 形状: (N, D)
    vb = torch.where(b < eps, vb / b * eps, vb)  # 形状: (N, D)
    vc = torch.where(c < eps, vc / c * eps, vc)  # 形状: (N, D)
    
    # 从中心点重新计算边界框的8个顶点
    new_bbox = torch.stack([
        center - va/2.0 - vb/2.0 - vc/2.0,
        center + va/2.0 - vb/2.0 - vc/2.0,
        center - va/2.0 + vb/2.0 - vc/2.0,
        center - va/2.0 - vb/2.0 + vc/2.0,
        center + va/2.0 + vb/2.0 + vc/2.0,
        center - va/2.0 + vb/2.0 + vc/2.0,
        center + va/2.0 - vb/2.0 + vc/2.0,
        center + va/2.0 + vb/2.0 - vc/2.0,
    ], dim=1)  # 形状: (N, 8, D)
    
    # 确保输出与输入在同一设备和数据类型
    new_bbox = new_bbox.to(bbox.device)
    new_bbox = new_bbox.type(bbox.dtype)
    
    return new_bbox
    
def compute_enclosing_vol(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    计算bbox1和bbox2中每对边界框之间的包围体积
    
    这是一个使用凸包的精确但较慢的版本
    
    Args:
        bbox1: 形状为(M, 8, D)的张量
        bbox2: 形状为(N, 8, D)的张量
    
    Returns:
        torch.Tensor: 形状为(M, N)的张量，表示每对边界框的包围体积
    """
    M = bbox1.shape[0]
    N = bbox2.shape[0]
    
    # 初始化包围体积矩阵
    enclosing_vol = torch.zeros((M, N), dtype=bbox1.dtype, device=bbox1.device)
    for i in range(bbox1.shape[0]):
        for j in range(bbox2.shape[0]):
            # 创建点云对象
            pcd_union = o3d.geometry.PointCloud()
            # 合并两个边界框的所有顶点
            bbox_points_union = torch.cat([bbox1[i], bbox2[j]], dim=0)  # (16, 3)
            pcd_union.points = o3d.utility.Vector3dVector(bbox_points_union.cpu().numpy())
            # 计算凸包
            enclosing_mesh, _ = pcd_union.compute_convex_hull(joggle_inputs=True)
            try:
                # 计算凸包体积
                enclosing_vol[i, j] = enclosing_mesh.get_volume()
            except:
                # 如果凸包计算失败（非水密网格），则使用轴对齐边界框作为替代
                enclosing_mesh = pcd_union.get_axis_aligned_bounding_box()
                enclosing_vol[i, j] = enclosing_mesh.volume()
                
    return enclosing_vol
    
def compute_enclosing_vol_fast(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    快速计算bbox1和bbox2中每对边界框之间的包围体积
    
    这是一个使用轴对齐边界框的快速但近似版本
    
    Args:
        bbox1: 形状为(M, 8, 3)的张量
        bbox2: 形状为(N, 8, 3)的张量
    
    Returns:
        torch.Tensor: 形状为(M, N)的张量，表示每对边界框的包围体积
    """
    M = bbox1.shape[0]
    N = bbox2.shape[0]
    
    # 扩展维度以计算所有边界框对的包围框
    bbox1 = bbox1.unsqueeze(1).expand(-1, N, -1, -1)  # (M, N, 8, 3)
    bbox2 = bbox2.unsqueeze(0).expand(M, -1, -1, -1)  # (M, N, 8, 3)
    
    # 计算每对边界框的最小和最大坐标
    min_coords = torch.minimum(bbox1, bbox2).amin(dim=2)  # (M, N, 3)
    max_coords = torch.maximum(bbox1, bbox2).amax(dim=2)  # (M, N, 3)

    # 计算包围框的尺寸
    enclosing_dims = max_coords - min_coords  # (M, N, 3)
    
    # 将尺寸限制为非负值（以防没有重叠）
    enclosing_dims = torch.clamp(enclosing_dims, min=0)  # (M, N, 3)
    
    # 计算包围框的体积
    vol = enclosing_dims[:, :, 0] * enclosing_dims[:, :, 1] * enclosing_dims[:, :, 2]  # (M, N)

    return vol

def compute_3d_giou_accurate_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    精确批量计算两组旋转的（或轴对齐的）3D边界框之间的GIoU
    
    Args:
        bbox1: 形状为(M, 8, D)的张量，例如(M, 8, 3)
        bbox2: 形状为(N, 8, D)的张量，例如(N, 8, 3)
    
    Returns:
        torch.Tensor: 形状为(M, N)的张量，表示GIoU值
    """
    # 必须预先扩展边界框，否则可能导致结果高估
    bbox1 = expand_3d_box(bbox1, 0.02)
    bbox2 = expand_3d_box(bbox2, 0.02)
    
    # 计算每个边界框的体积
    bbox1_vol = compute_3d_box_volume_batch(bbox1)
    bbox2_vol = compute_3d_box_volume_batch(bbox2)
    
    import pytorch3d.ops as ops

    # 计算边界框重叠
    inter_vol, iou = ops.box3d_overlap(
        bbox1[:, [0, 2, 5, 3, 1, 7, 4, 6]].float(), 
        bbox2[:, [0, 2, 5, 3, 1, 7, 4, 6]].float()
    )
    # 计算并集体积
    union_vol = bbox1_vol.unsqueeze(1) + bbox2_vol.unsqueeze(0) - inter_vol
    
    # 计算包围体积（可以选择使用精确版本或快速版本）
    enclosing_vol = compute_enclosing_vol(bbox1, bbox2)
    # enclosing_vol = compute_enclosing_vol_fast(bbox1, bbox2)
    
    # 计算GIoU
    giou = iou - (enclosing_vol - union_vol) / enclosing_vol
    
    return giou

def compute_3d_contain_ratio_accurate_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    计算bbox1中每个边界框被bbox2中每个边界框包含的比例
    
    Args:
        bbox1: 形状为(M, 8, D)的张量，例如(M, 8, 3)
        bbox2: 形状为(N, 8, D)的张量，例如(N, 8, 3)
    
    Returns:
        tuple: 包含两个张量的元组
            - contain_ratio: 形状为(M, N)的张量，表示包含比例
            - iou: 形状为(M, N)的张量，表示IoU值
    """
    # 必须预先扩展边界框，否则可能导致结果高估
    bbox1 = expand_3d_box(bbox1)
    bbox2 = expand_3d_box(bbox2)
    
    # 计算每个边界框的体积
    bbox1_vol = compute_3d_box_volume_batch(bbox1)  # (M,)
    bbox2_vol = compute_3d_box_volume_batch(bbox2)  # (M,)
    
    import pytorch3d.ops as ops
    
    # 计算边界框重叠
    inter_vol, iou = ops.box3d_overlap(
        bbox1[:, [0, 2, 5, 3, 1, 7, 4, 6]].float(), 
        bbox2[:, [0, 2, 5, 3, 1, 7, 4, 6]].float()
    )  # (M, N), (M, N)
    
    # 计算包含比例：交集体积 / bbox1体积
    contain_ratio = inter_vol / bbox1_vol.unsqueeze(1)  # (M, N)
    
    # 手动将包含比例限制在[0, 1]范围内，避免数值误差导致的超过1的情况
    contain_ratio = contain_ratio.clamp(min=0, max=1)
    
    return contain_ratio, iou

def compute_2d_box_contained_batch(bbox: torch.Tensor, thresh:float=0.95) -> torch.Tensor:
    """
    对于每个边界框，计算有多少其他边界框包含它
    
    首先计算每对边界框之间的交集面积，然后对每个边界框，统计有多少个边界框的交集面积
    大于其自身面积的thresh比例
    
    Args:
        bbox: 形状为(N, 4)的张量，格式为(x1, y1, x2, y2)
        thresh: 浮点数，判断包含关系的阈值
    
    Returns:
        torch.Tensor: 形状为(N,)的张量，表示每个边界框被包含的次数
    """
    N = bbox.shape[0]

    # 计算每个边界框的面积
    areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

    # 计算交集边界框的左上角和右下角坐标
    lt = torch.max(bbox[:, None, :2], bbox[:, :2])  # 左上角点 (N, N, 2)
    rb = torch.min(bbox[:, None, 2:], bbox[:, 2:])  # 右下角点 (N, N, 2)

    # 计算交集的尺寸，将负值设为0
    inter = (rb - lt).clamp(min=0)  # 交集尺寸 (dx, dy) (N, N, 2)

    # 计算交集面积
    inter_areas = inter[:, :, 0] * inter[:, :, 1]  # (N, N)

    # 统计满足包含条件的边界框数量
    mask = inter_areas > (areas * thresh).unsqueeze(1)  # (N, N)
    count = mask.sum(dim=1) - 1  # 排除自身 (N,)

    return count

def mask_subtract_contained(xyxy: np.ndarray, mask: np.ndarray, th1=0.8, th2=0.7):
    """
    计算所有边界框对之间的包含关系，并从包含其他边界框的掩码中减去被包含的掩码
    
    Args:
        xyxy: 形状为(N, 4)的数组，格式为(x1, y1, x2, y2)
        mask: 形状为(N, H, W)的数组，二值掩码
        th1: 浮点数，计算box1对box2的交集比例阈值
        th2: 浮点数，计算box2对box1的交集比例阈值
    
    Returns:
        np.ndarray: 形状为(N, H, W)的数组，表示减去被包含边界框后的掩码
    """
    N = xyxy.shape[0]  # 边界框数量

    # 计算每个边界框的面积
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])  # (N,)

    # 计算交集边界框的左上角和右下角坐标
    lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # 左上角点 (N, N, 2)
    rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])  # 右下角点 (N, N, 2)
    
    # 计算交集的尺寸，将负值设为0
    inter = (rb - lt).clip(min=0)  # 交集尺寸 (dx, dy) (N, N, 2)

    # 计算交集面积
    inter_areas = inter[:, :, 0] * inter[:, :, 1]  # (N, N)
    
    # 计算交集比例
    inter_over_box1 = inter_areas / areas[:, None]  # (N, N)，box1与box2的交集占box1的比例
    inter_over_box2 = inter_over_box1.T  # (N, N)，box1与box2的交集占box2的比例（等同于转置）
    
    # 判断包含关系：如果box1与box2的交集占box1的比例小于th2，
    # 且占box2的比例大于th1，则认为box2被box1包含
    contained = (inter_over_box1 < th2) & (inter_over_box2 > th1)  # (N, N)
    contained_idx = contained.nonzero()  # (num_contained, 2)，存储所有包含关系对

    # 复制原始掩码
    mask_sub = mask.copy()  # (N, H, W)
    
    # 从包含者的掩码中减去被包含者的掩码
    for i in range(len(contained_idx[0])):
        mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (~mask_sub[contained_idx[1][i]])

    return mask_sub