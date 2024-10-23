# 作者：concha
# 创建时间：2024-10-23
# 最后修改时间：2020-12-12
# 版本：
#
# 该文件包含了齿轮六面体网格生成的工具函数，主要包括：
# 1. get_helix_points: 根据端面网格生成螺旋线上的点
# 2. sweep_points: 根据端面网格，使用扫掠法，生成整体网格节点
# 3. cylindrical_to_cartesian: 给定宽度坐标以及半径，计算其对应的笛卡尔坐标，适用于外齿轮
# 4. sign_of_tetrahedron_volume: 给定三维空间中一个面上三个点坐标，以及任意另一点的坐标，通过计算四点构成的四面体的有符号体积，判断任一点是在面的左侧、右侧还是面上
# 5. barycentric_coordinates: 计算点 P 在四面体 ABCD 上的重心坐标
# 6. hex_cell_search: 六面体网格单元搜索
# 7. hex_cell_search: 六面体网格单元搜索
# 8. hex_cell_search: 六面体网格单元搜索
#
# 注：这些函数后续可能会进一步封装成齿轮的类方法，但是为了后续工作能顺利开展目前暂时作为工具函数使用
# ======================================================================================================================
import pickle
import json
import numpy as np
from numpy import tan, arctan, sin, cos, pi, arctan2
from scipy.optimize import fsolve

from fealpy.mesh import QuadrangleMesh, HexahedronMesh
from .gear import ExternalGear, InternalGear


def get_helix_points(points, beta, r, h, t):
    """
    根据端面网格，基于扫掠法，获取指定参数位置的节点坐标
    :param points: 端面网格点
    :param beta: 分度圆柱面螺旋角（弧度制）
    :param r: 分度圆半径
    :param h: 齿宽
    :param t: 目标点参数
    :return: 指定参数位置的节点坐标
    """
    r_points = np.sqrt(np.sum(points ** 2, axis=-1))
    tooth_helix = h * tan(beta) / r
    start_angle = np.arctan2(points[..., 1], points[..., 0])

    if isinstance(t, (float, int)):
        volume_points = np.zeros_like(points)

        volume_points[..., 0] = r_points * cos((tooth_helix * t) + start_angle)
        volume_points[..., 1] = r_points * sin((tooth_helix * t) + start_angle)
        volume_points[..., 2] = h * t
    else:
        volume_points = np.zeros((len(t),) + points.shape)
        volume_points[0, :] = points

        x = r_points[None, :] * cos((tooth_helix * t[1:])[:, None] + start_angle[None, :])
        y = r_points[None, :] * sin((tooth_helix * t[1:])[:, None] + start_angle[None, :])
        z = h * t[1:, None]

        volume_points[1:, :, 0] = x
        volume_points[1:, :, 1] = y
        volume_points[1:, :, 2] = z

    return volume_points

def sweep_points(points, beta, r, h, n):
    """
    根据端面网格，使用扫掠法，生成整体网格节点
    :param points: 端面网格点
    :param beta: 分度圆柱面螺旋角（弧度制）
    :param r: 分度圆半径
    :param h: 齿宽
    :param n: 沿齿宽分段数
    :return: 整体网格节点
    """
    t = np.linspace(0, 1, n + 1)
    volume_points = get_helix_points(points, beta, r, h,t)

    return volume_points

def generate_hexahedral_mesh(quad_mesh, beta, r, tooth_width, nw):
    """
    根据齿轮端面网格，使用扫掠法，生成整体网格
    :param quad_mesh: 端面四面体网格
    :param beta: 螺旋角（弧度制）
    :param r: 端面分度圆半径
    :param tooth_width: 齿宽
    :param nw: 齿宽方向分段数
    :return: 端面四边形网格对应的六面体网格
    """
    node = quad_mesh.node
    cell = quad_mesh.cell
    # 数据处理，将二维点转换为三维点
    new_node = np.zeros((len(node), 3))
    new_node[:, 0:2] = node
    one_section_node_num = len(new_node)

    # 创建齿轮整体网格
    # 拉伸节点
    volume_node = sweep_points(new_node, beta, r, tooth_width, nw).reshape(-1, 3)
    # 将端面四边形单元拉伸为六面体单元
    volume_cell = np.zeros((nw, len(cell), 8), dtype=np.int64)
    cell_domain_tag = np.zeros((nw, len(cell)))
    cell_tooth_tag = np.zeros((nw, len(cell)))
    transverse_cell_domain_tag = quad_mesh.celldata['cell_domain_tag'][None, :]
    transverse_cell_tooth_tag = quad_mesh.celldata['cell_tooth_tag'][None, :]
    # 填充单元的节点索引
    for i in range(nw):
        volume_cell[i, :, 0:4] = cell + i * one_section_node_num
        volume_cell[i, :, 4:8] = cell + (i + 1) * one_section_node_num
        cell_domain_tag[i, :] = transverse_cell_domain_tag
        cell_tooth_tag[i, :] = transverse_cell_tooth_tag
    volume_cell = volume_cell.reshape(-1, 8)
    cell_domain_tag = cell_domain_tag.reshape(-1)
    cell_tooth_tag = cell_tooth_tag.reshape(-1)

    hex_mesh = HexahedronMesh(volume_node, volume_cell)
    hex_mesh.celldata['cell_domain_tag'] = cell_domain_tag
    hex_mesh.celldata['cell_tooth_tag'] = cell_tooth_tag

    return hex_mesh

def cylindrical_to_cartesian(d, width, gear):
    """
    给定宽度坐标以及半径，计算其对应的笛卡尔坐标，适用于外齿轮，内齿轮需要进一步测试
    :param d: 节点所在圆弧的直径
    :param width: 宽度
    :param gear: 齿轮对象
    :return: 节点笛卡尔坐标
    """
    r = d/2

    if isinstance(r, (float, int)):
        def involutecross(t2):
            return gear.get_tip_intersection_points(t2) - r
        # 计算端面点（z=0）坐标
        t = fsolve(involutecross, gear.m_n)[0]
        point_t = np.zeros(3)
        point_t[0:2] = gear.get_involute_points(t)

        # 根据螺旋线与当前宽度，计算实际坐标
        total_width = gear.tooth_width
        t2 = width / total_width
        point = get_helix_points(point_t, gear.beta, gear.r, total_width, t2)
    elif isinstance(r, (np.ndarray, list)):
        point_t = np.zeros((len(r), 3))
        total_width = gear.tooth_width
        t2 = width / total_width
        point = np.zeros((len(r), 3))
        for i in range(len(r)):
            def involutecross(t2):
                return gear.get_tip_intersection_points(t2) - r[i]
            # 计算端面点（z=0）坐标
            t = fsolve(involutecross, gear.m_n)[0]
            point_t[i, 0:2] = gear.get_involute_points(t)
            point[i] = get_helix_points(point_t[i], gear.beta, gear.r, total_width, t2[i])

    return point

def sign_of_tetrahedron_volume(p0, p1, p2, p3):
    """
    计算四点构成的四面体的有符号体积，四点满足右手系则结果为正，否则为负
    :param p0: 面上的第一个点
    :param p1: 面上的第二个点
    :param p2: 面上的第三个点
    :param p3: 面外的第四个点
    :return: 有符号体积
    """
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = p3 - p0
    return np.dot(np.cross(v0, v1, axis=-1), v2) / 6

def find_node_location_kd_tree(target_node, mesh: HexahedronMesh, error=1e-3):
    """
    查找目标节点在六面体网格中的位置，基于 kd_tree
    :param target_node: 目标节点坐标
    :param mesh: 网格
    :param error: 误差限制
    :return: 目标节点所在的单元索引，若未找到则返回-1
    """
    # 使用 kd_tree 算法，先计算所有单元的重心坐标，再根据重心坐标与target_node构建 kd_tree
    cell_barycenter = mesh.entity_barycenter('cell')
    # 计算每个单元重心坐标与 target_node 的距离，并排序从而构建kd_tree
    distance = np.linalg.norm(cell_barycenter - target_node, axis=1)
    kd_tree = np.argsort(distance)
    # 获取网格实体信息
    cell = mesh.cell
    node = mesh.node
    local_tetra = np.array([
        [0, 1, 2, 6],
        [0, 5, 1, 6],
        [0, 4, 5, 6],
        [0, 7, 4, 6],
        [0, 3, 7, 6],
        [0, 2, 3, 6]], dtype=np.int32)
    tetra_local_face = np.array([
        (1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)], dtype=np.int32)
    # 根据网格单元测度设置误差限制
    error = np.max(mesh.entity_measure('cell')) * error

    # 遍历单元搜寻
    for i in range(len(kd_tree)):
        cell_idx = kd_tree[i]
        cell_node = node[cell[cell_idx]]
        # ==============================================================================================
        # 方法一：优化效果不好，不推荐
        # 从距离最近的单元开始，使用 objective 函数与 trilinear_interpolation 函数，计算其对应的三线性插值参数
        # 若三个参数都在 [0, 1] 范围内，则找到目标单元，返回其索引，以及相关参数
        # 否则，继续查找下一个单元
        # # cell_idx = 163
        # P = node[cell[cell_idx]]
        # result = minimize(objective, [0.0, 0.0, 0.0], args=(P, target_node), bounds=[(-0.01, 1.01), (-0.01, 1.01), (-0.01, 1.01)])
        # # result = differential_evolution(objective, bounds=[(-0.01, 1.01), (-0.01, 1.01), (-0.01, 1.01)], args=(P, target_node))
        # u, v, w = result.x
        # if -1e-7 <= u <= 1 + 1e-7 and -1e-7 <= v <= 1 + 1e-7 and -1e-7 <= w <= 1 + 1e-7:
        #     return cell_idx, (u, v, w)
        # ==============================================================================================
        # 方法二：
        # 将一个六面体分成六个四面体，计算目标点是否在某一个四面体内（包括边界面与点）
        # 若六个四面体中有一个包含目标点，则返回当前六面体单元索引
        tetras = cell_node[local_tetra]
        # 遍历六个四面体
        for tetra in tetras:
            for i in range(tetra_local_face.shape[0]):
                current_face_node = tetra[tetra_local_face[i]]
                v = -sign_of_tetrahedron_volume(current_face_node[0], current_face_node[1], current_face_node[2],
                                                target_node)
                if v < 0 and abs(v - 0) > error:
                    break
                if (v > 0 or abs(v - 0) < error) and i == tetra_local_face.shape[0] - 1:
                    return cell_idx

    return -1








