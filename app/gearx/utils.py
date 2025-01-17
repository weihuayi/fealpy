# 作者：concha
# 创建时间：2024-10-23
# 最后修改时间：
# 版本：
#
# 该文件包含了齿轮六面体网格生成的工具函数，主要包括：
# 1. get_helix_points：根据端面网格，基于扫掠法，获取指定参数位置的节点坐标
# 2. sweep_points：根据端面网格，使用扫掠法，生成整体网格节点
# 3. generate_hexahedral_mesh：根据齿轮端面网格，使用扫掠法，生成整体网格
# 4. cylindrical_to_cartesian：给定宽度坐标以及半径，计算其对应的笛卡尔坐标，适用于外齿轮
# 5. sign_of_tetrahedron_volume：计算四点构成的四面体的有符号体积，四点满足右手系则结果为正，否则为负
# 6. find_node_location_kd_tree：查找目标节点在六面体网格中的位置，基于 kd_tree
#
# 注：这些函数后续可能会进一步封装成齿轮的类方法，但是为了后续工作能顺利开展目前暂时作为工具函数使用
# ======================================================================================================================
import pickle
import json
import numpy as np
from numpy import tan, arctan, sin, cos, pi, arctan2
from scipy.optimize import fsolve, minimize, differential_evolution

from fealpy.mesh import QuadrangleMesh, HexahedronMesh


def get_helix_points(points, beta, r, h, t, rotation_direction=1):
    """
    根据端面网格，基于扫掠法，获取指定参数位置的节点坐标
    :param points: 端面网格点
    :param beta: 分度圆柱面螺旋角（弧度制）
    :param r: 分度圆半径
    :param h: 齿宽
    :param t: 目标点参数
    :param rotation_direction: 旋转方向，1 为右旋齿轮，-1 为左旋齿轮，默认为右旋
    :return: 指定参数位置的节点坐标
    """
    r_points = np.sqrt(np.sum(points ** 2, axis=-1))
    tooth_helix = rotation_direction * h * tan(beta) / r
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


def sweep_points(points, beta, r, h, n, rotation_direction=1):
    """
    根据端面网格，使用扫掠法，生成整体网格节点
    :param points: 端面网格点
    :param beta: 分度圆柱面螺旋角（弧度制）
    :param r: 分度圆半径
    :param h: 齿宽
    :param n: 沿齿宽分段数
    @param rotation_direction: 旋转方向，1 为右旋齿轮，-1 为左旋齿轮，默认为右旋
    :return: 整体网格节点
    """
    t = np.linspace(0, 1, n + 1)
    volume_points = get_helix_points(points, beta, r, h, t, rotation_direction)

    return volume_points


def generate_hexahedron_mesh(quad_mesh, beta, r, tooth_width, nw, rotation_direction=1):
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
    volume_node = sweep_points(new_node, beta, r, tooth_width, nw, rotation_direction).reshape(-1, 3)
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
    r = d / 2

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
        point = get_helix_points(point_t, gear.beta, gear.r, total_width, t2, gear.rotation_direction)
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
            point[i] = get_helix_points(point_t[i], gear.beta, gear.r, total_width, t2[i], gear.rotation_direction)
    point[..., 0:2] = np.dot(point[..., 0:2], np.array([[0, -1], [1, 0]]))
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


def trilinear_interpolation(u, v, w, P):
    P000, P100, P010, P110, P001, P101, P011, P111 = P
    return (
            (1 - u) * (1 - v) * (1 - w) * P000 +
            u * (1 - v) * (1 - w) * P100 +
            (1 - u) * v * (1 - w) * P010 +
            u * v * (1 - w) * P110 +
            (1 - u) * (1 - v) * w * P001 +
            u * (1 - v) * w * P101 +
            (1 - u) * v * w * P011 +
            u * v * w * P111
    )


def find_node_location_kd_tree(target_node, gear, mesh: HexahedronMesh, error=1e-3):
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
    tetra_face_to_hex_face = np.array([
        (3, -1, -1, 0),
        (3, -1, -1, 4),
        (1, -1, -1, 4),
        (1, -1, -1, 2),
        (5, -1, -1, 2),
        (5, -1, -1, 0)], dtype=np.int32)
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
        for j, tetra in enumerate(tetras):
            for i in range(tetra_local_face.shape[0]):
                current_face_node = tetra[tetra_local_face[i]]
                v = -sign_of_tetrahedron_volume(current_face_node[0], current_face_node[1], current_face_node[2],
                                                target_node)
                if v < 0 and abs(v - 0) > error:
                    break
                if (v > 0 or abs(v - 0) < error) and i == tetra_local_face.shape[0] - 1:
                    t = (target_node[2] - cell_node[0, 2]) / (cell_node[4, 2] - cell_node[0, 2]);
                    r_points = np.sqrt(np.sum(cell_node[0:4, 0:2] ** 2, axis=-1))
                    tooth_helix = (cell_node[4, 2] - cell_node[0, 2]) * tan(gear.beta) / gear.r
                    start_angle = arctan2(cell_node[0:4, 1], cell_node[0:4, 0])
                    t_z = (cell_node[4, 2] - cell_node[0, 2]) * t + cell_node[0, 2]

                    # 构建目标节点所在截面四边形
                    t_node = np.zeros((4, 2))
                    t_node[:, 0] = r_points * cos((tooth_helix * t) + start_angle)
                    t_node[:, 1] = r_points * sin((tooth_helix * t) + start_angle)
                    # t_node[:, 2] = t_z
                    P00 = t_node[0]
                    P10 = t_node[1]
                    P11 = t_node[2]
                    P01 = t_node[3]
                    P = target_node[0:2]

                    # # 判断五点是否共面
                    # v_sign1 = sign_of_tetrahedron_volume(P00, P10, P11, P01)
                    # v_sign2 = sign_of_tetrahedron_volume(P00, P10, P11, target_node)
                    # v_sign3 = sign_of_tetrahedron_volume(P00, P11, P01, target_node)


                    # 计算二元一次方程组系数
                    # (P00-P10)x(P01-P11)u**2 + ((P-P00)x(P01-P11)-(P-P01)x(P00-P10))u + (P-P00)x(P-P01) = 0
                    a = (P00[0] * P01[1] - P01[0] * P00[1] - P00[0] * P11[1] + P01[0] * P10[1]
                         - P10[0] * P01[1] + P11[0] * P00[1] + P10[0] * P11[1] - P11[0] * P10[1])
                    b = (P[0] * P01[1] - P[0] * P00[1] + P00[0] * P[1] - P01[0] * P[1]
                         + P[0] * P10[1] - P[0] * P11[1] + P11[0] * P[1] - P10[0] * P[1]
                         - 2 * P00[0] * P01[1] + 2 * P01[0] * P00[1] +
                         P00[0] * P11[1] - P01[0] * P10[1] + P10[0] * P01[1] - P11[0] * P00[1])
                    c = (P[0] * P00[1] - P[0] * P01[1] + P01[0] * P[1] - P00[0] * P[1]
                         + P00[0] * P01[1] - P01[0] * P00[1])

                    # a1 = np.cross((P00-P10), (P01-P11))
                    # b1 = np.cross((P-P00), (P01-P11))-np.cross((P-P01), (P00-P10))
                    # c1 = np.cross((P-P00), (P-P01))
                    # 计算参数
                    delta = b ** 2 - 4 * a * c
                    u = -1
                    if delta < 0:
                        break
                    else:
                        u0 = (-b + np.sqrt(delta)) / (2 * a)
                        u1 = (-b - np.sqrt(delta)) / (2 * a)
                        if 0-error*10 <= u0 <= 1+error*10:
                            u = u0
                        elif 0-error*10 <= u1 <= 1+error*10:
                            u = u1
                    v = -1
                    if u != -1:
                        Pu0 = (1 - u) * P00 + u * P10
                        Pu1 = (1 - u) * P01 + u * P11
                        v0 = (P[0]-Pu0[0]) / (Pu1[0] - Pu0[0])
                        v1 = (P[1]-Pu0[1]) / (Pu1[1] - Pu0[1])
                        if abs(v0-v1) < error**2:
                                v = v0
                    w = t

                    return cell_idx, tetra_face_to_hex_face[j, i], (u, v, w)

    return -1, -1, -1

def export_to_inp(filename, nodes, elements, fixed_nodes, load_nodes, loads, young_modulus, poisson_ratio, density, used_app='abaqus', mesh_type='hex'):
    """
    齿轮专用的，相关信息导出为 Abaqus 的 inp 文件
    :param filename: 文件名
    :param nodes: 网格节点
    :param elements: 网格单元
    :param fixed_nodes: 固定点索引
    :param load_nodes: 载荷点索引
    :param loads: 载荷点载荷
    :param young_modulus: 杨氏模量（GP）
    :param poisson_ratio: 泊松比
    :param density: 密度
    :param used_app: 使用的有限元软件，默认为 Abaqus
    :param mesh_type: 网格类型，默认为六面体网格（hex），可选四面体网格（tet）
    :return:
    """
    assert used_app in ['abaqus', 'ansys']
    assert mesh_type in ['hex', 'tet']
    if used_app == 'abaqus':
        export_to_inp_abaqus(filename, nodes, elements, fixed_nodes, load_nodes, loads, young_modulus, poisson_ratio, density, mesh_type)
    elif used_app == 'ansys':
        export_to_inp_ansys(filename, nodes, elements, fixed_nodes, load_nodes, loads, young_modulus, poisson_ratio, density, mesh_type)
    else:
        raise ValueError("Invalid used_app parameter!")

def export_to_inp_abaqus(filename, nodes, elements, fixed_nodes, load_nodes, loads, young_modulus, poisson_ratio, density, mesh_type='hex'):
    """
    齿轮专用的，相关信息导出为 Abaqus 的 inp 文件
    :param filename: 文件名
    :param nodes: 网格节点
    :param elements: 网格单元
    :param fixed_nodes: 固定点索引
    :param load_nodes: 载荷点索引
    :param loads: 载荷点载荷
    :param young_modulus: 杨氏模量（GP）
    :param poisson_ratio: 泊松比
    :param density: 密度
    :param mesh_type: 网格类型，默认为六面体网格（hex），可选四面体网格（tet）
    :return:
    """
    with open(filename, 'w') as file:
        file.write("*Heading\n** Generated by Custom Export Script\n*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
        file.write("*Part, name=Gear\n")
        file.write("*Node\n")

        elements = elements+1
        fixed_nodes = fixed_nodes+1
        load_nodes = load_nodes+1
        # 写入节点信息
        for i, node in enumerate(nodes):
            file.write(f"{i+1}, {node[0]}, {node[1]}, {node[2]}\n")

        # 写入单元信息
        if mesh_type == 'hex':  # 六面体网格
            file.write("*Element, type=C3D8, elset=AllElements\n")
            for i, elem in enumerate(elements):
                file.write(
                    f"{i + 1}, {elem[0]}, {elem[1]}, {elem[2]}, {elem[3]}, {elem[4]}, {elem[5]}, {elem[6]}, {elem[7]}\n")
        elif mesh_type == 'tet':  # 四面体网格
            file.write("*Element, type=C3D4, elset=AllElements\n")
            for i, elem in enumerate(elements):
                file.write(f"{i + 1}, {elem[0]}, {elem[1]}, {elem[2]}, {elem[3]}\n")

        # 写入截面
        file.write("*Solid Section, elset=AllElements, material=Steel\n")
        file.write("*End Part\n**\n")

        # 定义装配和实例
        file.write("*Assembly, name=Assembly\n*Instance, name=GearInstance, part=Gear\n*End Instance\n")

        # 定义固定节点集
        file.write("*Nset, nset=FixedNodes, instance=GearInstance\n")
        for i in range(0, len(fixed_nodes), 16):
            file.write(", ".join(str(node) for node in fixed_nodes[i:i + 16]) + ",\n")

        # 定义载荷节点集
        file.write("*Nset, nset=LoadNodes, instance=GearInstance\n")
        for i in range(0, len(load_nodes), 16):
            file.write(", ".join(str(node) for node in load_nodes[i:i + 16]) + ",\n")
        file.write("*End Assembly\n")

        # 写入材料信息
        file.write("*Material, name=Steel\n")
        file.write(f"*Density\n{density}\n")
        file.write(f"*Elastic\n{young_modulus}, {poisson_ratio}\n")

        # 写入步骤、边界条件和载荷
        file.write("** STEP: LoadStep\n*Step, name=LoadStep, nlgeom=NO\n*Static\n1., 1., 1e-05, 1.\n")

        # 固定边界条件
        file.write("*Boundary\nFixedNodes, ENCASTRE\n")

        # 施加集中载荷
        file.write(f"*Cload\n")
        for i, load_node in enumerate(load_nodes):
            node_id = load_node
            forces = loads[i]
            file.write(f"GearInstance.{node_id}, 1, {forces[0]}\n")
            file.write(f"GearInstance.{node_id}, 2, {forces[1]}\n")
            file.write(f"GearInstance.{node_id}, 3, {forces[2]}\n")

        file.write("*Output, field, variable=PRESELECT\n")
        file.write("*Output, history, variable=PRESELECT\n")
        file.write("*End Step\n")
        file.write("** Output Global Stiffness Matrix\n")
        file.write("*Step, name=Global_Stiffness_Matrix\n")
        file.write("*MATRIX GENERATE, STIFFNESS, element by element\n")
        file.write("*MATRIX OUTPUT, STIFFNESS, FORMAT=COORDINATE\n")
        file.write("*End Step\n")

        print("Export to inp file successfully!")

def export_to_inp_ansys(filename, nodes, elements, fixed_nodes, load_nodes, loads, young_modulus, poisson_ratio, density, mesh_type='hex'):
    """
        齿轮专用的，相关信息导出为 Abaqus 的 inp 文件
        :param filename: 文件名
        :param nodes: 网格节点
        :param elements: 网格单元
        :param fixed_nodes: 固定点索引
        :param load_nodes: 载荷点索引
        :param loads: 载荷点载荷
        :param young_modulus: 杨氏模量（GP）
        :param poisson_ratio: 泊松比
        :param density: 密度
        :param mesh_type: 网格类型，默认为六面体网格（hex），可选四面体网格（tet）
        :return:
        """
    with open(filename, 'w') as file:
        file.write(
            "*Heading\n** Generated by Custom Export Script\n*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
        file.write("*Node\n")

        elements = elements + 1
        fixed_nodes = fixed_nodes + 1
        load_nodes = load_nodes + 1
        # 写入节点信息
        for i, node in enumerate(nodes):
            file.write(f"{i + 1}, {node[0]}, {node[1]}, {node[2]}\n")

        # 写入单元信息
        if mesh_type == 'hex':  # 六面体网格
            file.write("*Element, type=C3D8, elset=AllElements\n")
            for i, elem in enumerate(elements):
                file.write(
                    f"{i + 1}, {elem[0]}, {elem[1]}, {elem[2]}, {elem[3]}, {elem[4]}, {elem[5]}, {elem[6]}, {elem[7]}\n")
        elif mesh_type == 'tet':  # 四面体网格
            file.write("*Element, type=C3D4, elset=AllElements\n")
            for i, elem in enumerate(elements):
                file.write(f"{i + 1}, {elem[0]}, {elem[1]}, {elem[2]}, {elem[3]}\n")

        # 定义固定节点集
        file.write("*Nset, nset=FixedNodes\n")
        for i in range(0, len(fixed_nodes), 16):
            file.write(", ".join(str(node) for node in fixed_nodes[i:i + 16]) + ",\n")

        # 定义载荷节点集
        file.write("*Nset, nset=LoadNodes\n")
        for i in range(0, len(load_nodes), 16):
            file.write(", ".join(str(node) for node in load_nodes[i:i + 16]) + ",\n")

        # 写入材料信息
        file.write("*Material, name=Steel\n")
        file.write(f"*Density\n{density}\n")
        file.write(f"*Elastic\n{young_modulus}, {poisson_ratio}\n")

        # 写入截面
        file.write("*Solid Section, elset=AllElements, material=Steel\n")
        file.write(",\n")
        file.write("*End Part\n**\n")
        file.write("*End Assembly\n")

        # 写入步骤、边界条件和载荷
        file.write("** STEP: LoadStep\n*Step, name=LoadStep, nlgeom=NO\n*Static\n1., 1., 1e-05, 1.\n")

        # 固定边界条件
        file.write("*Boundary\nFixedNodes, ENCASTRE\n")

        # 施加集中载荷
        file.write(f"*Cload\n")
        for i, load_node in enumerate(load_nodes):
            node_id = load_node
            forces = loads[i]
            if forces[0] != 0:
                file.write(f"{node_id}, 1, {forces[0]}\n")
            if forces[1] != 0:
                file.write(f"{node_id}, 2, {forces[1]}\n")
            if forces[2] != 0:
                file.write(f"{node_id}, 3, {forces[2]}\n")

        file.write("*End Step\n")

        print("Export to inp file successfully!")

def export_to_inp_by_u(filename, nodes, elements, boundary_nodes_idx, boundary_nodes_u, young_modulus, poisson_ratio, density, used_app='abaqus', mesh_type='hex'):
    """
    齿轮专用的，相关信息导出为 Abaqus 的 inp 文件
    :param filename: 文件名
    :param nodes: 网格节点
    :param elements: 网格单元
    :param boundary_nodes_idx: 边界点索引
    :param boundary_nodes_u: 边界点位移
    :param young_modulus: 杨氏模量（GP）
    :param poisson_ratio: 泊松比
    :param density: 密度
    :param used_app: 使用的有限元软件，默认为 Abaqus
    :param mesh_type: 网格类型，默认为六面体网格（hex），可选四面体网格（tet）
    :return:
    """
    assert used_app in ['abaqus', 'ansys']
    assert mesh_type in ['hex', 'tet']
    if used_app == 'abaqus':
        export_to_inp_by_u_abaqus(filename, nodes, elements, boundary_nodes_idx, boundary_nodes_u, young_modulus, poisson_ratio, density, mesh_type)
    elif used_app == 'ansys':
        export_to_inp_by_u_ansys(filename, nodes, elements, boundary_nodes_idx, boundary_nodes_u, young_modulus, poisson_ratio, density, mesh_type)
    else:
        raise ValueError("Invalid used_app parameter!")

def export_to_inp_by_u_abaqus(filename, nodes, elements, boundary_nodes_idx, boundary_nodes_u, young_modulus, poisson_ratio, density, mesh_type='hex'):
    """
    齿轮专用的，相关信息导出为 Abaqus 的 inp 文件
    :param filename: 文件名
    :param nodes: 网格节点
    :param elements: 网格单元
    :param boundary_nodes_idx: 边界点索引
    :param boundary_nodes_u: 边界点位移
    :param young_modulus: 杨氏模量（GP）
    :param poisson_ratio: 泊松比
    :param density: 密度
    :param mesh_type: 网格类型，默认为六面体网格（hex），可选四面体网格（tet）
    :return:
    """
    with open(filename, 'w') as file:
        file.write("*Heading\n** Generated by Custom Export Script\n*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
        file.write("*Part, name=Gear\n")
        file.write("*Node\n")

        elements = elements+1
        boundary_nodes_idx = boundary_nodes_idx + 1
        # 写入节点信息
        for i, node in enumerate(nodes):
            file.write(f"{i+1}, {node[0]}, {node[1]}, {node[2]}\n")

        # 写入单元信息
        if mesh_type == 'hex':  # 六面体网格
            file.write("*Element, type=C3D8, elset=AllElements\n")
            for i, elem in enumerate(elements):
                file.write(
                    f"{i + 1}, {elem[0]}, {elem[1]}, {elem[2]}, {elem[3]}, {elem[4]}, {elem[5]}, {elem[6]}, {elem[7]}\n")
        elif mesh_type == 'tet':  # 四面体网格
            file.write("*Element, type=C3D4, elset=AllElements\n")
            for i, elem in enumerate(elements):
                file.write(f"{i + 1}, {elem[0]}, {elem[1]}, {elem[2]}, {elem[3]}\n")

        # 写入截面
        file.write("*Solid Section, elset=AllElements, material=Steel\n")
        file.write("*End Part\n**\n")

        # 定义装配和实例
        file.write("*Assembly, name=Assembly\n*Instance, name=GearInstance, part=Gear\n*End Instance\n")
        file.write("*End Assembly\n")

        # 写入材料信息
        file.write("*Material, name=Steel\n")
        file.write(f"*Density\n{density}\n")
        file.write(f"*Elastic\n{young_modulus}, {poisson_ratio}\n")

        # 写入步骤、边界条件和载荷
        file.write("** STEP: LoadStep\n*Step, name=LoadStep, nlgeom=NO\n*Static\n1., 1., 1e-05, 1.\n")

        # 固定边界条件
        file.write("*Boundary\n")
        for i, fixed_node in enumerate(boundary_nodes_idx):
            node_id = fixed_node
            u = boundary_nodes_u[i]
            file.write(f"GearInstance.{node_id}, 1, 1, {u[0]}\n")
            file.write(f"GearInstance.{node_id}, 2, 2, {u[1]}\n")
            file.write(f"GearInstance.{node_id}, 3, 3, {u[2]}\n")

        file.write("*Output, field, variable=PRESELECT\n")
        file.write("*Output, history, variable=PRESELECT\n")
        file.write("*End Step\n")
        file.write("** Output Global Stiffness Matrix\n")
        file.write("*Step, name=Global_Stiffness_Matrix\n")
        file.write("*MATRIX GENERATE, STIFFNESS, element by element\n")
        file.write("*MATRIX OUTPUT, STIFFNESS, FORMAT=COORDINATE\n")
        file.write("*End Step\n")

        print("Export to inp file successfully!")

def export_to_inp_by_u_ansys(filename, nodes, elements, boundary_nodes_idx, boundary_nodes_u, young_modulus, poisson_ratio, density, mesh_type='hex'):
    """
    齿轮专用的，相关信息导出为 Abaqus 的 inp 文件
    :param filename: 文件名
    :param nodes: 网格节点
    :param elements: 网格单元
    :param boundary_nodes_idx: 边界点索引
    :param boundary_nodes_u: 边界点位移
    :param young_modulus: 杨氏模量（GP）
    :param poisson_ratio: 泊松比
    :param density: 密度
    :param mesh_type: 网格类型，默认为六面体网格（hex），可选四面体网格（tet）
    :return:
    """
    with open(filename, 'w') as file:
        file.write("*Heading\n** Generated by Custom Export Script\n*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
        file.write("*Node\n")

        elements = elements+1
        boundary_nodes_idx = boundary_nodes_idx + 1
        # 写入节点信息
        for i, node in enumerate(nodes):
            file.write(f"{i+1}, {node[0]}, {node[1]}, {node[2]}\n")

        # 写入单元信息
        if mesh_type == 'hex':  # 六面体网格
            file.write("*Element, type=C3D8, elset=AllElements\n")
            for i, elem in enumerate(elements):
                file.write(
                    f"{i + 1}, {elem[0]}, {elem[1]}, {elem[2]}, {elem[3]}, {elem[4]}, {elem[5]}, {elem[6]}, {elem[7]}\n")
        elif mesh_type == 'tet':  # 四面体网格
            file.write("*Element, type=C3D4, elset=AllElements\n")
            for i, elem in enumerate(elements):
                file.write(f"{i + 1}, {elem[0]}, {elem[1]}, {elem[2]}, {elem[3]}\n")

        # 写入材料信息
        file.write("*Material, name=Steel\n")
        file.write(f"*Density\n{density}\n")
        file.write(f"*Elastic\n{young_modulus}, {poisson_ratio}\n")
        # 写入截面
        file.write("*Solid Section, elset=AllElements, material=Steel\n")
        file.write(",\n")
        file.write("*End Part\n**\n")
        file.write("*End Assembly\n")

        # 写入步骤、边界条件和载荷
        file.write("** STEP: LoadStep\n*Step, name=LoadStep, nlgeom=NO\n*Static\n1., 1., 1e-05, 1.\n")

        # 固定边界条件
        file.write("*Boundary\n")
        for i, fixed_node in enumerate(boundary_nodes_idx):
            node_id = fixed_node
            u = boundary_nodes_u[i]
            file.write(f"{node_id}, 1, 1, {u[0]}\n")
            file.write(f"{node_id}, 2, 2, {u[1]}\n")
            file.write(f"{node_id}, 3, 3, {u[2]}\n")

        file.write("*End Step\n")

        print("Export to inp file successfully!")

def face_normal_bilinear(cell_nodes, u, v, w, error=1e-3, is_normalize=True):
    """
    计算面外法线方向
    :param cell: 计算单元节点
    :param u: 参数
    :param v: 参数
    :param w: 参数
    :return:
    """
    P0 = cell_nodes[0]
    P1 = cell_nodes[1]
    P2 = cell_nodes[2]
    P3 = cell_nodes[3]
    P4 = cell_nodes[4]
    P5 = cell_nodes[5]
    P6 = cell_nodes[6]
    P7 = cell_nodes[7]

    if abs(u - 0) <= error*10:
        u = w
        P0 = P0
        P1 = P4
        P2 = P7
        P3 = P3
    elif abs(u - 1) <= error*10:
        u = w
        P0 = P1
        P1 = P2
        P2 = P6
        P3 = P5
    elif abs(v - 0) <= error*10:
        v = w
        P0 = P0
        P1 = P1
        P2 = P5
        P3 = P4
    elif abs(v - 1) <= error*10:
        v = w
        P0 = P3
        P1 = P7
        P2 = P6
        P3 = P2

    Pu = -(1 - v) * P0 + (1 - v) * P1 + v * P2 - v * P3
    Pv = -(1 - u) * P0 - u * P1 + u * P2 + (1 - u) * P3

    if is_normalize:
        normal = np.cross(Pu, Pv)
        normal = normal / np.linalg.norm(normal)
        return normal
    else:
        return np.cross(Pu, Pv)

# 给定参考法向计算某点的法向
def get_face_normal_with_reference(cell_node, reference_normal, error=1e-3, is_normalize=True):
    '''
    给定一组六面体单元，以及一个参考法向，计算该单元的六个面的法向，并返回与参考法向最接近的法向
    :param cell_node: 计算单元节点坐标
    :param reference_normal: 参考法向
    :param error: 误差限制
    :param is_normalize: 是否归一化
    '''
    NC = cell_node.shape[0]
    face_normal = np.zeros((NC, 3))
    parameters = np.array([
        [0.5, 0.5, 0],
        [0.5, 0.5, 1],
        [0, 0.5, 0.5],
        [1, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 1, 0.5]
    ])
    for i in range(NC):
        min_err = 100
        min_vec = np.zeros(3)
        for j in range(6):
            temp_normal = face_normal_bilinear(cell_node[i], parameters[j, 0], parameters[j, 1], parameters[j, 2], error, is_normalize)
            vec_err = np.sqrt(np.sum((temp_normal.reshape(-1) - reference_normal)**2))
            if vec_err < min_err:
                min_err = vec_err
                min_vec = temp_normal
        face_normal[i] = min_vec
    return face_normal
