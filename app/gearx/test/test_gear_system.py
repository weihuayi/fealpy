import pickle
import json
import numpy as np
from numpy import tan, arctan, sin, cos, pi, arctan2
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pytest

from fealpy.mesh import QuadrangleMesh, HexahedronMesh, IntervalMesh
from app.gearx.gear import ExternalGear, InternalGear
from app.gearx.utils import *


class TestGearSystem:
    def test_external_gear(self):
        with open('../data/external_gear_data.json', 'r') as file:
            data = json.load(file)
        m_n = data['mn']  # 法向模数
        z = data['z']  # 齿数
        alpha_n = data['alpha_n']  # 法向压力角
        beta = data['beta']  # 螺旋角
        x_n = data['xn']  # 法向变位系数
        hac = data['hac']  # 齿顶高系数
        cc = data['cc']  # 顶隙系数
        rcc = data['rcc']  # 刀尖圆弧半径
        jn = data['jn']  # 法向侧隙
        n1 = data['n1']  # 渐开线分段数
        n2 = data['n2']  # 过渡曲线分段数
        n3 = data['n3']
        na = data['na']
        nf = data['nf']
        nw = data['nw']
        tooth_width = data['tooth_width']
        inner_diam = data['inner_diam']  # 轮缘内径
        chamfer_dia = data['chamfer_dia']  # 倒角高度（直径）

        external_gear = ExternalGear(m_n, z, alpha_n, beta, x_n, hac, cc, rcc, jn, n1, n2, n3, na, nf, nw, chamfer_dia,
                                     inner_diam, tooth_width)

        quad_mesh = external_gear.generate_mesh()
        # quad_mesh.to_vtk(fname='external_quad_mesh.vtu')
        hex_mesh = external_gear.generate_hexahedron_mesh()
        # hex_mesh.to_vtk(fname='external_hex_mesh.vtu')
        #
        # with open('external_gear.pkl', 'wb') as f:
        #     pickle.dump({'quad_mesh': quad_mesh, 'gear': external_gear, 'hex_mesh': hex_mesh}, f)

        # node_from_cpp = np.loadtxt("../data/external_node.csv", delimiter=",")
        # cell_from_cpp = np.loadtxt("../data/external_cell.csv", delimiter=",", dtype=np.int64)
        #
        # quad_mesh_from_cpp = QuadrangleMesh(node_from_cpp, cell_from_cpp)
        # quad_mesh_from_cpp.to_vtk(fname='external_quad_mesh_cpp.vtu')

        # with open('../data/external_gear.pkl', 'wb') as f:
        #     pickle.dump({'external_gear': external_gear, 'hex_mesh': hex_mesh, 'quad_mesh': quad_mesh}, f)

    def test_internal_gear(self):
        with open('../data/internal_gear_data.json', 'r') as file:
            data = json.load(file)
        m_n = data['mn']  # 法向模数
        z = data['z']  # 齿数
        alpha_n = data['alpha_n']  # 法向压力角
        beta = data['beta']  # 螺旋角
        x_n = data['xn']  # 法向变位系数
        hac = data['hac']  # 齿顶高系数
        cc = data['cc']  # 顶隙系数
        rcc = data['rcc']  # 刀尖圆弧半径
        jn = data['jn']  # 法向侧隙
        n1 = data['n1']  # 渐开线分段数
        n2 = data['n2']  # 过渡曲线分段数
        n3 = data['n3']
        na = data['na']
        nf = data['nf']
        nw = data['nw']
        tooth_width = data['tooth_width']
        outer_diam = data['outer_diam']  # 轮缘内径
        z_cutter = data['z_cutter']
        xn_cutter = data['xn_cutter']

        internal_gear = InternalGear(m_n, z, alpha_n, beta, x_n, hac, cc, rcc, jn, n1, n2, n3, na, nf, nw, outer_diam,
                                     z_cutter,
                                     xn_cutter, tooth_width)

        t = np.array([0.1, 0.2, 25])
        p1 = internal_gear.get_involute_points(t)
        p1_dis = internal_gear.get_tip_intersection_points(t)

        alphawt = InternalGear.ainv(
            2 * (internal_gear.x_n - internal_gear.xn_cutter) * tan(internal_gear.alpha_n) / (
                    internal_gear.z - internal_gear.z_cutter) + (
                    tan(internal_gear.alpha_t) - internal_gear.alpha_t))
        E = 0.5 * (internal_gear.d - internal_gear.d_cutter) + internal_gear.m_t * (
                0.5 * (internal_gear.z - internal_gear.z_cutter) * (cos(internal_gear.alpha_t) / cos(alphawt) - 1))
        ratio = internal_gear.z / internal_gear.z_cutter
        p2 = internal_gear.get_transition_points(E, 0, 0, internal_gear.ra_cutter, ratio, t)
        p2_dis = internal_gear.get_transition_intersection_points(E, 0, 0, internal_gear.ra_cutter, ratio, t)

        # p = internal_gear.get_profile_points()

        quad_mesh = internal_gear.generate_mesh()
        r = internal_gear.r
        # hex_mesh = generate_hexahedral_mesh(quad_mesh, internal_gear.beta, r, tooth_width, nw)
        # hex_mesh.to_vtk(fname='internal_hex_mesh.vtu')
        # 读取 CSV 文件
        node_from_cpp = np.loadtxt("../data/internal_node.csv", delimiter=",")
        cell_from_cpp = np.loadtxt("../data/internal_cell.csv", delimiter=",", dtype=np.int64)

        quad_mesh_from_cpp = QuadrangleMesh(node_from_cpp, cell_from_cpp)
        quad_mesh_from_cpp.to_vtk(fname='internal_quad_mesh_cpp.vtu')

    def test_get_profile_node(self):
        with open('../data/external_gear.pkl', 'rb') as f:
            data = pickle.load(f)
        external_gear = data['external_gear']

        idx0, node0 = external_gear.get_profile_node(tooth_tag=0)
        idx1, node1 = external_gear.get_profile_node(tooth_tag=(0, 2, 3))
        idx2, node2 = external_gear.get_profile_node(tooth_tag=None)

        print(-1)

    def test_find_node_and_parameters(self):
        with open('../data/external_gear.pkl', 'rb') as f:
            data = pickle.load(f)
        external_gear = data['external_gear']
        hex_mesh = data['hex_mesh']
        quad_mesh = data['quad_mesh']

        n = 15
        helix_d = np.linspace(external_gear.d, external_gear.effective_da, n)
        helix_width = np.linspace(0, external_gear.tooth_width, n)
        helix_node = external_gear.cylindrical_to_cartesian(helix_d, helix_width)

        target_cell_idx = np.zeros(n, np.int32)
        local_face_idx = np.zeros(n, np.int32)
        parameters = np.zeros((n, 3), np.float64)
        for i, t_node in enumerate(helix_node):
            target_cell_idx[i], local_face_idx[i], parameters[i] = external_gear.find_node_location_kd_tree(t_node)

        print(-1)

    def test_get_one_tooth(self):
        with open('../data/external_gear_data.json', 'r') as file:
            data = json.load(file)
        m_n = data['mn']  # 法向模数
        z = data['z']  # 齿数
        alpha_n = data['alpha_n']  # 法向压力角
        beta = data['beta']  # 螺旋角
        x_n = data['xn']  # 法向变位系数
        hac = data['hac']  # 齿顶高系数
        cc = data['cc']  # 顶隙系数
        rcc = data['rcc']  # 刀尖圆弧半径
        jn = data['jn']  # 法向侧隙
        n1 = data['n1']  # 渐开线分段数
        n2 = data['n2']  # 过渡曲线分段数
        n3 = data['n3']
        na = data['na']
        nf = data['nf']
        nw = data['nw']
        tooth_width = data['tooth_width']
        inner_diam = data['inner_diam']  # 轮缘内径
        chamfer_dia = data['chamfer_dia']  # 倒角高度（直径）

        external_gear = ExternalGear(m_n, z, alpha_n, beta, x_n, hac, cc, rcc, jn, n1, n2, n3, na, nf, nw, chamfer_dia,
                                     inner_diam, tooth_width)

        hex_mesh = external_gear.generate_hexahedron_mesh()

        target_hex_mesh = external_gear.set_target_tooth([0, 1, 18])
        # target_hex_mesh.to_vtk(fname='../data/target_hex_mesh.vtu')

        n = 15
        helix_d = np.linspace(external_gear.d, external_gear.effective_da, n)
        helix_width = np.linspace(0, external_gear.tooth_width, n)
        helix_node = external_gear.cylindrical_to_cartesian(helix_d, helix_width)
        helix_cell = np.array([[i, i + 1] for i in range(n - 1)])
        i_mesh = IntervalMesh(helix_node, helix_cell)
        # i_mesh.to_vtk(fname='../data/target_interval_mesh.vtu')

        target_cell_idx = np.zeros(n, np.int32)
        local_face_idx = np.zeros(n, np.int32)
        parameters = np.zeros((n, 3), np.float64)
        for i, t_node in enumerate(helix_node):
            target_cell_idx[i], local_face_idx[i], parameters[i] = external_gear.find_node_location_kd_tree(t_node)

        node = target_hex_mesh.node
        # 寻找内圈上节点
        node_r = np.sqrt(node[:, 0] ** 2 + node[:, 1] ** 2)
        is_inner_node = np.abs(node_r - external_gear.inner_diam / 2) < 1e-11
        inner_node_idx = np.where(np.abs(node_r - external_gear.inner_diam / 2) < 1e-11)[0]

        with open('../data/external_gear_test_data.pkl', 'wb') as f:
            pickle.dump({'external_gear': external_gear, 'hex_mesh': target_hex_mesh, 'helix_node': helix_node,
                         'target_cell_idx': target_cell_idx, 'parameters': parameters,
                         'is_inner_node': is_inner_node, 'inner_node_idx': inner_node_idx}, f)

    def test_export_to_inp(self):
        with open('../data/external_gear_test_data.pkl', 'rb') as f:
            data = pickle.load(f)
        external_gear = data['external_gear']
        hex_mesh = data['hex_mesh']
        helix_node = data['helix_node']
        target_cell_idx = data['target_cell_idx']
        parameters = data['parameters']
        is_inner_node = data['is_inner_node']
        inner_node_idx = data['inner_node_idx']

        node = hex_mesh.node
        cell = hex_mesh.cell
        fixed_nodes = inner_node_idx
        load_nodes = np.array([454, 120, 121, 455, 2632, 2298, 2299, 2633, 2631,
                               2297, 2298, 2632, 4809, 4475, 4476, 4810, 4808, 4474,
                               4475, 4809, 6986, 6652, 6653, 6987, 6985, 6651, 6652,
                               6986, 9163, 8829, 8830, 9164, 9162, 8828, 8829, 9163,
                               11340, 11006, 11007, 11341, 11339, 11005, 11006, 11340, 13517,
                               13183, 13184, 13518, 13516, 13182, 13183, 13517, 15694, 15360,
                               15361, 15695, 17871, 17537, 17538, 17872, 20049, 19715, 19716,
                               20050, 20048, 19714, 19715, 20049, 22226, 21892, 21893, 22227,
                               22225, 21891, 21892, 22226, 24403, 24069, 24070, 24404, 24402,
                               24068, 24069, 24403, 26580, 26246, 26247, 26581, 26579, 26245,
                               26246, 26580, 28757, 28423, 28424, 28758, 28756, 28422, 28423,
                               28757, 30934, 30600, 30601, 30935, 30599, 30501, 30600, 30934,
                               32777, 32679, 32778, 33112, 32777, 32679, 32778, 33112, 34955,
                               34857, 34956, 35290])
        loads = np.array([[0., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [299.37136845, 299.37136845, 299.37136845],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [260.01223405, 260.01223405, 260.01223405],
                          [26.33068321, 26.33068321, 26.33068321],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [299.37136845, 299.37136845, 299.37136845],
                          [342.35133024, 342.35133024, 342.35133024],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [356.301615, 356.301615, 356.301615],
                          [41.34705476, 41.34705476, 41.34705476],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [342.35133024, 342.35133024, 342.35133024],
                          [403.59356272, 403.59356272, 403.59356272],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [460.91807739, 460.91807739, 460.91807739],
                          [34.05978846, 34.05978846, 34.05978846],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [403.59356272, 403.59356272, 403.59356272],
                          [513.60593762, 513.60593762, 513.60593762],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [400.30123444, 400.30123444, 400.30123444],
                          [28.9499708, 28.9499708, 28.9499708],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [513.60593762, 513.60593762, 513.60593762],
                          [667.75137915, 667.75137915, 667.75137915],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [351.14466476, 351.14466476, 351.14466476],
                          [32.53252752, 32.53252752, 32.53252752],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [667.75137915, 667.75137915, 667.75137915],
                          [828.37446848, 828.37446848, 828.37446848],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [258.41556042, 258.41556042, 258.41556042],
                          [58.92425681, 58.92425681, 58.92425681],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [828.37446848, 828.37446848, 828.37446848],
                          [9.82070947, 9.82070947, 9.82070947],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [135.89357625, 135.89357625, 135.89357625],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [256.67649633, 256.67649633, 256.67649633],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [966.27329305, 966.27329305, 966.27329305],
                          [29.90735348, 29.90735348, 29.90735348],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [256.67649633, 256.67649633, 256.67649633],
                          [425.76910538, 425.76910538, 425.76910538],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [797.69873627, 797.69873627, 797.69873627],
                          [93.67501549, 93.67501549, 93.67501549],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [425.76910538, 425.76910538, 425.76910538],
                          [557.24179034, 557.24179034, 557.24179034],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [615.81246126, 615.81246126, 615.81246126],
                          [204.08860554, 204.08860554, 204.08860554],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [557.24179034, 557.24179034, 557.24179034],
                          [622.73987876, 622.73987876, 622.73987876],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [430.73900214, 430.73900214, 430.73900214],
                          [375.09254767, 375.09254767, 375.09254767],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [622.73987876, 622.73987876, 622.73987876],
                          [591.74426803, 591.74426803, 591.74426803],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [262.96630353, 262.96630353, 262.96630353],
                          [611.00371415, 611.00371415, 611.00371415],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [591.74426803, 591.74426803, 591.74426803],
                          [439.89833767, 439.89833767, 439.89833767],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [124.16994291, 124.16994291, 124.16994291],
                          [918.78886228, 918.78886228, 918.78886228],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [439.89833767, 439.89833767, 439.89833767],
                          [153.13147705, 153.13147705, 153.13147705],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [1342.58280867, 1342.58280867, 1342.58280867],
                          [153.13147705, 153.13147705, 153.13147705],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [1342.58280867, 1342.58280867, 1342.58280867],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 0.],
                          [0., 0., 0.]])
        young_modulus = 206e3
        poisson_ratio = 0.3
        density = 7850

        export_to_inp('../data/external_gear_test.inp', node, cell, fixed_nodes, load_nodes, loads, young_modulus,
                      poisson_ratio, density)


if __name__ == "__main__":
    pytest.main(["./test_gear_system.py", "-k", "test_external_gear"])
