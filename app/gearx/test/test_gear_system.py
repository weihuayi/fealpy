import pickle
import json
import numpy as np
import pandas as pd
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
        quad_mesh.to_vtk(fname='../data/external_quad_mesh.vtu')
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

        internal_gear = InternalGear(m_n, z, alpha_n, beta, x_n, hac, cc, rcc, jn, n1, n2, n3, na, nf, nw, outer_diam, z_cutter,
                                     xn_cutter, tooth_width)

        t = np.array([0.1, 0.2, 25])
        p1 = internal_gear.get_involute_points(t)
        p1_dis = internal_gear.get_tip_intersection_points(t)

        alphawt = InternalGear.ainv(
            2 * (internal_gear.x_n - internal_gear.xn_cutter) * tan(internal_gear.alpha_n) / (internal_gear.z - internal_gear.z_cutter) + (
                    tan(internal_gear.alpha_t) - internal_gear.alpha_t))
        E = 0.5 * (internal_gear.d - internal_gear.d_cutter) + internal_gear.m_t * (
                0.5 * (internal_gear.z - internal_gear.z_cutter) * (cos(internal_gear.alpha_t) / cos(alphawt) - 1))
        ratio = internal_gear.z / internal_gear.z_cutter
        p2 = internal_gear.get_transition_points(E, 0, 0, internal_gear.ra_cutter, ratio, t)
        p2_dis = internal_gear.get_transition_intersection_points(E, 0, 0, internal_gear.ra_cutter, ratio, t)

        # p = internal_gear.get_profile_points()

        quad_mesh = internal_gear.generate_mesh()
        quad_mesh.to_vtk(fname='../data/internal_quad_mesh.vtu')
        # r = internal_gear.r
        # # hex_mesh = generate_hexahedral_mesh(quad_mesh, internal_gear.beta, r, tooth_width, nw)
        # # hex_mesh.to_vtk(fname='internal_hex_mesh.vtu')
        # # 读取 CSV 文件
        # node_from_cpp = np.loadtxt("../data/internal_node.csv", delimiter=",")
        # cell_from_cpp = np.loadtxt("../data/internal_cell.csv", delimiter=",", dtype=np.int64)
        #
        # quad_mesh_from_cpp = QuadrangleMesh(node_from_cpp, cell_from_cpp)
        # quad_mesh_from_cpp.to_vtk(fname='internal_quad_mesh_cpp.vtu')

    def test_get_profile_node(self):
        with open('../data/external_gear.pkl', 'rb') as f:
            data = pickle.load(f)
        external_gear = data['external_gear']

        idx0, node0 = external_gear.get_profile_node_index(tooth_tag=0)
        idx1, node1 = external_gear.get_profile_node_index(tooth_tag=(0, 2, 3))
        idx2, node2 = external_gear.get_profile_node_index(tooth_tag=None)

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

        helix_cell = np.array([[i, i + 1] for i in range(n - 1)])
        i_mesh = IntervalMesh(helix_node, helix_cell)
        i_mesh.to_vtk('../data/interval_mesh.vtu')

        # target_cell_idx = np.zeros(n, np.int32)
        # local_face_idx = np.zeros(n, np.int32)
        # parameters = np.zeros((n, 3), np.float64)
        # for i, t_node in enumerate(helix_node):
        #     target_cell_idx[i], local_face_idx[i], parameters[i] = external_gear.find_node_location_kd_tree(t_node)
        # print(target_cell_idx)
        # print(parameters)
        # print(-1)

    def test_data(self):
        def read_csv_to_numpy(file_path):
            # 读取 CSV 文件，跳过第一行（i），从第二行开始读取
            df = pd.read_csv(file_path, skiprows=1, header=None)

            # 将 DataFrame 转换为 NumPy 数组
            return df.to_numpy()

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
        hex_mesh = external_gear.generate_hexahedron_mesh()

        node = hex_mesh.node
        cell = hex_mesh.cell

        node_cpp = read_csv_to_numpy("../data/hex_node.csv")
        cell_cpp = read_csv_to_numpy("../data/hex_cell.csv")

        print(-1)

        np.testing.assert_array_equal(cell_cpp, cell)
        np.testing.assert_allclose(node_cpp, node, atol=1e-3)

        print(-1)



if __name__ == "__main__":
    pytest.main(["./test_gear_system.py", "-k", "test_external_gear"])