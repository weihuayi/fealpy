import pickle
import json
import numpy as np
import pandas as pd
from numpy import tan, arctan, sin, cos, pi, arctan2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
import pytest

from fealpy.mesh import QuadrangleMesh, HexahedronMesh, IntervalMesh, TetrahedronMesh
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
        # m_n = 2.25
        # z = 18
        # alpha_n = 17.5
        # beta = 30
        # x_n = 0.4
        # hac = 1
        # cc = 1.4
        # inner_diam = 10
        chamfer_dia = data['chamfer_dia']  # 倒角高度（直径）

        external_gear = ExternalGear(m_n, z, alpha_n, beta, x_n, hac, cc, rcc, jn, n1, n2, n3, na, nf, nw, chamfer_dia,
                                     inner_diam, tooth_width)

        quad_mesh = external_gear.generate_mesh()
        # quad_mesh.to_vtk(fname='../data/external_quad_mesh_new.vtu')
        hex_mesh = external_gear.generate_hexahedron_mesh()
        profile_node_idx_right, profile_node_idx_left = external_gear.get_profile_node_index(tooth_tag=0)[0]
        profile_node_idx_right = profile_node_idx_right.reshape(-1)
        profile_node_idx_left = profile_node_idx_left.reshape(-1)
        profile_node_normal_right, profile_node_normal_left = external_gear.get_profile_node_normal_with_tooth(tooth_tag=0)
        node_data = np.zeros((hex_mesh.number_of_nodes(), 3))
        node_data[profile_node_idx_right] = profile_node_normal_right
        node_data[profile_node_idx_left] = profile_node_normal_left
        hex_mesh.nodedata['normal'] = node_data
        hex_mesh.to_vtk(fname='../data/external_hex_mesh_new.vtu')
        #
        # with open('external_gear.pkl', 'wb') as f:
        #     pickle.dump({'quad_mesh': quad_mesh, 'gear': external_gear, 'hex_mesh': hex_mesh}, f)

        # node_from_cpp = np.loadtxt("../data/external_node.csv", delimiter=",")
        # cell_from_cpp = np.loadtxt("../data/external_cell.csv", delimiter=",", dtype=np.int64)
        #
        # quad_mesh_from_cpp = QuadrangleMesh(node_from_cpp, cell_from_cpp)
        # quad_mesh_from_cpp.to_vtk(fname='external_quad_mesh_cpp.vtu')

        # with open('../data/external_gear_new.pkl', 'wb') as f:
        #     pickle.dump({'external_gear': external_gear, 'hex_mesh': hex_mesh, 'quad_mesh': quad_mesh}, f)

    def test_internal_gear(self):
        with open('../data/internal_gear_data_new.json', 'r') as file:
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
        # z = 70

        internal_gear = InternalGear(m_n, z, alpha_n, beta, x_n, hac, cc, rcc, jn, n1, n2, n3, na, nf, nw, outer_diam, z_cutter,
                                     xn_cutter, tooth_width)

        quad_mesh = internal_gear.mesh
        hex_mesh = internal_gear.generate_hexahedron_mesh()
        profile_node_idx_right, profile_node_idx_left = internal_gear.get_profile_node_index(tooth_tag=0)[0]
        profile_node_idx_right = profile_node_idx_right.reshape(-1)
        profile_node_idx_left = profile_node_idx_left.reshape(-1)
        profile_node_normal_right, profile_node_normal_left = internal_gear.get_profile_node_normal_with_tooth(
            tooth_tag=0)
        node_data = np.zeros((hex_mesh.number_of_nodes(), 3))
        node_data[profile_node_idx_right] = profile_node_normal_right
        node_data[profile_node_idx_left] = profile_node_normal_left
        hex_mesh.nodedata['normal'] = node_data
        hex_mesh.to_vtk(fname='../data/test_internal_hex_mesh_new.vtu')

        # with open('../data/internal_gear_new.pkl', 'wb') as f:
        #     pickle.dump({'internal_gear': internal_gear, 'hex_mesh': hex_mesh, 'quad_mesh': quad_mesh}, f)
        #
        # t = np.array([0.1, 0.2, 25])
        # p1 = internal_gear.get_involute_points(t)
        # p1_dis = internal_gear.get_tip_intersection_points(t)
        #
        # alphawt = InternalGear.ainv(
        #     2 * (internal_gear.x_n - internal_gear.xn_cutter) * tan(internal_gear.alpha_n) / (internal_gear.z - internal_gear.z_cutter) + (
        #             tan(internal_gear.alpha_t) - internal_gear.alpha_t))
        # E = 0.5 * (internal_gear.d - internal_gear.d_cutter) + internal_gear.m_t * (
        #         0.5 * (internal_gear.z - internal_gear.z_cutter) * (cos(internal_gear.alpha_t) / cos(alphawt) - 1))
        # ratio = internal_gear.z / internal_gear.z_cutter
        # p2 = internal_gear.get_transition_points(E, 0, 0, internal_gear.ra_cutter, ratio, t)
        # p2_dis = internal_gear.get_transition_intersection_points(E, 0, 0, internal_gear.ra_cutter, ratio, t)
        #
        # # p = internal_gear.get_profile_points()
        #
        # # quad_mesh = internal_gear.generate_mesh()
        # # quad_mesh.to_vtk(fname='../data/internal_quad_mesh.vtu')
        # # r = internal_gear.r
        # # # hex_mesh = generate_hexahedral_mesh(quad_mesh, internal_gear.beta, r, tooth_width, nw)
        # # # hex_mesh.to_vtk(fname='internal_hex_mesh.vtu')
        # # # 读取 CSV 文件
        # # node_from_cpp = np.loadtxt("../data/internal_node.csv", delimiter=",")
        # # cell_from_cpp = np.loadtxt("../data/internal_cell.csv", delimiter=",", dtype=np.int64)
        # #
        # # quad_mesh_from_cpp = QuadrangleMesh(node_from_cpp, cell_from_cpp)
        # # quad_mesh_from_cpp.to_vtk(fname='internal_quad_mesh_cpp.vtu')

    def test_get_profile_node(self):
        with open('../data/external_gear.pkl', 'rb') as f:
            data = pickle.load(f)
        external_gear = data['external_gear']

        idx0, node0 = external_gear.get_profile_node_index(tooth_tag=0)
        # idx1, node1 = external_gear.get_profile_node_index(tooth_tag=(0, 2, 3))
        # idx2, node2 = external_gear.get_profile_node_index(tooth_tag=None)
        idx_right = idx0[0][0, ...]
        idx_left = idx0[1][0, ...]
        node_right = node0[0][0, ...]
        node_left = node0[1][0, ...]

        num_col = idx_right.shape[0]
        num_row = idx_right.shape[1]
        num_cell = (num_col-1) * (num_row-1)
        num_node = num_col * num_row
        # profile_mesh_node = np.concatenate([node_right.reshape(-1, 3), node_left.reshape(-1, 3)], axis=0)
        # profile_mesh_cell = np.zeros((num_cell*2, 4), dtype=np.int32)
        # temp_idx_right = np.arange(num_node).reshape(num_col, num_row)
        # temp_idx_left = temp_idx_right + num_node
        # profile_mesh_cell[0:num_cell, 0] = temp_idx_right[0:num_col-1, 0:num_row-1].reshape(-1)
        # profile_mesh_cell[0:num_cell, 1] = temp_idx_right[1:num_col, 0:num_row-1].reshape(-1)
        # profile_mesh_cell[0:num_cell, 2] = temp_idx_right[1:num_col, 1:num_row].reshape(-1)
        # profile_mesh_cell[0:num_cell, 3] = temp_idx_right[0:num_col-1, 1:num_row].reshape(-1)
        #
        # profile_mesh_cell[num_cell:, 0] = temp_idx_left[0:num_col-1, 0:num_row-1].reshape(-1)
        # profile_mesh_cell[num_cell:, 1] = temp_idx_left[1:num_col, 0:num_row-1].reshape(-1)
        # profile_mesh_cell[num_cell:, 2] = temp_idx_left[1:num_col, 1:num_row].reshape(-1)
        # profile_mesh_cell[num_cell:, 3] = temp_idx_left[0:num_col-1, 1:num_row].reshape(-1)

        profile_mesh_node = node_left.reshape(-1, 3)
        profile_mesh_cell = np.zeros((num_cell, 4), dtype=np.int32)
        temp_idx_right = np.arange(num_node).reshape(num_col, num_row)
        # temp_idx_left = temp_idx_right + num_node
        profile_mesh_cell[0:num_cell, 0] = temp_idx_right[0:num_col-1, 0:num_row-1].reshape(-1)
        profile_mesh_cell[0:num_cell, 1] = temp_idx_right[1:num_col, 0:num_row-1].reshape(-1)
        profile_mesh_cell[0:num_cell, 2] = temp_idx_right[1:num_col, 1:num_row].reshape(-1)
        profile_mesh_cell[0:num_cell, 3] = temp_idx_right[0:num_col-1, 1:num_row].reshape(-1)

        profile_mesh = QuadrangleMesh(profile_mesh_node, profile_mesh_cell)
        np.savetxt('/home/concha/openfinite3/app/gearx/data/profile_mesh_nodes.txt',
                   np.column_stack((np.arange(profile_mesh.node.shape[0]), profile_mesh.node)),
                   delimiter=',', fmt='%d,%.6f,%.6f,%.6f')
        np.savetxt('/home/concha/openfinite3/app/gearx/data/profile_mesh_cells.txt',
                   np.column_stack((np.arange(profile_mesh.cell.shape[0]), profile_mesh.cell)),
                   delimiter=',', fmt='%d,%d,%d,%d,%d')
        profile_mesh.to_vtk(fname="/home/concha/openfinite3/app/gearx/data/one_tooth_external_gear_quad_mesh_left.vtu")

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
        # i_mesh.to_vtk('../data/interval_mesh.vtu')

        target_cell_idx = np.zeros(n, np.int32)
        face_normal = np.zeros((n, 3), np.float64)
        parameters = np.zeros((n, 3), np.float64)
        for i, t_node in enumerate(helix_node):
            target_cell_idx[i], face_normal[i], parameters[i] = external_gear.find_node_location_kd_tree(t_node)

        # print(target_cell_idx)
        print(face_normal)
        # 法向量后处理
        # 计算平均法向量
        average_normal = np.mean(face_normal, axis=0)
        average_normal /= np.linalg.norm(average_normal)

        threshold = 0.1
        for i in range(len(face_normal)):
            deviation = np.linalg.norm(face_normal[i] - average_normal)
            if deviation > threshold:
                face_normal[i] = average_normal
        print((face_normal,))
        # print((parameters,))
        print(-1)

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
        ansys_uh = np.loadtxt('D:/ansys/gear_15_load_u.txt', skiprows=1)[:, 1]
        target_hex_mesh.nodedata['ansys_u'] = ansys_uh
        target_hex_mesh.to_vtk(fname='D:/ansys/gear_15_load_u.vtu')

        # n = 15
        # helix_d = np.linspace(external_gear.d, external_gear.effective_da, n)
        # helix_width = np.linspace(0, external_gear.tooth_width, n)
        # helix_node = external_gear.cylindrical_to_cartesian(helix_d, helix_width)
        # helix_cell = np.array([[i, i + 1] for i in range(n - 1)])
        # i_mesh = IntervalMesh(helix_node, helix_cell)
        # # i_mesh.to_vtk(fname='../data/target_interval_mesh.vtu')
        #
        # target_cell_idx = np.zeros(n, np.int32)
        # local_face_idx = np.zeros(n, np.int32)
        # parameters = np.zeros((n, 3), np.float64)
        # for i, t_node in enumerate(helix_node):
        #     target_cell_idx[i], local_face_idx[i], parameters[i] = external_gear.find_node_location_kd_tree(t_node)
        #
        # node = target_hex_mesh.node
        # # 寻找内圈上节点
        # node_r = np.sqrt(node[:, 0] ** 2 + node[:, 1] ** 2)
        # is_inner_node = np.abs(node_r - external_gear.inner_diam / 2) < 1e-11
        # inner_node_idx = np.where(np.abs(node_r - external_gear.inner_diam / 2)<1e-11)[0]

        # with open('../data/external_gear_test_data.pkl', 'wb') as f:
        #     pickle.dump({'external_gear': external_gear, 'hex_mesh': target_hex_mesh, 'helix_node': helix_node,
        #                  'target_cell_idx': target_cell_idx, 'parameters': parameters,
        #                  'is_inner_node': is_inner_node, 'inner_node_idx': inner_node_idx}, f)

    def test_export_to_inp(self):
        with open('../data/external_gear_data_part.pkl', 'rb') as f:
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
        load_nodes = cell[target_cell_idx[0]]
        loads = np.array([[-10, -1., 0],
                          [-10, -2., 0],
                          [-10, -3., 0],
                          [-10, -4., 0],
                          [-10, -5., 0],
                          [-10, -6., 0],
                          [-10, -7., 0],
                          [-10, -8., 0]])
        young_modulus = 206e9
        poisson_ratio = 0.3
        density = 7.85e-09

        export_to_inp('../data/external_gear_test.inp', node, cell, fixed_nodes, load_nodes, loads, young_modulus, poisson_ratio, density)
        export_to_inp('../data/external_gear_test_ansys.inp', node, cell, fixed_nodes, load_nodes, loads, young_modulus, poisson_ratio, density, used_app='ansys')

    def test_export_to_inp_by_u(self):
        young_modulus = 206e9
        poisson_ratio = 0.3
        density = 7.85e-09
        mesh = HexahedronMesh.from_seven_hex_cube()
        node = mesh.node
        # 四面体网格
        nx, ny, nz = 2, 2, 2
        tet_mesh = TetrahedronMesh.from_box(nx=nx, ny=ny, nz=nz)

        u = lambda p: (10e-3*(2*p[..., 0]+p[..., 1]+p[..., 2])/2).reshape(-1, 1)
        v = lambda p: (10e-3*(p[..., 0]+2*p[..., 1]+p[..., 2])/2).reshape(-1, 1)
        w = lambda p: (10e-3*(p[..., 0]+p[..., 1]+2*p[..., 2])/2).reshape(-1, 1)

        boundary_nodes_idx = np.array([8, 9, 10, 11, 12, 13, 14, 15])
        boundary_nodes = node[boundary_nodes_idx]
        boundary_nodes_u = np.concatenate([u(boundary_nodes), v(boundary_nodes), w(boundary_nodes)], axis=1)

        export_to_inp_by_u('../data/seven_hex_by_u.inp', node, mesh.cell, boundary_nodes_idx, boundary_nodes_u,
                           young_modulus, poisson_ratio, density)
        export_to_inp_by_u('../data/seven_hex_ansys_by_u.inp', node, mesh.cell, boundary_nodes_idx, boundary_nodes_u,
                           young_modulus, poisson_ratio, density, used_app='ansys')

        # 四面体网格导出文件测试
        tet_boundary_nodes_idx = tet_mesh.boundary_node_index()
        tet_boundary_nodes = tet_mesh.node[tet_boundary_nodes_idx]
        tet_boundary_nodes_u = np.concatenate([u(tet_boundary_nodes), v(tet_boundary_nodes), w(tet_boundary_nodes)], axis=1)
        export_to_inp_by_u('../data/box_tet_by_u.inp', tet_mesh.node, tet_mesh.cell, tet_boundary_nodes_idx, tet_boundary_nodes_u,
                           young_modulus, poisson_ratio, density, mesh_type='tet')
        export_to_inp_by_u('../data/box_tet_ansys_by_u.inp', tet_mesh.node, tet_mesh.cell, tet_boundary_nodes_idx, tet_boundary_nodes_u,
                           young_modulus, poisson_ratio, density, used_app='ansys', mesh_type='tet')


    def test_face_normal(self):
        import matplotlib.pyplot as plt

        tesh_mesh = HexahedronMesh.from_box(nx=1, ny=1, nz=1)
        normal = tesh_mesh.cell_normal()
        print(normal)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        tesh_mesh.add_plot(ax)
        tesh_mesh.find_face(ax, showindex=True)
        plt.show()


    def test_get_part_internal_gear(self):
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

        hex_mesh = internal_gear.generate_hexahedron_mesh()
        target_hex_mesh = internal_gear.set_target_tooth([0, 1, 2, 78, 79])
        part_internal_gear_mesh = internal_gear.set_target_tooth([0, 1, 2, 78, 79])
        part_internal_gear_mesh_wheel = internal_gear.set_target_tooth([0, 1, 2, 78, 79], get_wheel=True)

        part_internal_gear_mesh.to_vtk(fname='../data/part_internal_gear_mesh.vtu')
        part_internal_gear_mesh_wheel.to_vtk(fname='../data/part_internal_gear_hex_mesh_wheel_new.vtu')

    def test_get_part_external_gear(self):
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
        part_external_gear_mesh = external_gear.set_target_tooth([18, 0, 1, 2, 3, 4])
        part_external_gear_mesh_wheel = external_gear.set_target_tooth([18, 0, 1, 2, 3, 4], get_wheel=True)

        part_external_gear_mesh.to_vtk(fname='../data/part_external_gear_mesh.vtu')
        part_external_gear_mesh_wheel.to_vtk(fname='../data/part_external_gear_hex_mesh_wheel.vtu')

    def test_get_profile_face_normal_external(self):
        with open('../data/external_gear_new.pkl', 'rb') as f:
            data = pickle.load(f)
        external_gear = data['external_gear']
        hex_mesh = data['hex_mesh']
        quad_mesh = data['quad_mesh']

        mesh = external_gear.set_target_tooth([0, 1, 18])
        # external_gear.target_quad_mesh.to_vtk(fname='../data/face_normal_target_quad_mesh.vtu')

        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()
        print(f"NC: {NC}")
        NN = mesh.number_of_nodes()
        print(f"NN: {NN}")
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        print(f"cell: {cell}")
        print(f"node: {node}")

        # 齿面上的节点索引和坐标
        node_indices_tuple, noe_coord_tuple, profile_node_normal = external_gear.get_profile_node_index(tooth_tag=0)

        print(-1)

    def test_get_profile_face_normal_internal(self):
        with open('../data/internal_gear_new.pkl', 'rb') as f:
            data = pickle.load(f)
        internal_gear = data['internal_gear']
        hex_mesh = data['hex_mesh']
        quad_mesh = data['quad_mesh']

        mesh = internal_gear.set_target_tooth([78, 79, 0, 1, 2])
        # internal_gear.target_quad_mesh.to_vtk(fname='../data/face_normal_target_quad_mesh_internal.vtu')

        # 齿面上的节点索引和坐标
        node_indices_tuple, noe_coord_tuple, profile_node_normal = internal_gear.get_profile_node_index(tooth_tag=0)

        print(-1)



if __name__ == "__main__":
    pytest.main(["./test_gear_system.py", "-k", "test_external_gear"])