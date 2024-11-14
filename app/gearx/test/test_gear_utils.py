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


class TestGearUtils:
    def test_get_helix_points_and_sweep_points(self):
        # 创建数据
        a = 1
        b = 1.5
        beta = 10
        beta = beta / 180 * np.pi if beta > np.pi / 2 else beta
        r = 1
        h = 0.5
        n = 8

        phi = np.linspace(0, -np.pi / 2, 10).reshape((-1, 1))
        points = np.concatenate([a * np.cos(phi), b * np.sin(phi), np.zeros((len(phi), 1))], axis=-1)

        volume_points = sweep_points(points, beta, r, h, n)
        print(-1)

        # # 绘制图像
        # # 创建3D图形对象
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # # 绘制曲面图
        # for lay in volume_points:
        #     ax.scatter(lay[:, 0], lay[:, 1], lay[:, 2], c='b', marker='o')
        #
        # X = volume_points[..., 0]
        # Y = volume_points[..., 1]
        # Z = volume_points[..., 2]
        # ax.plot_surface(X, Y, Z, cmap='rainbow')
        # # 设置坐标轴标签
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        #
        # plt.axis('equal')
        # plt.show()

    def test_generate_hexahedral_mesh(self):
        # 读取齿轮端面网格数据
        with open('../data/external_gear.pkl', 'rb') as f:
            data = pickle.load(f)
        mesh = data['mesh']
        gear = data['gear']
        beta = gear.beta
        r = gear.r
        # 读取external_gear_data.json文件中的数据
        with open('../data/external_gear_data.json', 'r') as f:
            gear_data = json.load(f)
        tooth_width = gear_data['tooth_width']
        nw = gear_data['nw']

        # 生成六面体网格
        hex_mesh = generate_hexahedron_mesh(mesh, beta, r, tooth_width, nw)
        volume_node = hex_mesh.node
        volume_cell = hex_mesh.cell
        NN = hex_mesh.number_of_nodes()
        NC = hex_mesh.number_of_cells()
        print(NN)
        print(NC)

    def test_generate_hexahedral_mesh2(self):
        node = np.array([[0.0, 0.0],
                         [0.0, 1.0],
                         [0.0, 2.0],
                         [1.0, 0.0],
                         [1.0, 1.0],
                         [1.0, 2.0],
                         [2.0, 0.0],
                         [2.0, 1.0],
                         [2.0, 2.0]])

        cell = np.array([[0, 3, 4, 1],
                         [1, 4, 5, 2],
                         [3, 6, 7, 4],
                         [4, 7, 8, 5]])

        quad_mesh = QuadrangleMesh(node, cell)
        quad_mesh.celldata['cell_domain_tag'] = np.array([1, 2, 3, 4])
        quad_mesh.celldata['cell_tooth_tag'] = np.array([1, 1, 2, 2])

        beta = 0.2617993877991494
        r = 49.17561856947894
        tooth_width = 36.0
        nw = 16
        hex_mesh = generate_hexahedron_mesh(quad_mesh, beta, r, tooth_width, nw)


        print(-1)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # hex_mesh.add_plot(ax)
        # hex_mesh.find_node(ax, showindex=True)
        # hex_mesh.find_cell(ax, showindex=True)
        # plt.show()

    def test_cylindrical_to_cartesian_and_find_node_location_kd_tree(self):
        with open('../data/external_gear.pkl', 'rb') as f:
            data = pickle.load(f)
        external_gear = data['external_gear']
        hex_mesh = data['hex_mesh']
        quad_mesh = data['quad_mesh']
        # quad_mesh.to_vtk(fname='external_quad_mesh.vtu')
        node = hex_mesh.node
        face = hex_mesh.face
        n = 15
        helix_d = np.linspace(external_gear.d, external_gear.effective_da, n)
        helix_width = np.linspace(0, external_gear.tooth_width, n)
        helix_node = cylindrical_to_cartesian(helix_d, helix_width, external_gear)
        # hex_mesh.to_vtk(fname='../data/external_hex_mesh.vtu')
        # helix_cell = np.array([[i, i + 1] for i in range(n - 1)])
        # i_mesh = IntervalMesh(helix_node, helix_cell)
        # i_mesh.to_vtk(fname='../data/interval_mesh.vtu')

        target_cell_idx = np.zeros(n, np.int32)
        local_face_idx = np.zeros(n, np.int32)
        parameters = np.zeros((n, 3), np.float64)
        for i, t_node in enumerate(helix_node):
            target_cell_idx[i], local_face_idx[i], parameters[i] = find_node_location_kd_tree(t_node, external_gear, hex_mesh)
        print(target_cell_idx)
        print(local_face_idx)
        print(parameters)

        # 寻找内圈上节点
        node_r = np.sqrt(node[:, 0] ** 2 + node[:, 1] ** 2)
        is_inner_node = np.abs(node_r - external_gear.inner_diam / 2) < 1e-11
        inner_node_idx = np.where(np.abs(node_r - external_gear.inner_diam / 2)<1e-11)[0]

        with open('../data/external_gear_test_data.pkl', 'wb') as f:
            pickle.dump({'external_gear': external_gear, 'hex_mesh': hex_mesh, 'quad_mesh': quad_mesh,
                         'helix_node': helix_node, 'target_cell_idx': target_cell_idx,
                         'parameters': parameters, 'is_inner_node': is_inner_node}, f)

if __name__ == "__main__":
    pytest.main(["./test_gear_utils.py", "-k", "test_get_helix_points_and_sweep_points"])