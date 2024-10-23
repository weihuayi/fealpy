import pickle
import json
import numpy as np
from numpy import tan, arctan, sin, cos, pi, arctan2
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pytest

from fealpy.mesh import QuadrangleMesh, HexahedronMesh
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
        hex_mesh = generate_hexahedral_mesh(mesh, beta, r, tooth_width, nw)
        volume_node = hex_mesh.node
        volume_cell = hex_mesh.cell
        NN = hex_mesh.number_of_nodes()
        NC = hex_mesh.number_of_cells()
        print(NN)
        print(NC)

    def test_cylindrical_to_cartesian_and_find_node_location_kd_tree(self):
        with open('../data/external_gear.pkl', 'rb') as f:
            data = pickle.load(f)
        external_gear = data['gear']
        hex_mesh = data['hex_mesh']
        n = 15
        helix_d = np.linspace(external_gear.d, external_gear.effective_da, n)
        helix_width = np.linspace(0, external_gear.tooth_width, n)
        helix_node = cylindrical_to_cartesian(helix_d, helix_width, external_gear)
        print(helix_node)

        for t_node in helix_node:
            target_cell_idx = find_node_location_kd_tree(t_node, hex_mesh)
            print(target_cell_idx)


if __name__ == "__main__":
    pytest.main(["./test_gear_utils.py", "-k", "test_get_helix_points_and_sweep_points"])