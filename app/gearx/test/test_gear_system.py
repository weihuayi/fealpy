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

        r = external_gear.r
        hex_mesh = generate_hexahedral_mesh(quad_mesh, external_gear.beta, r, tooth_width, nw)

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
        quad_mesh = internal_gear.generate_mesh()
        r = internal_gear.r
        hex_mesh = generate_hexahedral_mesh(quad_mesh, internal_gear.beta, r, tooth_width, nw)
        # hex_mesh.to_vtk(fname='internal_hex_mesh.vtu')



if __name__ == "__main__":
    pytest.main(["./test_gear_system.py", "-k", "test_external_gear"])