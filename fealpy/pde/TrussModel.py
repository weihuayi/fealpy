#!/usr/bin/env python3
# 
import numpy as np

from fealpy.decorator  import cartesian 
from TrussMesh import TrussMesh

class Truss_3d():
    def __init__(self):
        self.A = 2000 # 横截面积 mm^2
        self.E = 1500 # 弹性模量 ton/mm^2

    def init_mesh(self, n=1):
        d1 = 952.5 # 单位 mm
        d2 = 2540
        h1 = 5080
        h2 = 2540
        node = np.array([
            [-d1, 0, h1], [d1, 0, h1], [-d1, d1, h2], [d1, d1, h2],
            [d1, -d1, h2], [-d1, -d1, h2], [-d2, d2, 0], [d2, d2, 0],
            [d2, -d2, 0], [-d2, -d2, 0]], dtype=np.float64)
        edge = np.array([
            [0, 1], [3, 0], [1, 2], [1, 5], [0, 4], 
            [1, 3], [1, 4], [0, 2], [0, 5], [2, 5],
            [4, 3], [2, 3], [4, 5], [2, 9], [6, 5], 
            [8, 3], [7, 4], [6, 3], [2, 7], [9, 4],
            [8, 5], [9, 5], [2, 6], [7, 3], [8, 4]], dtype=np.int_)
        mesh = TrussMesh(node, edge)
        return mesh

    @cartesian
    def displacement(self, p):
        pass

    @cartesian
    def jacobian(self, p):
        pass

    @cartesian
    def strain(self, p):
        pass

    @cartesian
    def stress(self, p):
        pass

    @cartesian
    def source(self, p):
        shape = len(p.shape[:-1])*(1,) + (-1, )
        val = np.zeros(shape, dtype=np.float_)
        return val 

    @cartesian
    def force(self):
        '''
        施加 (0, 900, 0) 的力，即平行于 y 轴方向大小为 900N 的力
        '''
        val = np.array([0, 900, 0])
        return val

    def is_force_boundary(self, p):
        '''
        对第 0，1 号节点施加力
        '''
        return np.abs(p[..., 2]) == 5080

    @cartesian
    def dirichlet(self, p):
        shape = len(p.shape)*(1, )
        val = np.array([0.0])
        return val.reshape(shape)

    @cartesian
    def is_dirichlet_boundary(self, p):
        return np.abs(p[..., 2]) < 1e-12


class Truss_2d():
    def __init__(self):
        self.A = 6451.6 # 横截面积 mm^2
        self.E = 0.7031 # 弹性模量 ton/mm^2

    def init_mesh(self, n=1):
        l = 9143 # 单位 mm
        node = np.array([
            [0, l], [l, l], [2*l, l],
            [0, 0], [l, 0], [2*l, 0]], dtype=np.float64)
        edge = np.array([
            [0, 1], [0, 4], [1, 2], [1, 3], [1, 4],
            [1, 5], [2, 3], [2, 4], [3, 4], [4, 5]], dtype=np.int_)
        mesh = TrussMesh(node, edge)
        return mesh

    @cartesian
    def displacement(self, p):
        pass

    @cartesian
    def jacobian(self, p):
        pass

    @cartesian
    def strain(self, p):
        pass

    @cartesian
    def stress(self, p):
        pass

    @cartesian
    def source(self, p):
        shape = len(p.shape[:-1])*(1,) + (-1, )
        val = np.zeros(shape, dtype=np.float_)
        return val 

    @cartesian
    def force(self, p):
        '''
        施加 (0, 900, 0) 的力，即平行于 y 轴方向大小为 900N 的力
        '''
        val = np.array([0, 900, 0])
        return val

    def is_force_boundary(self, p):
        '''
        对第 3, 4 号节点施加力
        '''
        return np.abs(p[..., 1]) < 1e-12 and np.ads(p[..., 0]) > 1e-12

    @cartesian
    def dirichlet(self, p):
        shape = len(p.shape)*(1, )
        val = np.array([0.0])
        return val.reshape(shape)

    @cartesian
    def is_dirichlet_boundary(self, p):
        return np.abs(p[..., 0]) < 1e-12


