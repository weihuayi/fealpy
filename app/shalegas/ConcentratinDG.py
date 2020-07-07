#!/usr/bin/env python3
# 
"""

Notes
-----

给定一个求解区域， 区域中有一个固定的流场 v 和一个初始浓度 c， 计算浓度随时间的
变化。
"""
import sys
import numpy as np
from numpy.linalg import inv
from scipy.sparse import bmat, spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


from fealpy.decorator import cartesian
from fealpy.mesh import MeshFactory
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.functionspace import ScaledMonomialSpace2d 
from fealpy.timeintegratoralg import UniformTimeLine 

"""
该模型例子，在给定流场的情况下， 用间断有限元模拟物质浓度的变化过程。
"""

class VelocityData:
    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        val = np.array([0.0], np.float64)
        shape = len(p.shape[:-1])*(1, )
        return val.reshape(shape) 

    @cartesian
    def neumann(self, p, n):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape[:-1], dtype=np.float64)
        flag0 = np.abs(x) < 1e-13
        val[flag0] = 1
        flag1 = np.abs(x-1) < 1e-13
        val[flag1] = -1
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = (np.abs(x) < 1e-13) | (np.abs(x-1) < 1e-13)
        return flag

class ConcentrationData:
    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        val = np.array([0.0], np.float64)
        shape = len(p.shape[:-1])*(1, )
        return val.reshape(shape) 

    @cartesian
    def neumann(self, p, n):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape[:-1], dtype=np.float64)
        flag0 = np.abs(x) < 1e-13
        val[flag0] = 1
        flag1 = np.abs(x-1) < 1e-13
        val[flag1] = -1
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = (np.abs(x) < 1e-13) | (np.abs(x-1) < 1e-13)
        return flag

class ConcentrationDG():
    def __init__(self, vdata, cdata, mesh, p=0):
        self.vdata = vdata
        self.cdata = cdata
        self.mesh = mesh
        self.uspace = RaviartThomasFiniteElementSpace2d(mesh, p=p)
        self.cspace = ScaledMonomialSpace2d(mesh, p=p+1)

        self.uh = self.uspace.function() # 速度场自由度数组
        self.ph = self.uspace.smspace.function() # 压力场自由度数组
        self.ch = self.cspace.function() # 浓度场自由度数组

        self.M = self.space.smspace.cell_mass_matrix() 
        self.H = inv(self.M)
        self.set_init_velocity_field() # 计算初始的速度场和压力场

    def set_init_velocity_field(self):
        vdata = self.vdata
        mesh = self.mesh
        space = self.space
        uh = self.uh
        ph = self.ph

        udof = space.number_of_global_dofs()
        pdof = space.smspace.number_of_global_dofs()
        gdof = udof + pdof + 1

        A = space.stiff_matrix()
        B = space.div_matrix()
        C = self.M[:, 0, :].reshape(-1)
        F1 = space.source_vector(vdata.source)

        AA = bmat([[A, -B, None], [-B.T, None, C[:, None]], [None, C, None]], format='csr')

        isBdDof = space.set_dirichlet_bc(uh, vdata.neumann)

        x = np.r_['0', uh, ph, 0] 
        isBdDof = np.r_['0', isBdDof, np.zeros(pdof+1, dtype=np.bool_)]
        FF = np.r_['0', np.zeros(udof, dtype=np.float64), F1, 0]

        FF -= AA@x
        bdIdx = np.zeros(gdof, dtype=np.int)
        bdIdx[isBdDof] = 1
        Tbd = spdiags(bdIdx, 0, gdof, gdof)
        T = spdiags(1-bdIdx, 0, gdof, gdof)
        AA = T@AA@T + Tbd
        FF[isBdDof] = x[isBdDof]
        x[:] = spsolve(AA, FF)
        uh[:] = x[:udof]
        ph[:] = x[udof:-1]

    def get_current_right_vector(self, data, timeline):
        cdata = self.cdata
        ch = data[0]
        uh = data[1]
        dt = timeline.current_time_step_length()
        nt = timeline.next_time_level()
        # 这里没有考虑源项，F 只考虑了单元内的流入和流出
        F = self.space.convection_vector(nt, cdata.neumann, ch, uh,
                threshold=cdata.is_neumann_boundary) 

        F = self.H@(F[:, :, None]/0.2)
        F *= dt
        return F.flat

    def solve(self, data, timeline):
        F = self.get_current_right_vector(data, timeline)
        ch = data[0]
        ch += F 

    def output(self, data, nameflag, queue, stop=False):
        ch = data[0]
        if stop:
            queue.put(-1)
        else:
            queue.put({'c'+nameflag: ('celldata', ch)})



if __name__ == '__main__':

    from fealpy.writer import MeshWriter
    mf = MeshFactory()
    mesh = mf.regular([0, 1, 0, 1], n=10)
    vdata = VelocityData()
    cdata = ConcentrationData()
    model = ConcentrationDG(vdata, cdata, mesh)
    timeline = UniformTimeLine(0, 1, 100)
    data = (model.ch, model.uh)

    writer = MeshWriter(mesh, 
            simulation=timeline.time_integration, 
            args=(data, model)
            )
    writer.run()
