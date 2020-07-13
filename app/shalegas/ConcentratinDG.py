#!/usr/bin/env python3
# 
"""

Notes
-----

给定一个求解区域， 区域中有一个固定的流场 v 和一个初始浓度 c， 计算浓度随时间的
变化。
"""
import sys

import vtk
import vtk.util.numpy_support as vnp

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
        val[flag0] = 0.01 
        flag1 = np.abs(x-1) < 1e-13
        val[flag1] = -0.01
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = (np.abs(x) < 1e-13) | (np.abs(x-1) < 1e-13)
        return flag

class ConcentrationData0:
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
        val[flag0] = 0.01
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = (np.abs(x) < 1e-13)
        return flag

class ConcentrationData1:
    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        val = np.array([0.0], np.float64)
        shape = len(p.shape[:-1])*(1, )
        return val.reshape(shape) 

    def init_value(self, p):
        val = np.array([1.0], np.float64)
        shape = len(p.shape[:-1])*(1, )
        return val.reshape(shape) 

class ConcentrationDG():
    def __init__(self, vdata, cdata, mesh, timeline, p=0):
        self.vdata = vdata
        self.cdata = cdata
        self.mesh = mesh
        self.timeline = timeline
        self.uspace = RaviartThomasFiniteElementSpace2d(mesh, p=p)
        self.cspace = ScaledMonomialSpace2d(mesh, p=p+1)

        self.uh = self.uspace.function() # 速度场自由度数组
        self.ph = self.uspace.smspace.function() # 压力场自由度数组
        self.ch = self.cspace.function() # 浓度场自由度数组

        # 初始浓度场设为 1
        ldof = self.cspace.number_of_local_dofs()
        self.ch[0::ldof] = 1.0

        self.M = self.cspace.cell_mass_matrix() 
        self.H = inv(self.M)
        self.set_init_velocity_field() # 计算初始的速度场和压力场

        # vtk 文件输出
        self.vtkmesh =vtk.vtkUnstructuredGrid() 
        self.writer = vtk.vtkXMLUnstructuredGridWriter()

    def set_init_velocity_field(self):
        """

        Notes
        ----
        利用最低阶混合元计算初始速度场
        """
        vdata = self.vdata
        mesh = self.mesh
        uspace = self.uspace
        uh = self.uh
        ph = self.ph

        udof = uspace.number_of_global_dofs()
        pdof = uspace.smspace.number_of_global_dofs()
        gdof = udof + pdof + 1

        M = uspace.smspace.cell_mass_matrix()
        A = uspace.stiff_matrix()
        B = uspace.div_matrix()
        C = M[:, 0, :].reshape(-1)
        F1 = uspace.source_vector(vdata.source)

        AA = bmat([[A, -B, None], [-B.T, None, C[:, None]], [None, C, None]], format='csr')

        isBdDof = uspace.set_dirichlet_bc(uh, vdata.neumann)

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

    def set_init_vtk_mesh_and_data(self, fname='test.vtu'):
        """

        Notes
        -----
        设定初始的vtk 数据
        """

        # 设定 vtk 网格数据
        node, cell, cellType, NC = self.mesh.to_vtk()
        points = vtk.vtkPoints()
        points.SetData(vnp.numpy_to_vtk(node))

        cells = vtk.vtkCellArray()
        cells.SetCells(NC, vnp.numpy_to_vtkIdTypeArray(cell))
        self.vtkmesh.SetPoints(points)
        self.vtkmesh.SetCells(cellType, cells)

        # 设定速度场数据
        if False:
            uh = self.uh 
            bc = np.array([1/3, 1/3, 1/3], dtype=np.float64)
            V = uh.value(bc)
            V = np.r_['1', V, np.zeros((len(V), 1), dtype=np.float64)]
            cdata = self.vtkmesh.GetCellData()
            val = vnp.numpy_to_vtk(V)
            val.SetName('velocity')
            cdata.AddArray(val)

        # 联接 vtkmesh 到 writer 中
        self.writer.SetFileName(fname)
        self.writer.SetInputData(self.vtkmesh)

    def get_current_right_vector(self):
        """

        Notes
        -----
        计算下一时刻对应的右端项。
        """

        timeline = self.timeline
        ch = self.ch 
        uh = self.uh 
        dt = timeline.current_time_step_length()
        nt = timeline.next_time_level()
        # 这里没有考虑源项，F 只考虑了单元内的流入和流出
        F = self.uspace.convection_vector(nt, ch, uh) 

        F = self.H@(F[:, :, None]/0.2)
        F *= dt
        return F.flat

    def one_step_solve(self):
        """

        Notes
        -----
        计算下一时刻的浓度。
        """
        F = self.get_current_right_vector()
        self.ch += F 


    def solve(self):
        """

        Notes
        -----

        计算所有的时间层。
        """

        # 重心处的值
        bc = np.array([1/3, 1/3, 1/3], dtype=np.float64)
        ps = self.mesh.bc_to_point(bc)

        timeline = self.timeline
        dt = timeline.current_time_step_length()
        timeline.reset() # 时间置零

        cdata = self.vtkmesh.GetCellData()

        NL = timeline.number_of_time_levels()
        writer = self.writer
        writer.SetNumberOfTimeSteps(NL)

        writer.Start()
        val = self.ch.value(ps)
        name = 'c' + str(timeline.current).zfill(6)
        val = vnp.numpy_to_vtk(val)
        val.SetName(name)
        cdata.AddArray(val)
        print(name)
        writer.WriteNextTime(timeline.current)    

        while not timeline.stop():
            self.one_step_solve()
            timeline.current += 1

            val = self.ch.value(ps)
            name = 'c' + str(timeline.current).zfill(6)
            val = vnp.numpy_to_vtk(val)
            val.SetName(name)
            cdata.AddArray(val)
            print(name)
            writer.WriteNextTime(timeline.current)    

        writer.Stop()
        timeline.reset()

if __name__ == '__main__':

    from fealpy.writer import MeshWriter
    mf = MeshFactory()
    mesh = mf.regular([0, 1, 0, 1], n=6)
    vdata = VelocityData()
    cdata = ConcentrationData1()
    timeline = UniformTimeLine(0, 1, 1000)
    model = ConcentrationDG(vdata, cdata, mesh, timeline)
    model.set_init_vtk_mesh_and_data(fname='test.vtu')
    model.solve()

