#!/usr/bin/env python3
# 
"""

Notes
-----

给定一个二维求解区域， 区域中有一个固定的流场 v 和一个初始浓度 c， 计算浓度随时间的
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


# 模型 0

class VelocityData_2:
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
        flag0 = (np.abs(x) < 1e-13) & (y < 1/16)
        flag1 = (np.abs(y) < 1e-13) & (x < 1/16)
        val[flag0 | flag1] = -0.1 #负的表示流入区域

        flag0 = (np.abs(x-1) < 1e-13) & (y > 15/16)
        flag1 = (np.abs(y-1) < 1e-13) & (x > 1 - 1/16)
        val[flag1 | flag0] = 0.1 # 负的表示流出区域
        return val


class ConcentrationData_2:
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
    def init_value(self, p):
        val = np.array([0.0], np.float64)
        shape = len(p.shape[:-1])*(1, )
        return val.reshape(shape) 

    @cartesian
    def dirichlet(self, p): # 这里在边界上，始终知道边界外面的浓度
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape[:-1], dtype=np.float64)
        flag0 = (np.abs(x) < 1e-13) & (y < 1/16)
        flag1 = (np.abs(y) < 1e-13) & (x < 1/16)
        val[flag0 | flag1] = 1
        return val


class ConcentrationDG():
    def __init__(self, vdata, cdata, mesh, timeline, p=0,
            options={'rdir':'/home/why/result', 'step':1000, 'porosity':0.2}):

        self.options = options
        self.vdata = vdata
        self.cdata = cdata
        self.mesh = mesh
        self.timeline = timeline
        self.uspace = RaviartThomasFiniteElementSpace2d(mesh, p=p)
        self.cspace = ScaledMonomialSpace2d(mesh, p=p+1)

        self.uh = self.uspace.function() # 速度场自由度数组
        self.ph = self.uspace.smspace.function() # 压力场自由度数组
        self.ch = self.cspace.function() # 浓度场自由度数组

        # 初始浓度场设为 0 
        ldof = self.cspace.number_of_local_dofs()

        self.M = self.cspace.cell_mass_matrix() 
        self.H = inv(self.M)
        self.set_init_velocity_field() # 计算初始的速度场和压力场

        # vtk 文件输出
        node, cell, cellType, NC = self.mesh.to_vtk()
        self.points = vtk.vtkPoints()
        self.points.SetData(vnp.numpy_to_vtk(node))
        self.cells = vtk.vtkCellArray()
        self.cells.SetCells(NC, vnp.numpy_to_vtkIdTypeArray(cell))
        self.cellType = cellType

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
        F1 = -uspace.source_vector(vdata.source)

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

    def get_current_right_vector(self):
        """

        Notes
        -----
        计算下一时刻对应的右端项。
        """

        phi = self.options['porosity']
        timeline = self.timeline
        ch = self.ch 
        uh = self.uh 
        dt = timeline.current_time_step_length()
        nt = timeline.next_time_level()
        # 这里没有考虑源项，F 只考虑了单元内的流入和流出
        F = self.uspace.convection_vector(nt, ch, uh, g=self.cdata.dirichlet) 

        F = self.H@(F[:, :, None]/phi)
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

        rdir = self.options['rdir']
        step = self.options['step']
        timeline = self.timeline
        dt = timeline.current_time_step_length()
        timeline.reset() # 时间置零

        fname = rdir + '/test_'+ str(timeline.current).zfill(10) + '.vtu'
        self.write_to_vtk(fname)
        print(fname)
        while not timeline.stop():
            self.one_step_solve()
            timeline.current += 1
            if timeline.current%step == 0:
                fname = rdir + '/test_'+ str(timeline.current).zfill(10) + '.vtu'
                print(fname)
                self.write_to_vtk(fname)
        timeline.reset()

    def write_to_vtk(self, fname):
        # 重心处的值
        bc = np.array([1/3, 1/3, 1/3], dtype=np.float64)
        ps = self.mesh.bc_to_point(bc)
        vmesh = vtk.vtkUnstructuredGrid()
        vmesh.SetPoints(self.points)
        vmesh.SetCells(self.cellType, self.cells)
        cdata = vmesh.GetCellData()
        pdata = vmesh.GetPointData()

        uh = self.uh 
        V = uh.value(bc)
        V = np.r_['1', V, np.zeros((len(V), 1), dtype=np.float64)]
        val = vnp.numpy_to_vtk(V)
        val.SetName('velocity')
        cdata.AddArray(val)

        if True:
            ch = self.ch
            rch = ch.to_cspace_function()
            val = vnp.numpy_to_vtk(rch)
            val.SetName('concentration')
            pdata.AddArray(val)
        else:
            ch = self.ch
            val = ch.value(ps)
            val = vnp.numpy_to_vtk(val)
            val.SetName('concentration')
            cdata.AddArray(val)

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(vmesh)
        writer.Write()



if __name__ == '__main__':

    """
    python3 ConcentrationDG.py 0 5 50000 2000 /home/why/result/c/corner
    """

    mf = MeshFactory()
    
    m = int(sys.argv[1])
    if m == 0:
        mesh = mf.boxmesh2d([0, 1, 0, 1], nx=64, ny=64, meshtype='tri')
        vdata = VelocityData_2()
        cdata = ConcentrationData_2()

    T = float(sys.argv[2])
    NT = int(sys.argv[3])
    step = int(sys.argv[4])
    timeline = UniformTimeLine(0, T, NT)
    options = {'rdir': sys.argv[5], 'step':step, 'porosity':0.2}
    model = ConcentrationDG(vdata, cdata, mesh, timeline, options=options)
    model.solve()

