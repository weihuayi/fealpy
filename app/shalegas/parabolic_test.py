
import numpy as np
from numpy.linalg import inv
from scipy.sparse import csr_matrix, bmat, spdiags
from scipy.sparse.linalg import spsolve

from fealpy.decorator import cartesian
from fealpy.mesh import MeshFactory
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.timeintegratoralg.timeline import UniformTimeLine

import vtk
import vtk.util.numpy_support as vnp

class Model_1():
    """

    Notes
    -----
    时间单位是天

    每天在区域右下角注入：

    50x50x0.2x0.1/365 = 0.136986301369863 体积的甲烷和乙烷的混合物，各自的摩尔分
    数分别是 0.8 和 0.2

    在注入边界上每天单位长度上注入的体积为 0.0684931506849315

    """
    def __init__(self):
        self.m = [0.01604, 0.03007, 0.044096] # kg/mol 一摩尔质量, TODO：确认是 g/mol
        self.R = 8.31446261815324 # J/K/mol
        self.T = 397 # K 绝对温度

    def init_pressure(self, pspace):
        """

        Notes
        ----
        目前压力用分片常数逼近。
        """
        ph = pspace.function()
        ph[:] = 50 # 初始压力
        return ph

    def init_molar_density(self, cspace):
        c = self.molar_dentsity(50)
        ch = cspace.function(dim=3)
        ch[:, 2] = cspace.local_projection(c)
        return ch

    def space_mesh(self, n=50):
        box = [0, 50, 0, 50]
        mf = MeshFactory()
        mesh = mf.boxmesh2d(box, nx=n, ny=n, meshtype='tri')
        return mesh

    def time_mesh(self, n=1000):
        timeline = UniformTimeLine(0, 1, n)
        return timeline

    def molar_dentsity(self, ph):
        """

        Notes
        ----
        给一个分片常数的压力，计算混合物的浓度 c
        """

        t = self.R*self.T 
        if type(ph) in {int, float}:
            A = 3*ph/t**2
            B = ph/t/3 
            a = np.ones(4, dtype=np.float64)
            a[1] = B - 1
            a[2] = A - 3*B**2 - 2*B
            a[3] = -A*B + B**2 - B**3
            Z = np.max(np.roots(a))
            c = ph/Z/t 
        else:
            A = 3*ph/t**2
            B = ph/t/3 

            a = np.ones((len(ph), 4), dtype=ph.dtype)
            a[:, 1] = B - 1
            a[:, 2] = A - 3*B**2 - 2*B
            a[:, 3] = -A*B + B**2 - B**3
            Z = np.max(np.array(list(map(np.roots, a))), axis=-1)
            c = ph/Z/t 
        return c

    @cartesian
    def concentration_bc(self, p, n=0):
        """

        Notes
        -----
        在区域左下角给一个浓度的边界条件 c_i 

        n 表示该函数计算第 n 个组分的边界条件
        """
        if n == 0:
            return 0.1
        elif n == 1:
            return 0.1
        elif n == 3:
            return 0.0

    @cartesian
    def is_concentration_bc(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = (x < 1) & (y < 1)
        return flag

    @cartesian
    def velocity_bc(self, p, n):
        """

        Notes
        -----
        在区域左下角给一个速度边界条件 v\cdot n
        """
        x = p[..., 0]
        y = p[..., 1]
        flag = (x < 1) & (y < 1)
        val = np.zeros(p.shape[0:-1], dtype=p.dtype)
        val[flag] = -0.01
        return val 

    @cartesian
    def is_velocity_bc(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = (x < 49) | (y < 49)
        return flag

    @cartesian
    def pressure_bc(self, p):
        """
        Notes
        -----
        在区域右上角给出一个压力控制条件，要低于区域内部的压力。
        """
        return 25 

    @cartesian
    def is_pressure_bc(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = (x > 49) & (y > 49)
        return flag

class ParabolicMFEMSolver():

    def __init__(self, model):
        self.model = model
        self.mesh = model.space_mesh()
        self.timeline = model.time_mesh()
        dt = self.timeline.current_time_step_length()
        self.space = RaviartThomasFiniteElementSpace2d(self.mesh, p=0)
        self.uh = self.space.function()
        self.ph = model.init_pressure(self.space.smspace)

        self.M = self.space.mass_matrix()
        self.B = -self.space.div_matrix()
        self.D = self.space.smspace.mass_matrix()/dt
        self.F0 = -self.space.set_neumann_bc(model.pressure_bc, threshold=model.is_pressure_bc) 

        # vtk 文件输出
        node, cell, cellType, NC = self.mesh.to_vtk()
        self.points = vtk.vtkPoints()
        self.points.SetData(vnp.numpy_to_vtk(node))
        self.cells = vtk.vtkCellArray()
        self.cells.SetCells(NC, vnp.numpy_to_vtkIdTypeArray(cell))
        self.cellType = cellType

    def one_step_solve(self):

        AA = bmat([[self.M, self.B], [self.B.T, self.D]], format='csr')
        F1 = self.D@self.ph
        FF = np.r_['0', self.F0, F1]

        udof = self.space.number_of_global_dofs()
        pdof = self.space.smspace.number_of_global_dofs()
        gdof = udof + pdof

        uh = self.space.function()
        ph = self.space.smspace.function()
        isBdDof = self.space.set_dirichlet_bc(uh, 
                model.velocity_bc, threshold=model.is_velocity_bc)
        x = np.r_['0', uh, ph] 
        isBdDof = np.r_['0', isBdDof, np.zeros(pdof, dtype=np.bool_)]

        FF -= AA@x
        bdIdx = np.zeros(gdof, dtype=np.int_)
        bdIdx[isBdDof] = 1
        Tbd = spdiags(bdIdx, 0, gdof, gdof)
        T = spdiags(1-bdIdx, 0, gdof, gdof)
        AA = T@AA@T + Tbd
        FF[isBdDof] = x[isBdDof]

        x = spsolve(AA, FF).reshape(-1)
        self.uh[:] = x[:udof]
        self.ph[:] = x[udof:]

    def solve(self, rdir='.'):
        """

        Notes
        -----

        计算所有的时间层。
        """

        timeline = self.timeline
        dt = timeline.current_time_step_length()
        timeline.reset() # 时间置零

        fname = rdir + '/test_'+ str(timeline.current).zfill(10) + '.vtu'
        self.write_to_vtk(fname)
        print(fname)
        while not timeline.stop():
            self.one_step_solve()
            timeline.current += 1
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
        ph = self.ph

        V = uh.value(bc)
        V = np.concatenate((V, np.zeros((V.shape[0], 1), dtype=V.dtype)), axis=1)
        val = vnp.numpy_to_vtk(V)
        val.SetName('velocity')
        cdata.AddArray(val)

        val = vnp.numpy_to_vtk(ph[:])
        val.SetName('pressure')
        cdata.AddArray(val)

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(vmesh)
        writer.Write()


if __name__ == '__main__':

    model = Model_1()
    solver = ParabolicMFEMSolver(model)
    solver.solve()
