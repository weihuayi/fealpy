
"""

Notes
-----

    1. 混合物的摩尔浓度 c 的计算公式为
        c = p/(ZRT), 
        Z^3 - (1 - B)Z^2 + (A - 3B^2 -2B)Z - (AB - B^2 -B^3) = 0
        A = a p/(R^2T^2)
        B = b p/(RT)
        其中:

        a = 3
        b = 1/3

        p 是压力 
        Z 是压缩系数 
        R 是气体常数
        T 是温度
        M_i 是组分 i 的摩尔质量
        
        混合物的密度计算公式为:

        rho = M_0 c_0 + M_1 c_1 + M_2 c_2 ...

    2. c_i : 组分 i 的摩尔浓度
       z_i : 组分 i 的摩尔分数

       c_i = z_i c

    3. 
    甲烷: methane,  CH_4,   16.04 g/mol,  0.42262 g/cm^3, 190.8 K, 4.640 MPa
    乙烷: Ethane, C_2H_6,   30.07 g/mol, 1.212 kg/m^3, 305.33 K, 4.872 MPa
    丙烷: Propane,  C_3H_8,  44.096 g/mol,  1.83 kg/m^3, 369.8 K, 4.26 MPa
    (25度 100 kPa

    4. 气体常数  R = 8.31446261815324 	J/K/mol

    5. 6.02214076 x 10^{23}

    6. 压力单位是 bar, 温度单位是 K, 时间单位是 d 

    7. acentricity factor 偏心因子


References
----------
[1] https://www.sciencedirect.com/science/article/abs/pii/S0378381217301851

Authors
    Huayi Wei, weihuayi@xtu.edu.cn
"""
import matplotlib.pyplot as plt
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
        self.m = [0.01604, 0.03007, 0.044096] # kg/mol 一摩尔质量, 这里是常数
        self.R = 8.31446261815324 # J/K/mol
        self.T = 397 # K 绝对温度
        self.p = 50 # 单位是 bar,  1 bar = 1e+5 Pa

        self.c = self.molar_dentsity(self.p) # 压强 p 下， 气体的摩尔浓度
        print("压强 {} (bar)下的气体摩尔浓度:".format(self.p), self.c)

        # 根据理想气体状态方程，计算单位边界输入的摩尔浓度
        self.V = 50**2*0.2*0.1/365 # 每天注入的体积数 
        self.n = self.p*1e+5**self.V/self.R/self.T/2 # 单位边界长度每天注入的摩尔数

        print("每天注入单位边界的体积数为:", self.V)
        print("每天注入单位边界的物质摩尔数为:", self.n)

    def init_pressure(self, pspace):
        """

        Notes
        ----
        目前压力用分片常数逼近。
        """
        ph = pspace.function()
        ph[:] = 50 # 单位是 bar， 1 bar = 1e+5  
        return ph

    def init_molar_density(self, cspace):
        c = self.molar_dentsity(50)
        ch = cspace.function(dim=3)
        ch[0::3, 2] = c
        #ch[:, 2] = cspace.local_projection(c)
        return ch

    def space_mesh(self, n=50):
        """

        Notes
        -----

        最小网格单元尺寸为 1 m
        """
        box = [0, 50, 0, 50]
        mf = MeshFactory()
        mesh = mf.boxmesh2d(box, nx=n, ny=n, meshtype='tri')
        return mesh

    def time_mesh(self, n=1):
        timeline = UniformTimeLine(0, 1, n)
        return timeline

    def molar_dentsity(self, ph):
        """

        Notes
        ----
        给一个分片常数的压力，计算混合物的浓度 c
        """

        ph *= 1e+5 # 转换为标准单位 Pa

        t = self.R*self.T  # 气体常数和绝对温度的乘积
        if type(ph) in {int, float}:
            A = 3*ph/t**2
            B = ph/t/3 
            a = np.ones(4, dtype=np.float64)
            a[1] = B - 1
            a[2] = A - 3*B**2 - 2*B
            a[3] = -A*B + B**2 + B**3
            Z = np.max(np.roots(a))
            print("Compressibility factor is :", Z, np.roots(a))
            c = ph/Z/t 
        else:
            A = 3*ph/t**2
            B = ph/t/3 

            a = np.ones((len(ph), 4), dtype=ph.dtype)
            a[:, 1] = B - 1
            a[:, 2] = A - 3*B**2 - 2*B
            a[:, 3] = -A*B + B**2 + B**3
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
            return self.c*0.8 
        elif n == 1:
            return self.c*0.2 
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
        """

        Notes
        -----
        在区域左下角给一个速度边界条件 v\cdot n
        """
        x = p[..., 0]
        y = p[..., 1]
        flag = (x < 1) & (y < 1)
        val = np.zeros(p.shape[0:-1], dtype=p.dtype)
        val[flag] = -self.n/self.c
        return val 

    @cartesian
    def is_velocity_bc(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = (x > 49) & (y > 49)
        return ~flag

    @cartesian
    def pressure_bc(self, p):
        """
        Notes
        -----
        在区域右上角给出一个压力控制条件，要低于区域内部的压力。
        """
        return 49.9999 

    @cartesian
    def is_pressure_bc(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = (x > 49) & (y > 49)
        return flag

class ShaleGasSolver():
    def __init__(self, model):
        self.model = model
        self.mesh = model.space_mesh()
        self.timeline =  model.time_mesh(n=3650) 
        self.uspace = RaviartThomasFiniteElementSpace2d(self.mesh, p=0)
        self.cspace = ScaledMonomialSpace2d(self.mesh, p=1) # 线性间断有限元空间

        self.uh = self.uspace.function() # 速度
        self.ph = model.init_pressure(self.uspace.smspace) # 初始压力

        # 三个组分的摩尔密度, 三个组分一起计算 
        self.ch = model.init_molar_density(self.cspace) 

        # TODO：初始化三种物质的浓度
        # 1 muPa*s = 1e-11 bar*s = 1e-11*24*3600 bar*d = 8.64e-07 bar*d
        # 1 md = 9.869 233e-16 m^2
        self.options = {
                'viscosity': 1.0,    # 粘性系数 1 muPa*s = 1e-6 Pa*s, 1 cP = 10^{−3} Pa⋅s = 1 mPa⋅s
                'permeability': 1.0, # 1 md 渗透率, 1 md = 9.869233e-16 m^2
                'temperature': 397, # 初始温度 K
                'pressure': 50,   # bar 初始压力
                'porosity': 0.2,  # 孔隙度
                'injecttion_rate': 0.1,  # 注入速率
                'compressibility': 0.001, #压缩率
                'pmv': (1.0, 1.0, 1.0), # 偏摩尔体积
                'rdir': '/home/why/result/test/',
                'step': 1} 
        self.CM = self.cspace.cell_mass_matrix() 
        self.H = inv(self.CM)

        c = 8.64/9.869233*1e+9 
        self.M = c*self.uspace.mass_matrix()
        self.B = -self.uspace.div_matrix()

        dt = self.timeline.dt
        c = self.options['porosity']*self.options['compressibility']/dt
        self.D = c*self.uspace.smspace.mass_matrix() 

        # 压力边界条件
        self.F0 = -self.uspace.set_neumann_bc(
                model.pressure_bc, threshold=model.is_pressure_bc)

        # vtk 文件输出
        node, cell, cellType, NC = self.mesh.to_vtk()
        self.points = vtk.vtkPoints()
        self.points.SetData(vnp.numpy_to_vtk(node))
        self.cells = vtk.vtkCellArray()
        self.cells.SetCells(NC, vnp.numpy_to_vtkIdTypeArray(cell))
        self.cellType = cellType

    def one_step_solve(self):
        """

        Notes
        -----
            求解一个时间层的数值解
        """
        udof = self.uspace.number_of_global_dofs()
        pdof = self.uspace.smspace.number_of_global_dofs()
        gdof = udof + pdof

        cdof = self.cspace.number_of_global_dofs()

        timeline = self.timeline
        dt = timeline.current_time_step_length()
        nt = timeline.next_time_level()

        # 1. 求解下一时间层的速度和压力
        M = self.M
        B = self.B
        D = self.D
        print('ch:', self.ch)
        E = self.uspace.pressure_matrix(self.ch)

        F1 = D@self.ph

        AA = bmat([[M, B], [E, D]], format='csr')
        FF = np.r_['0', self.F0, F1]

        isBdDof = self.uspace.set_dirichlet_bc(self.uh, 
                self.model.velocity_bc, threshold=self.model.is_velocity_bc)
        x = np.r_['0', self.uh, self.ph] 
        isBdDof = np.r_['0', isBdDof, np.zeros(pdof, dtype=np.bool_)]
        
        FF -= AA@x
        bdIdx = np.zeros(gdof, dtype=np.int)
        bdIdx[isBdDof] = 1
        Tbd = spdiags(bdIdx, 0, gdof, gdof)
        T = spdiags(1-bdIdx, 0, gdof, gdof)
        AA = T@AA@T + Tbd
        FF[isBdDof] = x[isBdDof]

        x = spsolve(AA, FF).reshape(-1)
        self.uh[:] = x[:udof]
        self.ph[:] = x[udof:]
        print('uh:', self.uh)
        print('ph:', self.ph)

        # 2. 求解下一层的浓度
        nc = self.ch.shape[1] # 这里有 nc 个组分
        phi = self.options['porosity'] # 孔隙度
        for i in range(nc-1):
            g = lambda x : self.model.concentration_bc(x, n=i)
            F = self.uspace.convection_vector(nt, self.ch.index(i), self.uh,
                    g=g) 
            F = self.H@(F[:, :, None]/phi)
            F *= dt
            print('ch:', self.ch.index(i))
            print('F:', F)
            self.ch[:, i] += F.flat

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
        ph = self.ph

        V = uh.value(bc)
        V = np.concatenate((V, np.zeros((V.shape[0], 1), dtype=V.dtype)), axis=1)
        val = vnp.numpy_to_vtk(V)
        val.SetName('velocity')
        cdata.AddArray(val)

        val = vnp.numpy_to_vtk(ph[:])
        val.SetName('pressure')
        cdata.AddArray(val)

        ch = self.ch
        val = ch.value(ps)
        if len(ch.shape) == 2:
            for i in range(ch.shape[1]):
                val0 = vnp.numpy_to_vtk(val[:, i])
                val0.SetName('concentration' + str(i))
                cdata.AddArray(val0)
        else:
            val = vnp.numpy_to_vtk(val)
            val.SetName('concentration')
            cdata.AddArray(val)
            

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(vmesh)
        writer.Write()



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model = Model_1()

    solver = ShaleGasSolver(model)


    if False:
        mesh = solver.mesh 
        isBdEdge = mesh.ds.boundary_edge_flag()
        bc = mesh.entity_barycenter('edge', index=isBdEdge)
        flag= model.velocity_bc(bc, None) < 0
        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        isPBC = model.is_pressure_bc(bc)
        isVBC = model.is_velocity_bc(bc)
        mesh.find_node(axes, node=bc, index=flag)
        plt.show()

    print(solver.ch)
    print(solver.ph)

    solver.one_step_solve()

