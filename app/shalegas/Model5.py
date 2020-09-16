
"""

Notes
-----
这是一个两相流和地质力学耦合的模型, 需要求的量有

* v: 总速度
* p: 压强
* S_w: 水的饱和度 S_w
* u: 岩石位移  

速度 RT0 元
压强 P0 元
饱和度 P0 元
岩石位移 P1 连续元离散

目前, 模型
* 忽略了毛细管压强和重力作用
* 没有考虑裂缝
* 饱和度用分片线性间断元求解, 非线性的迎风格式



渐近解决方案:
1. Picard 迭代
2. 气的可压性随着压强的变化而变化
3. 考虑渗透率随着孔隙度的变化而变化 
4. 考虑裂缝，裂缝用自适应网格加密，设设置更大的渗透率实现

体积模量:  K = E/3/(1 - 2*nu) = lambda + 2/3* mu
"""

import numpy as np

import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, coo_matrix, bmat, diags
from scipy.sparse.linalg import spsolve

from fealpy.decorator import barycentric

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.functionspace import RaviartThomasFiniteElementSpace3d

import pyamg 

import vtk
import vtk.util.numpy_support as vnp


class WaterFloodingModel2d():

    def __init__(self):
        self.domain=[0, 10, 0, 10] # m
        self.rock = {
            'permeability': 2, # 1 d = 9.869 233e-13 m^2 
            'porosity': 0.3, # None
            'lame':(1.0e+2, 3.0e+2), # lambda and mu 拉梅常数, MPa
            'biot': 1.0,
            'initial pressure': 3, # MPa
            'initial stress': 60.66+86.5, # MPa 初始应力 sigma_0 , sigma_eff
            'solid grain stiffness': 2.0e+2 # MPa 固体体积模量
            }
        self.water = {
            'viscosity': 1, # 1 cp = 1 mPa*s
            'compressibility': 1.0e-3, # MPa^{-1}
            'initial saturation': 0.0, 
            'injection rate': 3.51e-6 # s^{-1}, 每秒注入多少水
            }
        self.oil = {'viscosity': 2, # cp
            'compressibility': 2.0e-3, # MPa^{-1}
            'initial saturation': 1.0, 
            'production rate': 3.50e-6 # s^{-1}, 每秒产出多少油
            }
        self.bc = {'displacement': 0.0, 'flux': 0.0}

        self.GD = 2

    def space_mesh(self, n=32):
        from fealpy.mesh import MeshFactory
        mf = MeshFactory()
        mesh = mf.boxmesh2d(self.domain, nx=n, ny=n, meshtype='tri')
        return mesh

    def time_mesh(self, T=1, n=100):
        from fealpy.timeintegratoralg.timeline import UniformTimeLine
        timeline = UniformTimeLine(0, T, n)
        return timeline

    def water_relative_permeability(self, Sw):
        """

        Notes
        ----
        给定水的饱和度, 计算水的相对渗透率
        """
        val = Sw**2
        return val

    def oil_relative_permeability(self, Sw):
        """

        Notes
        ----
        给定水的饱和度, 计算油的相对渗透率
        """
        val = (1 - Sw)**2 
        return val

    def gas_relative_permeability(self, Sw):
        pass


class WaterFloodingModel3d():

    def __init__(self):
        self.domain=[0, 10, 0, 10, 0, 10] # m
        self.rock = {
            'permeability': 2, # 1 d = 9.869 233e-13 m^2 
            'porosity': 0.3, # None
            'lame':(1.0e+2, 3.0e+2), # lambda and mu 拉梅常数, MPa
            'biot': 1.0,
            'initial pressure': 3, # MPa
            'initial stress': 60.66+86.5, # MPa 初始应力 sigma_0 , sigma_eff
            'solid grain stiffness': 2.0e+2 # MPa 固体体积模量 lambda + 2/3*mu
            }
        self.water = {
            'viscosity': 1, # 1 cp = 1 mPa*s
            'compressibility': 1.0e-3, # MPa^{-1}
            'initial saturation': 0.0, 
            'injection rate': 3.51e-6 # s^{-1}, 每秒注入多少水
            }
        self.oil = {'viscosity': 2, # cp
            'compressibility': 2.0e-3, # MPa^{-1}
            'initial saturation': 1.0, 
            'production rate': 3.50e-6 # s^{-1}, 每秒产出多少油
            }
        self.bc = {'displacement': 0.0, 'flux': 0.0}

        self.GD = 3

    def space_mesh(self, n=32):
        from fealpy.mesh import MeshFactory
        mf = MeshFactory()
        mesh = mf.boxmesh3d(self.domain, nx=n, ny=n, meshtype='tet')
        return mesh

    def time_mesh(self, T=1, n=100):
        from fealpy.timeintegratoralg.timeline import UniformTimeLine
        timeline = UniformTimeLine(0, T, n)
        return timeline

    def water_relative_permeability(self, Sw):
        """

        Notes
        ----
        给定水的饱和度, 计算水的相对渗透率
        """
        val = Sw**2
        return val

    def oil_relative_permeability(self, Sw):
        """

        Notes
        ----
        给定水的饱和度, 计算油的相对渗透率
        """
        val = (1 - Sw)**2 
        return val

    def gas_relative_permeability(self, Sw):
        pass

class WaterFloodingModelSolver():
    """

    Notes
    -----

    """
    def __init__(self, model, T=800*3600*24, NS=32, NT=800*24):
        self.model = model
        self.mesh = model.space_mesh(n=NS)
        self.timeline = model.time_mesh(T=T, n=NT)

        self.GD = model.GD
        if self.GD == 2:
            self.vspace = RaviartThomasFiniteElementSpace2d(self.mesh, p=0) # 速度空间
        elif self.GD == 3:
            self.vspace = RaviartThomasFiniteElementSpace3d(self.mesh, p=0)

        self.pspace = self.vspace.smspace # 压强和饱和度所属的空间, 分片常数
        self.cspace = LagrangeFiniteElementSpace(self.mesh, p=1) # 位移空间

        # 上一时刻物理量的值
        self.v = self.vspace.function() # 速度函数
        self.p = self.pspace.function() # 压强函数
        self.s = self.pspace.function() # 水的饱和度函数 默认为0, 初始时刻区域内水的饱和度为0
        self.u = self.cspace.function(dim=self.GD) # 位移函数
        self.phi = self.pspace.function() # 孔隙度函数, 分片常数

        # 当前时刻物理量的值, 用于保存临时计算出的值, 模型中系数的计算由当前时刻
        # 的物理量的值决定
        self.cv = self.vspace.function() # 速度函数
        self.cp = self.pspace.function() # 压强函数
        self.cs = self.pspace.function() # 水的饱和度函数 默认为0, 初始时刻区域内水的饱和度为0
        self.cu = self.cspace.function(dim=self.GD) # 位移函数
        self.cphi = self.pspace.function() # 孔隙度函数, 分片常数

        # 初值
        self.p[:] = model.rock['initial pressure']  # MPa
        self.phi[:] = model.rock['porosity'] # 初始孔隙度 
        self.cp[:] = model.rock['initial pressure']  # 初始地层压强
        self.cphi[:] = model.rock['porosity'] # 当前孔隙度系数

        # 源项,  TODO: 注意这里假设用的是结构网格, 换其它的网格需要修改代码
        self.fo = self.cspace.function()
        self.fo[-1] = -self.model.oil['production rate'] # 产出

        self.fw = self.cspace.function()
        self.fw[0] = self.model.water['injection rate'] # 注入


        # 一些常数矩阵和向量

        # 速度散度矩阵, 速度方程对应的散度矩阵, (\nabla\cdot v, w) 
        self.B = self.vspace.div_matrix()

        # 压强方程对应的位移散度矩阵, (\nabla\cdot u, w) 位移散度矩阵
        # * 注意这里利用了压强空间分片常数, 线性函数导数也是分片常数的事实
        c = self.mesh.entity_measure('cell')
        c *= self.model.rock['biot']

        val = self.mesh.grad_lambda() # (NC, TD+1, GD)
        val *= c[:, None, None]
        pc2d = self.pspace.cell_to_dof()
        cc2d = self.cspace.cell_to_dof()
        pgdof = self.pspace.number_of_global_dofs()
        cgdof = self.cspace.number_of_global_dofs()
        I = np.broadcast_to(pc2d, shape=cc2d.shape)
        J = cc2d 
        self.PU0 = csr_matrix(
                (val[..., 0].flat, (I.flat, J.flat)), 
                shape=(pgdof, cgdof)
                )
        self.PU1 = csr_matrix(
                (val[..., 1].flat, (I.flat, J.flat)),
                shape=(pgdof, cgdof)
                )

        if self.GD == 3:
            self.PU2 = csr_matrix(
                    (val[..., 2].flat, (I.flat, J.flat)),
                    shape=(pgdof, cgdof)
                    )

        # 线弹性矩阵的右端向量
        sigma0 = self.pspace.function()
        sigma0[:] = self.model.rock['initial stress']
        self.FU = np.zeros(self.GD*cgdof, dtype=np.float64)
        self.FU[0*cgdof:1*cgdof] -= self.p@self.PU0
        self.FU[1*cgdof:2*cgdof] -= self.p@self.PU1

        if self.GD == 3:
            self.FU[2*cgdof:3*cgdof] -= self.p@self.PU2

        # 初始应力和等效应力项
        self.FU[0*cgdof:1*cgdof] -= sigma0@self.PU0
        self.FU[1*cgdof:2*cgdof] -= sigma0@self.PU1
        if self.GD == 3:
            self.FU[2*cgdof:3*cgdof] -= sigma0@self.PU2

        # vtk 文件输出
        node, cell, cellType, NC = self.mesh.to_vtk()
        self.points = vtk.vtkPoints()
        self.points.SetData(vnp.numpy_to_vtk(node))
        self.cells = vtk.vtkCellArray()
        self.cells.SetCells(NC, vnp.numpy_to_vtkIdTypeArray(cell))
        self.cellType = cellType

    def recover(self, val):
        """

        Notes
        -----
        给定一个分片常数的量, 恢复为分片连续的量
        """

        mesh = self.mesh
        cell = self.mesh.entity('cell')
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()

        w = 1/self.mesh.entity_measure('cell') # 恢复权重
        w = np.broadcast_to(w[:, None], shape=cell.shape)

        r = np.zeros(NN, dtype=np.float64)
        d = np.zeros(NN, dtype=np.float64)

        np.add.at(d, cell, w)
        np.add.at(r, cell, val[:, None]*w)

        return r/d


    def pressure_coefficient(self):

        """

        Notes
        -----
        计算当前物理量下的压强质量矩阵系数
        """

        b = self.model.rock['biot'] 
        Ks = self.model.rock['solid grain stiffness']
        cw = self.model.water['compressibility']
        co = self.model.oil['compressibility']

        Sw = self.cs[:].copy() # 当前的水饱和度 (NQ, NC)
        phi = self.cphi[:].copy() # 当前的孔隙度

        val = phi*Sw*cw
        val += phi*(1 - Sw)*co # 注意这里的 co 是常数, 但在水气混合物条件下应该依赖于压力
        val += (b - phi)/Ks
        return val

    def saturation_pressure_coefficient(self):
        """

        Notes
        -----
        计算当前物理量下, 饱和度方程中, 压强项对应的系数
        """

        b = self.model.rock['biot'] 
        Ks = self.model.rock['solid grain stiffness']
        cw = self.model.water['compressibility']

        val = self.cs[:].copy() # 当前水饱和度
        phi = self.cphi[:].copy() # 当前孔隙度

        val *= (b-phi)/Ks + phi*cw

        return val

    def flux_coefficient(self):
        """
        Notes
        -----

        计算**当前**物理量下, 速度方程对应的系数

        计算通量的系数, 流体的相对渗透率除以粘性系数得到流动性系数 
        krg/mu_g 是气体的流动性系数 
        kro/mu_o 是油的流动性系数
        krw/mu_w 是水的流动性系数

        这里假设压强的单位是 MPa 

        1 d = 9.869 233e-13 m^2 = 1000 md
        1 cp = 1 mPa s = 1e-9 MPa.s

        """

        muw = self.model.water['viscosity'] # 单位是 1 cp = 1 mPa.s
        muo = self.model.oil['viscosity'] # 单位是 1 cp = 1 mPa.s 

        # 岩石的绝对渗透率, 这里考虑了量纲的一致性, 压强是 MPa
        k = self.model.rock['permeability']*9.869233e-4  

     
        Sw = self.cs[:].copy() # 当前水的饱和度系数

        lamw = self.model.water_relative_permeability(Sw)
        lamw /= muw
        lamo = self.model.oil_relative_permeability(Sw)
        lamo /= muo

        val = 1/(lamw + lamo)/k # 

        return val

    def water_fractional_flow_coefficient(self):
        """

        Notes
        -----

        计算**当前**物理量下, 饱和度方程中, 水的流动性系数
        """

        Sw = self.cs[:].copy() # 当前水的饱和度系数
        lamw = self.model.water_relative_permeability(Sw)
        lamw /= self.model.water['viscosity'] 
        lamo = self.model.oil_relative_permeability(Sw)
        lamo /= self.model.oil['viscosity'] 
        val = lamw/(lamw + lamo)
        return val




    def get_total_system(self):
        """
        Notes
        -----
        构造整个系统

        二维情形：

        x = [v, p, s, u0, u1]

        A = [[   V,  VP, None, None,  None]
             [  PV,   P, None,  PU0,   PU1]
             [None,  SP,    S,  SU0,   SU1] 
             [None, UP0, None,  U00,   U01]
             [None, UP1, None,  U10,   U11]

        F = [FV, FP, FS, FU0, FU1]

        三维情形：
        x = [v, p, s, u0, u1, u2]

        A = [[   V,  VP, None, None,  None, None]
             [  PV,   P, None,  PU0,   PU1,  PU2]
             [None,  SP,    S,  SU0,   SU1,  SU2] 
             [None, UP0, None,  U00,   U01,  U02]
             [None, UP1, None,  U10,   U11,  U12]
             [None, UP2, None,  U20,   U21,  U22]]

        F = [FV, FP, FS, FU0, FU1, FU2]

        FS 中考虑的迎风格式
        """

        GD = self.GD
        A0, FV, isBdDof0 = self.get_velocity_system(q=2)
        A1, FP, isBdDof1 = self.get_pressure_system(q=2)
        A2, FS, isBdDof2 = self.get_saturation_system(q=2)

        if GD == 2:
            A3, A4, FU, isBdDof3 = self.get_dispacement_system(q=2)
            A = bmat([A0, A1, A2, A3, A4], format='csr')
            F = np.r_['0', FV, FP, FS, FU]
        elif GD == 3:
            A3, A4, A5, FU, isBdDof3 = self.get_dispacement_system(q=2)
            A = bmat([A0, A1, A2, A3, A4, A5], format='csr')
            F = np.r_['0', FV, FP, FS, FU]

        isBdDof = np.r_['0', isBdDof0, isBdDof1, isBdDof2, isBdDof3]
        return A, F, isBdDof

    def get_velocity_system(self, q=2):
        """
        Notes
        -----
        计算速度方程对应的离散系统.

        if GD == 2:
            [   V,  VP, None, None,  None]
        elif GD == 3:
            [   V,  VP, None, None,  None, None]

        """

        GD = self.GD
        dt = self.timeline.current_time_step_length()
        cellmeasure = self.mesh.entity_measure('cell')
        qf = self.mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        # 速度对应的矩阵  V
        c = self.flux_coefficient()
        c *= cellmeasure
        phi = self.vspace.basis(bcs)
        V = np.einsum('q, qcin, qcjn, c->cij', ws, phi, phi, c, optimize=True)

        c2d = self.vspace.cell_to_dof()
        I = np.broadcast_to(c2d[:, :, None], shape=V.shape)
        J = np.broadcast_to(c2d[:, None, :], shape=V.shape)

        gdof = self.vspace.number_of_global_dofs()
        V = csr_matrix(
                (V.flat, (I.flat, J.flat)), 
                shape=(gdof, gdof)
                )

        # 压强矩阵
        VP = -self.B

        # 右端向量, 0 向量
        FV = np.zeros(gdof, dtype=np.float64)
        isBdDof = self.vspace.dof.is_boundary_dof()

        if GD == 2:
            return [V, VP, None, None, None], FV, isBdDof 
        elif GD == 3:
            return [V, VP, None, None, None, None], FV, isBdDof 

    def get_pressure_system(self, q=2):
        """
        Notes
        -----
        计算压强方程对应的离散系统

        这里组装矩阵时, 利用了压强是分片常数的特殊性 

        [  PV,   P, None,  PU0,   PU1]

        [  PV,   P, None,  PU0,   PU1, PU2]
        """

        GD = self.GD
        dt = self.timeline.current_time_step_length()
        cellmeasure = self.mesh.entity_measure('cell')
        qf = self.mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        PV = dt*self.B.T

        # P 是对角矩阵, 利用分片常数的
        c = self.pressure_coefficient() # (NQ, NC)
        c *= cellmeasure 
        P = diags(c, 0)

        # 组装压强方程的右端向量
        # * 这里利用了压强空间基是分片常数
        FP = self.fo.value(bcs) + self.fw.value(bcs) # (NQ, NC)
        FP *= ws[:, None]

        FP = np.sum(FP, axis=0)
        FP *= cellmeasure
        FP *= dt

        FP += P@self.p # 上一步的压强向量
        FP += self.PU0@self.u[:, 0] 
        FP += self.PU1@self.u[:, 1]

        if GD == 3:
            FP += self.PU2@self.u[:, 2]

        gdof = self.pspace.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)

        if GD == 2:
            return [PV, P, None, self.PU0, self.PU1], FP, isBdDof
        elif GD == 3:
            return [PV, P, None, self.PU0, self.PU1, self.PU2], FP, isBdDof

    def get_saturation_system(self, q=2):
        """
        Notes
        ----
        计算饱和度方程对应的离散系统

        [ None,  SP,    S, SU0,  SU1] 

        """

        GD = self.GD
        qf = self.mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        cellmeasure = self.mesh.entity_measure('cell')

        # SP 是对角矩阵 
        c = self.saturation_pressure_coefficient() # (NC, )
        c *= cellmeasure 
        SP = diags(c, 0)

        # S 质量矩阵组装, 对角矩阵
        c = self.cphi[:]*cellmeasure
        S = diags(c, 0)

        # SU0, SU1, 饱和度方程中的位移散度对应的矩阵
        val = self.mesh.grad_lambda() # (NC, TD+1, GD)
        c = self.cs[:]*self.model.rock['biot'] # (NC, ), 注意用当前的水饱和度
        c *= cellmeasure 
        val *= c[:, None, None]

        pgdof = self.pspace.number_of_global_dofs() # 压力空间自由度个数
        cgdof = self.cspace.number_of_global_dofs() # 连续空间自由度个数

        pc2d = self.pspace.cell_to_dof()
        cc2d = self.cspace.cell_to_dof()

        I = np.broadcast_to(pc2d, shape=cc2d.shape)
        J = cc2d 

        SU0 = csr_matrix(
                (val[..., 0].flat, (I.flat, J.flat)),
                shape=(pgdof, cgdof)
                )
        SU1 = csr_matrix(
                (val[..., 1].flat, (I.flat, J.flat)),
                shape=(pgdof, cgdof)
                )

        if GD == 3:
            SU2 = csr_matrix(
                    (val[..., 2].flat, (I.flat, J.flat)),
                    shape=(pgdof, cgdof)
                    )



        # 右端矩阵
        dt = self.timeline.current_time_step_length()
        FS = self.fw.value(bcs) # (NQ, NC)
        FS *= ws[:, None]
        FS = np.sum(FS, axis=0)
        FS *= cellmeasure
        FS *= dt

        FS += SP@self.p # 上一时间步的压强 
        FS += S@self.s # 上一时间步的饱和度
        FS += SU0@self.u[:, 0] # 上一时间步的位移 x 分量
        FS += SU1@self.u[:, 1] # 上一个时间步的位移 y 分量 

        if GD == 3:
            FS += SU2@self.u[:, 2] # 上一个时间步的位移 y 分量 


        # 用当前时刻的速度场, 构造非线性迎风格式
        face2cell = self.mesh.ds.face_to_cell()
        isBdFace = face2cell[:, 0] == face2cell[:, 1]

        qf = self.mesh.integrator(2, 'face') # 边上的积分公式
        bcs, ws = qf.get_quadrature_points_and_weights()
        facemeasure = self.mesh.entity_measure('face') 

        # 边的定向法线，它是左边单元的外法线，右边单元内法线。
        fn = self.mesh.face_unit_normal() 
        val0 = np.einsum('qfm, fm->qf', self.cv.face_value(bcs), fn) # 当前速度和法线的内积

        # 水的流动分数, 与水的饱和度有关, 如果饱和度为0, 则为 0
        Fw = self.water_fractional_flow_coefficient()
        val1 = Fw[face2cell[:, 0]] # 边的左边单元的水流动分数
        val2 = Fw[face2cell[:, 1]] # 边的右边单元的水流动分数 
        val2[isBdFace] = 0.0 # 边界的贡献是 0 

        flag = val0 < 0.0 # 对于左边单元来说，是流出项
                           # 对于右边单元来说，是流入项

        # 左右单元流入流出的绝对量是一样的
        val = val0*val1[None, :] # val0 >= 0.0, 左边单元是流出
                                 # val0 < 0.0, 左边单元是流入
        val[flag] = (val0*val2[None, :])[flag] # val0 >= 0, 右边单元是流入
                                               # val0 < 0, 右边单元是流出

        b = np.einsum('q, qf, f->f', ws, val, facemeasure)
        b *= dt
        np.subtract.at(FS, face2cell[:, 0], b)  

        isInFace = ~isBdFace # 只处理内部边
        np.add.at(FS, face2cell[isInFace, 1], b[isInFace])  


        isBdDof = np.zeros(pgdof, dtype=np.bool_) 

        if GD == 2:
            return [None, SP, S, SU0, SU1], FS, isBdDof
        elif GD == 3:
            return [None, SP, S, SU0, SU1, SU2], FS, isBdDof

    def get_dispacement_system(self, q=2):
        """
        Notes
        -----
        计算位移方程对应的离散系统, 即线弹性方程系统.
        
        GD == 2:

         [None, UP0, None,  U00,   U01]
         [None, UP1, None,  U10,   U11]

        GD == 3:
         [None, UP0, None,  U00,   U01, U02]
         [None, UP1, None,  U10,   U11, U12]
         [None, UP2, None,  U20,   U21, U22]

        """


        GD = self.GD
        # 拉梅参数 (lambda, mu)
        lam, mu = self.model.rock['lame']
        U = self.cspace.linear_elasticity_matrix(lam, mu, format='list')

        isBdDof = self.cspace.dof.is_boundary_dof()

        if GD == 2:
            return (
                    [None, -self.PU0.T, None, U[0][0], U[0][1]], 
                    [None, -self.PU1.T, None, U[1][0], U[1][1]], 
                    self.FU, np.r_['0', isBdDof, isBdDof]
                    )
        elif GD == 3:
            return (
                    [None, -self.PU0.T, None, U[0][0], U[0][1], U[0][2]], 
                    [None, -self.PU1.T, None, U[1][0], U[1][1], U[1][2]], 
                    [None, -self.PU2.T, None, U[2][0], U[2][1], U[2][2]], 
                    self.FU, np.r_['0', isBdDof, isBdDof, isBdDof]
                    )

    def picard_iteration(self, maxit=10):

        GD = self.GD
        e0 = 1.0
        k = 0
        while e0 > 1e-10: 
            # 构建总系统
            A, F, isBdDof = self.get_total_system()

            # 处理边界条件, 这里是 0 边界
            gdof = len(isBdDof)
            bdIdx = np.zeros(gdof, dtype=np.int_)
            bdIdx[isBdDof] = 1 
            Tbd = diags(bdIdx)
            T = diags(1-bdIdx)
            A = T@A@T + Tbd
            F[isBdDof] = 0.0

            # 求解

            x = spsolve(A, F)

            vgdof = self.vspace.number_of_global_dofs()
            pgdof = self.pspace.number_of_global_dofs()
            cgdof = self.cspace.number_of_global_dofs()

            e0 = 0.0
            start = 0
            end = vgdof
            self.cv[:] = x[start:end]


            start = end
            end += pgdof
            e0 += np.sum((self.cp - x[start:end])**2)
            self.cp[:] = x[start:end]


            start = end
            end += pgdof

            e0 += np.sum((self.cs - x[start:end])**2)
            self.cs[:] = x[start:end]
            e0 = np.sqrt(e0) # 误差的 l2 norm
            k += 1

            for i in range(GD):
                start = end
                end += cgdof
                self.cu[:, i] = x[start:end]

            print(e0)

            if k >= maxit: 
                print('picard iteration arrive max iteration with error:', e0)
                break

    def update_solution(self):
        self.v[:] = self.cv
        self.p[:] = self.cp
        self.s[:] = self.cs
        #flag = self.s < 0.0
        #self.s[flag] = 0.0
        self.u[:] = self.cu

    def solve(self, step=24):
        """

        Notes
        -----

        计算所有的时间层。
        """

        timeline = self.timeline
        dt = timeline.current_time_step_length()
        timeline.reset() # 时间置零

        n = timeline.current
        fname = 'test_'+ str(n).zfill(10) + '.vtu'
        self.write_to_vtk(fname)
        while not timeline.stop():
            ct = timeline.current_time_level()/3600/24 # 天为单位
            print('当前时刻为第', ct, '天')
            self.picard_iteration()
            self.update_solution()
            timeline.current += 1
            if timeline.current%step == 0:
                n = timeline.current
                fname = 'test_'+ str(n).zfill(10) + '.vtu'
                self.write_to_vtk(fname)
        timeline.reset()

    def write_to_vtk(self, fname):
        # 重心处的值
        mesh = self.mesh

        GD = self.GD

        bc = np.array((GD+1)*[1/(GD+1)], dtype=np.float64)

        ps = self.mesh.bc_to_point(bc)
        vmesh = vtk.vtkUnstructuredGrid()
        vmesh.SetPoints(self.points)
        vmesh.SetCells(self.cellType, self.cells)
        cdata = vmesh.GetCellData()
        pdata = vmesh.GetPointData()

        v = self.v 
        p = self.p
        s = self.s
        u = self.u

        val = v.value(bc)
        if GD == 2:
            val = np.concatenate((val, np.zeros((val.shape[0], 1), dtype=val.dtype)), axis=1)
        val = vnp.numpy_to_vtk(val)
        val.SetName('velocity')
        cdata.AddArray(val)

        val = self.recover(p[:])
        val = vnp.numpy_to_vtk(val)
        val.SetName('pressure')
        pdata.AddArray(val)

        val = self.recover(s[:])
        val = vnp.numpy_to_vtk(val)
        val.SetName('saturation')
        pdata.AddArray(val)

        if GD == 2:
            val = np.concatenate((u[:], np.zeros((u.shape[0], 1), dtype=u.dtype)), axis=1)
        val = vnp.numpy_to_vtk(val)
        val.SetName('displacement')
        pdata.AddArray(val)

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(vmesh)
        writer.Write()


if __name__ == '__main__':

    model = WaterFloodingModel2d()
    solver = WaterFloodingModelSolver(model)

    solver.solve()

