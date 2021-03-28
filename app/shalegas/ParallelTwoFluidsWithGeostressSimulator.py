import numpy as np

import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, coo_matrix, bmat, diags
from scipy.sparse.linalg import spsolve

from fealpy.decorator import barycentric
from fealpy.timeintegratoralg.timeline import UniformTimeLine
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.functionspace import RaviartThomasFiniteElementSpace3d

import pyamg 

from fast_solver import FastSover


class ParallelTwoFluidsWithGeostressSimulator():
    """

    Notes
    -----
    这是一个并行模拟两种流体和地质力学耦合的程序, 这里只是线性系统是并行的  

    * S_0: 流体 0 的饱和度，一般是水
    * S_1: 流体 1 的饱和度，可以为气或油
    * v: 总速度
    * p: 压力
    * u: 岩石位移  

    饱和度 P0 间断元，单元边界用迎风格式
    速度 RT0 元
    压强 P0 元
    岩石位移 P1 连续元离散

    目前, 模型
    * 忽略了毛细管压强和重力作用
    * 饱和度用分片线性间断元求解, 非线性的迎风格式

    渐近解决方案:
    1. Picard 迭代
    2. 气的可压性随着压强的变化而变化
    3. 考虑渗透率随着孔隙度的变化而变化 
    4. 考虑裂缝，裂缝用自适应网格加密，设设置更大的渗透率实现

    体积模量:  K = E/3/(1 - 2*nu) = lambda + 2/3* mu

    Develop
    ------

    """
    def __init__(self, mesh, args, ctx):
        self.ctx = ctx
        self.args = args # 模拟相关参数

        NT = int((args.T1 - args.T0)/args.DT)
        self.timeline = UniformTimeLine(args.T0, args.T1, NT)
        self.mesh = mesh 

        if self.ctx.myid == 0:
            self.GD = mesh.geo_dimension()
            if self.GD == 2:
                self.vspace = RaviartThomasFiniteElementSpace2d(self.mesh, p=0) # 速度空间
            elif self.GD == 3:
                self.vspace = RaviartThomasFiniteElementSpace3d(self.mesh, p=0)

            self.pspace = self.vspace.smspace # 压力和饱和度所属的空间, 分片常数
            self.cspace = LagrangeFiniteElementSpace(self.mesh, p=1) # 位移空间

            # 上一时刻物理量的值
            self.v = self.vspace.function() # 速度函数
            self.p = self.pspace.function() # 压力函数
            self.s = self.pspace.function() # 水的饱和度函数 默认为0, 初始时刻区域内水的饱和度为0
            self.u = self.cspace.function(dim=self.GD) # 位移函数
            self.phi = self.pspace.function() # 孔隙度函数, 分片常数

            # 当前时刻物理量的值, 用于保存临时计算出的值, 模型中系数的计算由当前时刻
            # 的物理量的值决定
            self.cv = self.vspace.function() # 速度函数
            self.cp = self.pspace.function() # 压力函数
            self.cs = self.pspace.function() # 水的饱和度函数 默认为0, 初始时刻区域内水的饱和度为0
            self.cu = self.cspace.function(dim=self.GD) # 位移函数
            self.cphi = self.pspace.function() # 孔隙度函数, 分片常数

            # 初值
            self.s[:] = self.mesh.celldata['fluid_0']
            self.p[:] = self.mesh.celldata['pressure_0']  # MPa
            self.phi[:] = self.mesh.celldata['porosity_0'] # 初始孔隙度 

            self.s[:] = self.mesh.celldata['fluid_0']
            self.cp[:] = self.p # 初始地层压力
            self.cphi[:] = self.phi # 当前孔隙度系数

            # 源汇项 
            self.f0 = self.cspace.function()
            self.f1 = self.cspace.function()

            self.f0[:] = self.mesh.nodedata['source_0'] # 流体 0 的源汇项
            self.f1[:] = self.mesh.nodedata['source_1'] # 流体 1 的源汇项

            # 一些常数矩阵和向量

            # 速度散度矩阵, 速度方程对应的散度矩阵, (\nabla\cdot v, w) 
            self.B = self.vspace.div_matrix()

            # 压力方程对应的位移散度矩阵, (\nabla\cdot u, w) 位移散度矩阵
            # * 注意这里利用了压力空间分片常数, 线性函数导数也是分片常数的事实
            c = self.mesh.entity_measure('cell')
            c *= self.mesh.celldata['biot']

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
            self.FU = np.zeros(self.GD*cgdof, dtype=np.float64)
            self.FU[0*cgdof:1*cgdof] -= self.p@self.PU0
            self.FU[1*cgdof:2*cgdof] -= self.p@self.PU1

            if self.GD == 3:
                self.FU[2*cgdof:3*cgdof] -= self.p@self.PU2

            # 初始应力和等效应力项
            sigma = self.mesh.celldata['stress_0'] + self.mesh.celldata['stress_eff']# 初始应力和等效应力之和
            self.FU[0*cgdof:1*cgdof] -= sigma0@self.PU0
            self.FU[1*cgdof:2*cgdof] -= sigma0@self.PU1
            if self.GD == 3:
                self.FU[2*cgdof:3*cgdof] -= sigma0@self.PU2

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

    def add_time(self, n):
        """

        Notes
        ----

        增加 n 步的计算时间 
        """

        if self.ctx.myid == 0:
            self.timeline.add_time(n)


    def pressure_coefficient(self):

        """

        Notes
        -----
        计算当前物理量下的压强质量矩阵系数
        """

        b = self.mesh.celldata['biot'] 
        K = self.mesh.celldata['K']
        c0 = self.mesh.meshdata['fluid_0']['compressibility']
        c1 = self.mesh.meshdata['fluid_1']['compressibility']

        s = self.cs # 流体 0 当前的饱和度 (NQ, NC)
        phi = self.cphi # 当前的孔隙度

        val = phi*s*c0  
        val += phi*(1 - s)*c1 # 注意这里的 co 是常数, 但在水气混合物条件下应该依赖于压力
        val += (b - phi)/K
        return val

    def saturation_pressure_coefficient(self):
        """

        Notes
        -----
        计算当前物理量下, 饱和度方程中, 压强项对应的系数
        """

        b = self.mesh.celldata['biot'] 
        K = self.mesh.celldata['K']
        c0 = self.mesh.meshdata['fluid_0']['compressibility']

        s = self.cs # 当前流体 0 的饱和度
        phi = self.cphi # 当前孔隙度

        val  = (b - phi)/K
        val += phi*c0
        val *= s 
        return val

    def flux_coefficient(self):
        """
        Notes
        -----

        计算**当前**物理量下, 速度方程对应的系数

        计算通量的系数, 流体的相对渗透率除以粘性系数得到流动性系数 
        k_0/mu_0 是气体的流动性系数 
        k_1/mu_1 是油的流动性系数

        这里假设压强的单位是 MPa 

        1 d = 9.869 233e-13 m^2 = 1000 md
        1 cp = 1 mPa s = 1e-9 MPa*s = 1e-3 Pa*s

        """

        mu0 = self.mesh.meshdata['fluid_0']['viscosity'] # 单位是 1 cp = 1 mPa.s
        mu1 = self.mesh.meshdata['fluid_1']['viscosity'] # 单位是 1 cp = 1 mPa.s 

        # 岩石的绝对渗透率, 这里考虑了量纲的一致性, 压强单位是 Pa
        k = self.mesh.celldata['permeability']*9.869233e-4 

        s = self.cs # 流体 0 当前的饱和度系数

        lam0 = self.mesh.fluid_relative_permeability_0(s) # TODO: 考虑更复杂的饱和度和渗透的关系 
        lam0 /= mu0

        lam1 = self.mesh.fluid_relative_permeability_1(s) # TODO: 考虑更复杂的饱和度和渗透的关系 
        lam1 /= mu1

        val = 1/(lam0 + lam1)
        val /=k 

        return val

    def fluid_fractional_flow_coefficient_0(self):
        """

        Notes
        -----

        计算**当前**物理量下, 饱和度方程中, 主流体的的流动性系数
        """


        s = self.cs # 流体 0 当前的饱和度系数

        mu0 = self.mesh.meshdata['fluid_0']['viscosity'] # 单位是 1 cp = 1 mPa.s
        lam0 = self.mesh.fluid_relative_permeability_0(s) 
        lam0 /= mu0

        lam1 = self.mesh.fluid_relative_permeability_1(s) 
        mu1 = self.mesh.meshdata['fluid_1']['viscosity'] # 单位是 1 cp = 1 mPa.s 
        lam1 /= mu1 

        val = lam0/(lam0 + lam1)
        return val


    def get_total_system(self):
        """
        Notes
        -----
        构造整个系统

        二维情形：
        x = [s, v, p, u0, u1]

        A = [[   S, None,   SP,  SU0,  SU1]
             [None,    V,   VP, None, None]
             [None,   PV,    P,  PU0,  PU1]
             [None, None,  UP0,  U00,  U01]
             [None, None,  UP1,  U10,  U11]]
        F = [FS, FV, FP, FU0, FU1]

        三维情形：

        x = [s, v, p, u0, u1, u2]
        A = [[   S, None,   SP,  SU0,  SU1,  SU2]
             [None,    V,   VP, None, None, None]
             [None,   PV,    P,  PU0,  PU1,  PU2]
             [None, None,  UP0,  U00,  U01,  U02]
             [None, None,  UP1,  U10,  U11,  U12]
             [None, None,  UP2,  U20,  U21,  U22]]
        F = [FS, FV, FP, FU0, FU1, FU2]

        """

        GD = self.GD
        A0, FS, isBdDof0 = self.get_saturation_system(q=2)
        A1, FV, isBdDof1 = self.get_velocity_system(q=2)
        A2, FP, isBdDof2 = self.get_pressure_system(q=2)

        if GD == 2:
            A3, A4, FU, isBdDof3 = self.get_dispacement_system(q=2)
            A = bmat([A0, A1, A2, A3, A4], format='csr')
            F = np.r_['0', FS, FV, FP, FU]
        elif GD == 3:
            A3, A4, A5, FU, isBdDof3 = self.get_dispacement_system(q=2)
            A = bmat([A0, A1, A2, A3, A4, A5], format='csr')
            F = np.r_['0', FS, FV, FP, FU]
        isBdDof = np.r_['0', isBdDof0, isBdDof1, isBdDof2, isBdDof3]
        return A, F, isBdDof

    def get_saturation_system(self, q=2):
        """
        Notes
        ----
        计算饱和度方程对应的离散系统

        [   S, None,   SP,  SU0,  SU1]

        [   S, None,   SP,  SU0,  SU1, SU2]
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
        b = self.mesh.celldata['biot']
        val = self.mesh.grad_lambda() # (NC, TD+1, GD)
        c = self.cs[:]*b # (NC, ), 注意用当前的水饱和度
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
        FS = self.f0.value(bcs) # (NQ, NC)
        FS *= ws[:, None]
        FS = np.sum(FS, axis=0)
        FS *= cellmeasure
        FS *= dt

        FS += SP@self.p # 上一时间步的压强 
        FS += S@self.s # 上一时间步的饱和度
        FS += SU0@self.u[:, 0] # 上一时间步的位移 x 分量
        FS += SU1@self.u[:, 1] # 上一个时间步的位移 y 分量 
        if GD == 3:
            FS += SU2@self.u[:, 2] # 上一个时间步的位移 z 分量 


        # 用当前时刻的速度场, 构造迎风格式
        face2cell = self.mesh.ds.face_to_cell()
        isBdFace = face2cell[:, 0] == face2cell[:, 1]

        qf = self.mesh.integrator(2, 'face') # 边或面上的积分公式
        bcs, ws = qf.get_quadrature_points_and_weights()
        facemeasure = self.mesh.entity_measure('face') 

        # 边的定向法线，它是左边单元的外法线，右边单元内法线。
        fn = self.mesh.face_unit_normal() 
        val0 = np.einsum('qfm, fm->qf', self.cv.face_value(bcs), fn) # 当前速度和法线的内积

        # 流体 0 的流动分数, 与水的饱和度有关, 如果饱和度为0, 则为 0
        F0 = self.fluid_fractional_flow_coefficient_0()
        val1 = F0[face2cell[:, 0]] # 边的左边单元的水流动分数
        val2 = F0[face2cell[:, 1]] # 边的右边单元的水流动分数 
        val2[isBdFace] = 0.0 # 边界的贡献是 0，没有流入和流出

        flag = val0 < 0.0 # 对于左边单元来说，是流入项
                           # 对于右边单元来说，是流出项

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
            return [   S, None,   SP,  SU0,  SU1], FS, isBdDof
        elif GD == 3:
            return [   S, None,   SP,  SU0,  SU1, SU2], FS, isBdDof

    def get_velocity_system(self, q=2):
        """
        Notes
        -----
        计算速度方程对应的离散系统.


         [None,    V,   VP, None, None]

         [None,    V,   VP, None, None, None]
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
            return [None, V, VP, None, None], FV, isBdDof 
        elif GD == 3:
            return [None, V, VP, None, None, None], FV, isBdDof 

    def get_pressure_system(self, q=2):
        """
        Notes
        -----
        计算压强方程对应的离散系统

        这里组装矩阵时, 利用了压强是分片常数的特殊性 

        [  Noe,  PV,   P, PU0,   PU1]

        [  None, PV,   P, PU0,   PU1, PU2]
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

        # 组装压力方程的右端向量
        # * 这里利用了压力空间基是分片常数
        FP = self.f1.value(bcs) + self.f0.value(bcs) # (NQ, NC)
        FP *= ws[:, None]

        FP = np.sum(FP, axis=0)
        FP *= cellmeasure
        FP *= dt

        FP += P@self.p # 上一步的压力向量
        FP += self.PU0@self.u[:, 0] 
        FP += self.PU1@self.u[:, 1]

        if GD == 3:
            FP += self.PU2@self.u[:, 2]

        gdof = self.pspace.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)

        if GD == 2:
            return [None, PV, P, self.PU0, self.PU1], FP, isBdDof
        elif GD == 3:
            return [None, PV, P, self.PU0, self.PU1, self.PU2], FP, isBdDof


    def get_dispacement_system(self, q=2):
        """
        Notes
        -----
        计算位移方程对应的离散系统, 即线弹性方程系统.
        
        GD == 2:
         [None, None, UP0, U00,   U01]
         [None, None, UP1, U10,   U11]

        GD == 3:
         [None, None, UP0, U00, U01, U02]
         [None, None, UP1, U10, U11, U12]
         [None, None, UP2, U20, U21, U22]

        """


        GD = self.GD
        # 拉梅参数 (lambda, mu)
        lam = self.mesh.celldata['lambda']
        mu = self.mesh.celldata['mu']
        U = self.linear_elasticity_matrix(lam, mu, format='list')

        isBdDof = self.cspace.dof.is_boundary_dof()

        if GD == 2:
            return (
                    [None, None, -self.PU0.T, U[0][0], U[0][1]], 
                    [None, None, -self.PU1.T, U[1][0], U[1][1]], 
                    self.FU, np.r_['0', isBdDof, isBdDof]
                    )
        elif GD == 3:
            return (
                    [None, None, -self.PU0.T, U[0][0], U[0][1], U[0][2]], 
                    [None, None, -self.PU1.T, U[1][0], U[1][1], U[1][2]], 
                    [None, None, -self.PU2.T, U[2][0], U[2][1], U[2][2]], 
                    self.FU, np.r_['0', isBdDof, isBdDof, isBdDof]
                    )

    def linear_elasticity_matrix(self, lam, mu, format='csr', q=None):
        """
        Notes
        -----
        注意这里的拉梅常数是单元分片常数
        lam.shape == (NC, ) # MPa
        mu.shape == (NC, ) # MPa
        """

        GD = self.GD
        if GD == 2:
            idx = [(0, 0), (0, 1),  (1, 1)]
            imap = {(0, 0):0, (0, 1):1, (1, 1):2}
        elif GD == 3:
            idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
            imap = {(0, 0):0, (0, 1):1, (0, 2):2, (1, 1):3, (1, 2):4, (2, 2):5}
        A = []

        qf = self.cspace.integrator if q is None else self.mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        grad = self.cspace.grad_basis(bcs) # (NQ, NC, ldof, GD)


        # 分块组装矩阵
        gdof = self.cspace.number_of_global_dofs()
        cellmeasure = self.cspace.cellmeasure
        for k, (i, j) in enumerate(idx):
            Aij = np.einsum('i, ijm, ijn, j->jmn', ws, grad[..., i], grad[..., j], cellmeasure)
            A.append(Aij)

        if GD == 2:
            C = [[None, None], [None, None]]
            D = mu[:, None, None]*(A[0] + A[2]) 
        elif GD == 3:
            C = [[None, None, None], [None, None, None], [None, None, None]]
            D = mu[:, None, None]*(A[0] + A[3] + A[5])

        
        cell2dof = self.cspace.cell_to_dof() # (NC, ldof)
        ldof = self.cspace.number_of_local_dofs()
        NC = self.mesh.number_of_cells()
        shape = (NC, ldof, ldof)
        I = np.broadcast_to(cell2dof[:, :, None], shape=shape)
        J = np.broadcast_to(cell2dof[:, None, :], shape=shape)

        for i in range(GD):
            Aii = D + (mu + lam)[:, None, None]*A[imap[(i, i)]] 
            C[i][i] = csr_matrix((Aii.flat, (I.flat, J.flat)), shape=(gdof, gdof))
            for j in range(i+1, GD):
                Aij = lam[:, None, None]*A[imap[(i, j)]] + mu[:, None, None]*A[imap[(i, j)]].swapaxes(-1, -2)
                C[i][j] = csr_matrix((Aij.flat, (I.flat, J.flat)), shape=(gdof, gdof)) 
                C[j][i] = C[i][j].T 

        if format == 'csr':
            return bmat(C, format='csr') # format = bsr ??
        elif format == 'bsr':
            return bmat(C, format='bsr')
        elif format == 'list':
            return C

    def solve_linear_system_0(self):
        # 构建总系统

        if self.ctx.myid == 0:
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
            self.ctx.set_centralized_sparse(A)
            x = F.copy()
            self.ctx.set_rhs(x) # Modified in place

        self.ctx.run(job=6) # Analysis + Factorization + Solve

        if self.ctx.myid == 0:
            vgdof = self.vspace.number_of_global_dofs()
            pgdof = self.pspace.number_of_global_dofs()
            cgdof = self.cspace.number_of_global_dofs()
            
            start = 0
            end = pgdof
            s = x[start:end]

            start = end
            end += vgdof
            v = x[start:end]

            start = end
            end += pgdof
            p = x[start:end]

            start = end
            u = x[start:]

            return s, v, p, u



    def solve_linear_system_1(self):
        """
        Notes
        -----
        构造整个系统

        二维情形：
        x = [s, v, p, u0, u1]

        A = [[   S, None,   SP,  SU0,  SU1]
             [None,    V,   VP, None, None]
             [None,   PV,    P,  PU0,  PU1]
             [None, None,  UP0,  U00,  U01]
             [None, None,  UP1,  U10,  U11]]
        F = [FS, FV, FP, FU0, FU1]

        三维情形：

        x = [s, v, p, u0, u1, u2]
        A = [[   S, None,   SP,  SU0,  SU1,  SU2]
             [None,    V,   VP, None, None, None]
             [None,   PV,    P,  PU0,  PU1,  PU2]
             [None, None,  UP0,  U00,  U01,  U02]
             [None, None,  UP1,  U10,  U11,  U12]
             [None, None,  UP2,  U20,  U21,  U22]]
        F = [FS, FV, FP, FU0, FU1, FU2]

        """
        if self.ctx.myid == 0:
            vgdof = self.vspace.number_of_global_dofs()
            pgdof = self.pspace.number_of_global_dofs()
            cgdof = self.cspace.number_of_global_dofs()

            GD = self.GD
            A0, FS, isBdDof0 = self.get_saturation_system(q=2)
            A1, FV, isBdDof1 = self.get_velocity_system(q=2)
            A2, FP, isBdDof2 = self.get_pressure_system(q=2)

            if GD == 2:
                A3, A4, FU, isBdDof3 = self.get_dispacement_system(q=2)
                A = bmat([A1[1:], A2[1:], A3[1:], A4[1:]], format='csr')
                F = np.r_['0', FV, FP, FU]
            elif GD == 3:
                A3, A4, A5, FU, isBdDof3 = self.get_dispacement_system(q=2)
                A = bmat([A1[1:], A2[1:], A3[1:], A4[1:], A5[1:]], format='csr')
                F = np.r_['0', FV, FP, FU]

            isBdDof = np.r_['0', isBdDof1, isBdDof2, isBdDof3]

            gdof = len(isBdDof)
            bdIdx = np.zeros(gdof, dtype=np.int_)
            bdIdx[isBdDof] = 1 
            Tbd = diags(bdIdx)
            T = diags(1-bdIdx)
            A = T@A@T + Tbd
            F[isBdDof] = 0.0

        #[   S, None,   SP,  SU0,  SU1, SU2]
            self.ctx.set_centralized_sparse(A)
            x = F.copy()
            self.ctx.set_rhs(x) # Modified in place

        self.ctx.run(job=6) # Analysis + Factorization + Solve

        if self.ctx.myid == 0:

            v = x[0:vgdof]
            p = x[vgdof:vgdof+pgdof]
            u = x[vgdof+pgdof:]

            s = FS
            s -= A0[2]@p 
            s -= A0[3]@u[0*cgdof:1*cgdof] 
            s -= A0[4]@u[1*cgdof:2*cgdof] 
            if GD == 3:
                s -= A0[5]@u[2*cgdof:3*cgdof]
            s /= A0[0].diagonal()

            return s, v, p, u
        else:
            return None, None, None, None


    def picard_iteration(self):
        """
        """

        e0 = 1.0
        k = 0
        while e0 > 1e-10: 

            s, v, p, u = self.solve_linear_system_1()

            if self.ctx.myid == 0:
                e0 = 0.0
                e0 += np.sum((self.cs - s)**2)
                self.cs[:] = s 

                e0 += np.sum((self.cp - p)**2)
                self.cp[:] = p 

                self.cv[:] = v 
                self.cu.T.flat[:] = u

                e0 = np.sqrt(e0) # 误差的 l2 norm
                print(e0)

            e0 = self.ctx.comm.bcast(e0, root=0)
            k += 1
            if k >= self.args.npicard: 
                if self.ctx.myid == 0:
                    print('picard iteration arrive max iteration with error:', e0)
                break

        if self.ctx.myid == 0:
            # 更新解
            self.s[:] = self.cs
            self.v[:] = self.cv
            self.p[:] = self.cp
            self.u[:] = self.cu

    def update_mesh_data(self):
        """

        Notes
        -----
        更新 mesh 中的数据
        """
        GD = self.GD
        bc = np.array((GD+1)*[1/(GD+1)], dtype=np.float64)

        mesh = self.mesh

        v = self.v 
        p = self.p
        s = self.s
        u = self.u


        # 单元中心的流体速度
        val = v.value(bc)
        if GD == 2:
            val = np.concatenate((val, np.zeros((val.shape[0], 1), dtype=val.dtype)), axis=1)
        mesh.celldata['velocity'] = val 

        # 分片常数的压强
        val = self.recover(p[:])
        mesh.nodedata['pressure'] = val

        # 分片常数的饱和度
        val = self.recover(s[:])
        mesh.nodedata['fluid_0'] = val

        val = self.recover(1 - s)
        mesh.nodedata['fluid_1'] = val 

        # 节点处的位移
        if GD == 2:
            u = np.concatenate((u[:], np.zeros((u.shape[0], 1), dtype=u.dtype)), axis=1)
        mesh.nodedata['displacement'] = u[:] 

        # 增加应变和应力的计算

        NC = self.mesh.number_of_cells()
        s0 = np.zeros((NC, 3, 3), dtype=np.float64)
        s1 = np.zeros((NC, 3, 3), dtype=np.float64)

        s = u.grad_value(bc) # (NC, GD, GD)
        s0[:, 0:GD, 0:GD] = s + s.swapaxes(-1, -2)
        s0[:, 0:GD, 0:GD] /= 2

        lam = self.mesh.celldata['lambda']
        mu = self.mesh.celldata['mu']

        s1[:, 0:GD, 0:GD] = s0[:, 0:GD, 0:GD]
        s1[:, 0:GD, 0:GD] *= mu[:, None, None]
        s1[:, 0:GD, 0:GD] *= 2 

        s1[:, range(GD), range(GD)] += (lam*s0.trace(axis1=-1, axis2=-2))[:, None]

        s1[:, range(GD), range(GD)] += mesh.celldata['stress_0'][:, None]

        s1[:, range(GD), range(GD)] -= (mesh.celldata['biot']*(p - mesh.celldata['pressure_0']))[:, None]

        mesh.celldata['strain'] = s0.reshape(NC, -1)
        mesh.celldata['stress'] = s1.reshape(NC, -1)



    def run(self, writer=None):
        """

        Notes
        -----

        计算所有时间层物理量。
        """

        args = self.args

        timeline = self.timeline
        dt = timeline.current_time_step_length()

        if (self.ctx.myid == 0) and (writer is not None):
            n = timeline.current
            fname = args.output + str(n).zfill(10) + '.vtu'
            self.update_mesh_data()
            writer(fname, self.mesh)

        while not timeline.stop():
            if self.ctx.myid == 0:
                ct = timeline.current_time_level()/3600/24 # 天为单位
                print('当前时刻为第', ct, '天')

            self.picard_iteration()
            timeline.current += 1
            if timeline.current%args.step == 0:
                if (self.ctx.myid == 0) and (writer is not None):
                    n = timeline.current
                    fname = args.output + str(n).zfill(10) + '.vtu'
                    self.update_mesh_data()
                    writer(fname, self.mesh)

        if (self.ctx.myid == 0) and (writer is not None):
            n = timeline.current
            fname = args.output + str(n).zfill(10) + '.vtu'
            self.update_mesh_data()
            writer(fname, self.mesh)
