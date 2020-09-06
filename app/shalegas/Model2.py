
"""

Notes
-----
这是一个两相流和地质力学耦合的模型, 需要求的量有

* p: 压强
* v: 总速度
* S_w: 水的饱和度 S_w
* u: 岩石位移  

目前, 模型
* 忽略了毛细管压强和重力作用
* 没有考虑裂缝

渐近解决方案:
1. Picard 迭代
2. 气的可压性随着压强的变化而变化
3. 考虑渗透率随着孔隙度的变化而变化 
4. 考虑裂缝


Data
----

Coupling of Stress Dependent Relative Permeability and Reservoir Simulation

Water-oil relative permeability SWT
 SW        krw      krow
0.200000 0.000000 0.510200
0.235714 0.000010 0.439917
0.271429 0.000163 0.374841
0.307143 0.000824 0.314970
0.342857 0.002603 0.260306
0.378571 0.006355 0.210848
0.414286 0.013177 0.166596
0.450000 0.024412 0.127550
0.485714 0.041647 0.093710
0.521429 0.066710 0.065077
0.557143 0.101676 0.041649
0.592857 0.148864 0.023428
0.628571 0.210836 0.010412
0.664286 0.290398 0.002603
0.700000 0.390600 0.000000

Liquid-Gas Relative Permeability SLT

 SL        krg      krog
0.500000 0.477200 0.000000
0.535714 0.411463 0.002603
0.571429 0.350596 0.010412
0.607143 0.294598 0.023428
0.642857 0.243469 0.041649
0.678571 0.197210 0.065077
0.714286 0.155820 0.093710
0.750000 0.119300 0.127550
0.785714 0.087649 0.166596
0.821429 0.060867 0.210848
0.857143 0.038955 0.260306
0.892857 0.021912 0.314970
0.928571 0.009739 0.374841
0.964286 0.002435 0.439917
1.000000 0.000000 0.510200


https://www.coursehero.com/file/14152920/Rel-Perm/ 

 np.array([[0.2     , 0.      , 0.8     ],
           [0.225   , 0.001172, 0.703125],
           [0.25    , 0.004688, 0.6125  ],
           [0.275   , 0.010547, 0.578254],
           [0.3     , 0.01875 , 0.45    ],
           [0.325   , 0.029297, 0.415314],
           [0.35    , 0.064931, 0.3125  ],
           [0.375   , 0.071057, 0.297703],
           [0.4     , 0.077182, 0.2634  ],
           [0.425   , 0.094922, 0.15559 ],
           [0.45    , 0.132312, 0.1125  ],
           [0.475   , 0.138438, 0.091884],
           [0.5     , 0.187443, 0.083308],
          [0.525   , 0.194793, 0.030628],
          [0.55    , 0.256049, 0.028178],
          [0.575   , 0.263672, 0.003125],
          [0.6     , 0.3     , 0.      ]])

体积模量:  K = E/3/(1 - 2*nu)
"""

import numpy as np

import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, coo_matrix, bmat, diags

from fealpy.decorator import barycentric
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d



class WaterFloodingModel():

    def __init__(self):
        self.domain=[0, 10, 0, 10] # m
        self.rock = {
            'permeability': 2, # 1 d = 9.869 233e-13 m^2 
            'porosity': 0.3, # None
            'lame':(1.0e+8, 3.0e+8), # lambda and mu 拉梅常数
            'biot': 1.0,
            'initial pressure': 3, # MPa
            'initial stress': 60.66, # MPa 先不考虑应力的计算 
            'solid grain stiffness': 6.25 # MPa
            }
        self.water = {
            'viscosity': 1, # 1 cp = 1 mPa*s
            'compressibility': 1.0e-3, # MPa^{-1}
            'initial saturation': 0.0, 
            'injection rate': 3.51e-6 # s^{-1}, 每秒多少体积
            }
        self.oil = {'viscosity': 2, # cp
            'compressibility': 2.0e-3, # MPa^{-1}
            'initial saturation': 1.0, 
            'production rate': 3.50e-6 # s^{-1}
            }
        self.bc = {'displacement': 0.0, 'flux': 0.0}

        # Water-oil relative permeability SWT
        #      SW        krw       krow == kro? TODO: make it clear
        self.SWT = np.array([
            (0.200000, 0.000000, 0.510200),
            (0.235714, 0.000010, 0.439917),
            (0.271429, 0.000163, 0.374841),
            (0.307143, 0.000824, 0.314970),
            (0.342857, 0.002603, 0.260306),
            (0.378571, 0.006355, 0.210848),
            (0.414286, 0.013177, 0.166596),
            (0.450000, 0.024412, 0.127550),
            (0.485714, 0.041647, 0.093710),
            (0.521429, 0.066710, 0.065077),
            (0.557143, 0.101676, 0.041649),
            (0.592857, 0.148864, 0.023428),
            (0.628571, 0.210836, 0.010412),
            (0.664286, 0.290398, 0.002603),
            (0.700000, 0.390600, 0.000000)], dtype=np.float64)

        # 二次拟合
        self.krw = np.poly1d(np.polyfit(self.SWT[:, 0], self.SWT[:, 1], 2))
        self.kro = np.poly1d(np.polyfit(self.SWT[:, 0], self.SWT[:, 2], 2))


    def space_mesh(self, n=32):
        from fealpy.mesh import MeshFactory
        mf = MeshFactory()
        mesh = mf.boxmesh2d(self.domain, nx=n, ny=n, meshtype='tri')
        return mesh

    def time_mesh(self, T=1, n=100):
        from fealpy.timeintegratoralg.timeline import UniformTimeLine
        timeline = UniformTimeLine(0, T, n)
        return timeline

    def relative_permeability_water(self, Sw):
        """

        Notes
        ----
        给定水的饱和度, 计算水的相对渗透率
        """
        val = self.krw(Sw)
        val[val<0.0] = 0.0
        val[val>1.0] = 1.0
        return val

    def relative_permeability_oil(self, Sw):
        """

        Notes
        ----
        给定水的饱和度, 计算油的相对渗透率
        """
        val = self.kro(Sw)
        val[val<0.0] = 0.0
        val[val>1.0] = 1.0
        return val




class WaterFloodingModelSolver():

    def __init__(self, model):
        self.model = model
        self.mesh = model.space_mesh()
        self.timeline = model.time_mesh()

        self.vspace = RaviartThomasFiniteElementSpace2d(self.mesh, p=0) # 速度空间
        self.pspace = self.vspace.smspace # 压强空间
        self.cspace = LagrangeFiniteElementSpace(self.mesh, p=1) # 位移和饱和度连续空间


        self.v = self.vspace.function() # 速度函数
        self.p = self.pspace.function() # 压强函数
        self.u = self.cspace.function(dim=2) # 位移函数
        self.s = self.cspace.function() # 水的饱和度函数 默认为0, 初始时刻区域内水的饱和度为0
        self.phi = self.pspace.function() # 孔隙度函数, 分片常数

        NN = self.mesh.number_of_nodes()
        self.fg = self.cspace.function()
        self.fw = self.cspace.function()

        # TODO: 注意这里假设用的是结构网格, 换其它的网格需要修改代码
        self.fg[0] = self.model.water['injection rate'] # 注入
        self.fw[-1] = -self.model.oil['production rate'] # 产出

        # 初值
        self.p[:] = model.rock['initial pressure']  # MPa
        self.phi[:] = model.rock['porosity'] 

        # 一些常数矩阵和向量

        # (\nu \nabla S_w, \nabla v)
        self.A = 0.001*self.cspace.stiff_matrix() # 稳定项

        # (\nabla\cdot v, w) 速度散度矩阵
        self.B = self.vspace.div_matrix()


        # (\nabla\cdot u, w) 位移散度矩阵
        cellmeasure = self.mesh.entity_measure('cell')
        cellmeasure *= self.model.rock['biot']
        gphi = self.mesh.grad_lambda() # (NC, TD+1, GD)
        gphi *= cellmeasure[:, None, None]
        c2d0 = self.pspace.cell_to_dof()
        c2d1 = self.cspace.cell_to_dof()
        gdof0 = self.pspace.number_of_global_dofs()
        gdof1 = self.cspace.number_of_global_dofs()
        I = np.broadcast_to(c2d0, shape=c2d1.shape)
        J = c2d1 
        self.U00 = csr_matrix(
                (gphi[..., 0].flat, (I.flat, J.flat)), 
                shape=(gdof0, gdof1)
                )
        self.U01 = csr_matrix(
                (gphi[..., 1].flat, (I.flat, J.flat)),
                shape=(gdof0, gdof1)
                )
        

    @barycentric
    def pressure_coefficient(self, bc):

        """

        Notes
        -----
        计算压强矩阵的系数
        """

        b = self.model.rock['biot'] 
        Ks = self.model.rock['solid grain stiffness']
        cw = self.model.water['compressibility']
        co = self.model.oil['compressibility']

        Sw = self.s.value(bc)

        ps = mesh.bc_to_point(bc)
        phi = self.phi.value(ps)

        val = b - phi
        val /= Ks
        val += phi*Sw*cw
        val += phi(1 - Sw)*co # 注意这里的 co 是常数, 在气体情况下 应该依赖于压力
        return val

    @barycentric
    def saturation_coefficient(self, bc):
        """

        Notes
        -----
        计算饱和度矩阵的系数
        """
        b = self.model.rock['biot'] 
        Ks = self.model.rock['solid grain stiffness']
        cw = self.model.water['compressibility']

        Sw = self.s.value(bc)
        ps = mesh.bc_to_point(bc)
        phi = self.phi.value(ps)

        val = b - phi
        val /= Ks
        val += phi*cw
        val *= Sw
        return val

    def flux_coefficient(self, bc):
        """
        Notes
        -----

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

        # 岩石的绝对渗透率, 这里考虑量纲的一致性
        k = self.model.rock['permeability']*9.869233e-4  

        Sw = self.s.value(bc) # 水的饱和度系数

        lamw = self.model.relative_permeability_water(Sw)
        lamw /= muw
        lamo = self.model.relative_permeability_oil(Sw)
        lamo /= muo

        val = 1/(lamw + lam)/k # 

        return val

    def stabilization_coefficient(self, bc):
        """
        Notes
        -----
        稳定项系数, 这里暂时假设为常数
        """
        alpha = 0.1
        beta = 0.2
        CR = 1.0
        return 0.01

    def water_fractional_flow(self, bc):
        """

        Notes
        -----
        """

        Sw = self.s.value(bc) # 水的饱和度系数
        lamw = self.model.relative_permeability_water(Sw)
        lamw /= muw
        lamo = self.model.relative_permeability_oil(Sw)
        lamo /= muo
        val = lamw/(lamw + lamo)
        return val

    def get_total_system(self):
        """
        Notes
        -----
        构造整个系统

        x = [v, p, s, u0, u1]

        A = [[   V,  -B, None, None,  None]
             [ B.T,   P, None, DU00,  DU01]
             [  FV,  SP,    S, DU10,  DU10] 
             [None, UP0, None,  U00,   U01]
             [None, UP1, None,  U10,   U11]

        F = [None, FP, FS, FU0, FU1]
        """
        pass

    def get_velocity_system(self, q=2):
        """
        Notes
        -----
        计算速度方程对应的离散系统.
        """
        cellmeasure = self.vspace.cellmeasure
        qf = self.mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        c = self.flux_coefficient(bcs)
        phi = self.vspace.basis(bcs)
        V = np.einsum('q, qc, qcin, qcjn, c->cij', ws, c, phi, phi,
                cellmeasure, optimize=True)

        c2d = self.vspace.cell_to_dof()
        I = np.broadcast_to(c2d[:, :, None], shape=V.shape)
        J = np.broadcast_to(c2d[:, None, :], shape=V.shape)

        gdof = self.vspace.number_of_global_dofs()
        V = csr_matrix(
                (V.flat, (I.flat, J.flat)), 
                shape=(gdof, gdof)

        return [V, -B, None, None, None], np.zeros(gdof, dtype=np.float64) 

    def get_pressure_system(self, q=2):
        """
        Notes
        -----
        计算压强方程对应的离散系统

        这里组装矩阵时, 利用了压强是分片常数的特殊性 
        """
        dt = self.timeline.current_time_step_length()
        cellmeasure = self.pspace.cellmeasure
        qf = self.mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        c = self.pressure_coefficient(bcs) # (NQ, NC)
        c = np.sum(c, axis=0)
        c *= cellmeasure 
        P = diags(val, 0)

        FP = self.fg.value(bcs) + self.fw.value(bcs)
        FP = np.sum(val, axis=0)
        FP *= cellmeasure
        FP *= dt

        FP += P@self.p
        FP += self.U00@self.u[:, 0]
        FP += self.U01@self.u[:, 1]

        return [dt*self.B.T, P, None, self.U00, self.U01], FP

    def get_saturation_system(self, q=2):
        """
        Notes
        ----
        计算饱和度方程对应的离散系统

        [  FV,  SP,    S, DU10,  DU10] 

        """

        dt = self.timeline.current_time_step_length()
        cellmeasure = self.cspace.cellmeasure
        qf = self.mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        gphi = self.mesh.grad_lambda() #(NC, TD+1, GD)
        vphi = self.vspace.basis(bcs) # (NQ, NC, TD+1, GD)
        c = self.water_fractional_flow(bcs) # (NQ, NC)

        FV = np.einsum('q, qc, cin, qcjn, c->cij', ws, c, gphi, vphi, cellmeasure)

        gdof0 = self.cspace.number_of_global_dofs()
        gdof1 = self.vspace.number_of_global_dofs()
        c2d0 = self.cspace.cell_to_dof()
        c2d1 = self.vspace.cell_to_dof()
        I = np.broadcast_to(c2d0[:, :, None], shape=FV.shape)
        J = np.broadcast_to(c2d1[:, None, :], shape=FV.shape)

        FV = csr_matrix(
                (FV.flat, (I.flat, J.flat)),
                shape=(gdof0, gdof1)
                )


        I = np.broadcast_to(c2d0[:, :, None], shape=M.shape)
        J = np.broadcast_to(c2d0[:, None, :], shape=M.shape)

        phi = self.cspace.basis(bcs) # (NQ, 1, ldof)
        c = self.phi[:] # (NC, )
        S = np.einsum('q, c, qci, qcj, c->cij', ws, c, phi, phi, cellmeasure)
        S = csr_matrix(
                (S.flat, (I.flat, I.flat)),
                shape=(gdof0, gdof1)
                )

        gphi = self.mesh.grad_lambda() # (NC, TD+1, GD)
        c = self.s.value(bcs)*self.model.rock['biot'] # (NQ, NC)
        U10 = np.einsum('q, qc, qci, qcj, c->cij', ws, c, phi, gphi[..., 0], cellmeasure)
        U11 = np.einsum('q, qc, qci, qcj, c->cij', ws, c, phi, gphi[..., 1], cellmeasure)

        U10 = csr_matrix(
                (U10.flat, (I.flat, J.flat)),
                shape=U10.shape
                )
        U11 = csr_matrix(
                (U11.flat, (I.flat, J.flat)),
                shape=U10.shape
                )


        c = self.saturation_coefficient(bcs) # (NQ, NC)
        c = np.sum(c, axis=0)
        c *= cellmeasure 
        SP = diags(val, 0)

        # 右端矩阵
        F = dt*self.cspace.source_vector(self.fw)
        F += dt*SP@self.p 
        F += S@self.s
        F += U10@self.u[:, 0]
        F += U11@self.u[:, 1]

        S += dt*self.A

    def get_dispacement_system(self):
        """
        Notes
        -----
        计算位移方程对应的离散系统, 即线弹性方程系统.
        """

        lam, mu = self.model.rock['lame']
        A = self.cspace.linear_elasticity_matrix(lam, mu)







if __name__ == '__main__':

    model = WaterFloodingModel()
    solver = WaterFloodingModelSolver(model)

    fig = plt.figure()
    axes = fig.gca()
    solver.mesh.add_plot(axes)
    solver.mesh.find_node(axes, showindex=True)
    plt.show()
