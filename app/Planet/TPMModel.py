import numpy as np
import meshio

from fealpy.decorator import cartesian, barycentric
from fealpy.mesh import LagrangeTriangleMesh, LagrangeWedgeMesh
from fealpy.writer import MeshWriter

class TPMModel():
    def __init__(self, options=None):
        if options is None:
            self.options = self.model_options()

    def model_options(self, 
            A=0.1,             # 邦德反照率
            epsilon=0.9,       # 辐射率
            rho=1400,          # kg/m^3 密度
            c=1200,            # Jkg^-1K^-1 比热容
            kappa=0.02,        # Wm^-1K^-1 热导率
            sigma=5.667e-8,    # Wm^-2K^-4 斯特藩-玻尔兹曼常数
            qs=1367.5,          # W/m^2 太阳辐射通量
            period=5,          # 小行星自转周期，单位小时
            sd=[1, 0, 1],      # 指向太阳的方向
            theta=150,         # 初始温度，单位 K
            ):

        """
        Notes
        -----
            设置模型参数信息
        """
        period *= 3600 # 转换为秒
        omega = 2*np.pi/period
        Tss = ((1-A)*qs/(epsilon*sigma))**(1/4) # 日下点温度 T_ss=[(1-A)*r/(epsilon*sigma)]^(1/4)
        Tau = np.sqrt(rho*c*kappa) # 热惯量 Tau = (rho*c*kappa)^(1/2)
        Phi = Tau*np.sqrt(omega)/(epsilon*sigma*Tss**3) # 热参数
        l = np.sqrt(kappa/(rho*c*omega)) # 趋肤深度 l=(kappa/(rho*c*omega))^(1/2)
        options = {
                "A": A,
                "epsilon": epsilon,
                "rho": rho,
                "c": c,
                "kappa": kappa,
                "sigma": sigma,
                "qs": qs,
                "period": period,
                "sd": np.array(sd, dtype=np.float64),
                "omega": omega,
                "Tss": Tss,
                "Tau": Tau,
                "Phi": Phi,
                "theta": theta,
                "l": l,
                }
        return options


    def init_mesh(self, n=0, h=0.005, nh=100, H=500, p=1):
        fname = 'initial/file1.vtu'
        data = meshio.read(fname)
        node = data.points # 无量纲数值
        cell = data.cells[0][1]

        node = node - np.mean(node, axis=0) # 以质心作为原点
        l = self.options['l']
        node*= H # H 小行星的规模
        node/=l # 无量纲化处理
        h/= l # 无量纲化处理

        mesh = LagrangeTriangleMesh(node, cell, p=p)
        mesh.uniform_refine(n)
        mesh = LagrangeWedgeMesh(mesh, h, nh, p=p)

        self.mesh = mesh
        self.p = p
        return mesh

    def init_mu(self, t, n0):
        boundary_face_index = self.is_robin_boundary()
        qf0, qf1 = self.mesh.integrator(self.p, 'face')
        bcs, ws = qf0.get_quadrature_points_and_weights()
        m = self.mesh.boundary_tri_face_unit_normal(bcs, index=boundary_face_index)

        # 指向太阳的向量绕 z 轴旋转, 这里 t 为 omega*t
        Z = np.array([[np.cos(-t), -np.sin(-t), 0],
            [np.sin(-t), np.cos(-t), 0],
            [0, 0, 1]], dtype=np.float64)
        n = Z@n0 # t 时刻指向太阳的方向
        n = n/np.sqrt(np.dot(n, n)) # 单位化处理

        mu = np.dot(m, n)
        mu[mu<0] = 0
        return mu
    
    def right_vector(self, uh):
        shape = uh.shape[0]
        f = np.zeros(shape, dtype=np.float)
        return f 
    
    @cartesian
    def neumann(self, p, n):
        gN = np.zeros((p.shape[0], p.shape[1]), dtype=np.float)
        return gN

    @cartesian
    def neumann_boundary_index(self):
        tface, qface = self.mesh.entity('face')
        NTF = len(tface)
        index = np.zeros(NTF, dtype=np.bool_)
        index[-NTF//2:] = True
        return boundary_neumann_tface_index 

    @cartesian    
    def robin(self, p, n, t):
        """ Robin boundary condition
        """
        Phi = self.options['Phi']
        sd = self.options['sd']
        mu = self.init_mu(t, sd)
       
        shape = len(mu.shape)*(1, )
        k = -np.array([1.0], dtype=np.float64).reshape(shape)/Phi
        return -mu/Phi, k
    
    @cartesian
    def robin_boundary_index(self):
        tface, qface = self.mesh.entity('face')
        NTF = len(tface)
        index = np.zeros(NTF, dtype=np.bool_)
        index[:NTF//2] = True
        return index 

