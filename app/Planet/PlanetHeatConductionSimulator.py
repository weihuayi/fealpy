
import sys
import numpy as np
import matplotlib.pyplot as plt
import meshio

from fealpy.decorator import cartesian, barycentric
from fealpy.mesh import LagrangeTriangleMesh, LagrangeWedgeMesh
from fealpy.writer import MeshWriter
from fealpy.functionspace import WedgeLagrangeFiniteElementSpace
from fealpy.boundarycondition import RobinBC, NeumannBC

from scipy.sparse.linalg import spsolve
import pyamg

from fealpy.tools.show import showmultirate, show_error_table

from nonlinear_robin import nonlinear_robin

class PlanetHeatConductionSimulator():
    def __init__(self, pde, mesh, p=1, option=None):

        self.pde = pde
        self.mesh = mesh
        self.space = WedgeLagrangeFiniteElementSpace(mesh, p=p, q=6)

        self.nr = nonlinear_robin(self.pde, self.space, self.mesh, p=p, q=6)
        
        self.S = self.space.stiff_matrix() # 刚度矩阵
        self.M = self.space.mass_matrix() # 质量矩阵

        if option is None:
            self.options = self.model_options()

        self.uh0 = self.init_solution()  # 当前时间层的温度分布
        self.uh1 = self.space.function() # 下一时间层的温度分布 
        
    def model_options(self, 
            A=0.1,             # 邦德反照率
            epsilon=0.9,       # 辐射率
            rho=1400,          # kg/m^3 密度
            c=1200,            # Jkg^-1K^-1 比热容
            kappa=0.02,        # Wm^-1K^-1 热导率
            sigma=5.667e-8,    # Wm^-2K^-4 斯特藩-玻尔兹曼常数
            r=1367.5,          # W/m^2 太阳辐射通量
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
        Tss = ((1-A)*r/(epsilon*sigma))**(1/4) # 日下点温度 T_ss=[(1-A)*r/(epsilon*sigma)]^(1/4)
        Tau = np.sqrt(rho*c*kappa) # 热惯量 Tau = (rho*c*kappa)^(1/2)
        Phi = Tau*np.sqrt(omega)/(epsilon*sigma*Tss**3) # 热参数
        options = {
                "A": A,
                "epsilon": epsilon,
                "rho": rho,
                "c": c,
                "kappa": kappa,
                "sigma": sigma,
                "r": r,
                "period": period， 
                "sd": np.array(sd, dtype=np.float64),
                "omega": omega,
                "Tss": Tss,
                "Tau": Tau, 
                "Phi": Phi，
                "theta": theta,
                }
        return options

    def time_mesh(self, T=10, NT=100):
        
        """ 
        Parameters
        ----------
            T: the final time (day)
        Notes
        ------
            无量纲化后 tau=omega*t 这里的时间层为 tau
        """
        from fealpy.timeintegratoralg.timeline import UniformTimeLine

        omega = self.options['omega']
        T *= 3600*24 # 换算成秒
        T *= omega # 归一化
        timeline = UniformTimeLine(0, T, NT)
        return timeline

    def init_solution(self):
        theta = self.options['theta']
        uh = self.space.function()
        uh[:] = self.options['theta']/self.options['Tss'] 
        return uh

    def apply_boundary_condition(self, A, b, uh, timeline):
        t1 = timeline.next_time_level()
       
       # 对 neumann 边界条件进行处理
        bc = NeumannBC(self.space, lambda x, n:self.pde.neumann(x, n), threshold=self.pde.is_neumann_boundary())
        b = bc.apply(b) # 混合边界条件不需要输入矩阵A

        # 对 robin 边界条件进行处理
        bc = RobinBC(self.space, lambda x, n:self.pde.robin(x, n, t1, self.Phi), threshold=self.pde.is_robin_boundary())
        A0, b = bc.apply(A, b)
        return b

    def solve(self, uh, timeline):
        '''  piccard 迭代  向后欧拉方法 '''
        from nonlinear_robin import nonlinear_robin

        i = timeline.current
        t1 = timeline.next_time_level()
        dt = timeline.current_time_step_length()
        F = self.pde.right_vector(uh, timeline)

        A = self.A
        M = self.M
        b = self.apply_boundary_condition(A, F, uh, timeline)
        
        e = 1e-10 
        error = 1
        xi_new = self.space.function()
        xi_new[:] = uh[:, i]
        while error > e:
            xi_tmp = xi_new.copy()
            R = self.nr.robin_bc(A, xi_new, lambda x, n:self.pde.robin(x,
                n, t1, self.Phi), threshold=self.pde.is_robin_boundary())
            r = M@uh[:, i] + dt*b
            R = M + dt*R

#            ml = pyamg.ruge_stuben_solver(R)
#            xi_new[:] = ml.solve(r, tol=1e-12, accel='cg').reshape(-1)
            xi_new[:] = spsolve(R, r).reshape(-1)

            error = np.max(np.abs(xi_tmp - xi_new[:]))
            print('error:', error)
        print('i:', i+1)
        uh[:, i+1] = xi_new

class TPMModel():
    def __init__(self):
        pass

    def init_mesh(self, n=0, h=0.005, nh=100, p=1):
        fname = 'file1.vtu'
        data = meshio.read(fname)
        node = data.points
        cell = data.cells[0][1]

        node = node - np.mean(node, axis=0)
        # 无量纲化处理
        l = np.sqrt(0.02*18000/(1400*1200*2*np.pi)) # 趋肤深度 l=(kappa/(rho*c*omega))^(1/2)
        node = 500*node/l
        h = h/l

        mesh = LagrangeTriangleMesh(node, cell, p=p)
        mesh.uniform_refine(n)
        mesh = LagrangeWedgeMesh(mesh, h, nh, p=p)

        self.mesh = mesh
        self.p = p
        return mesh

    def init_mu(self, t):
        boundary_face_index = self.is_robin_boundary()
        qf0, qf1 = self.mesh.integrator(self.p, 'face')
        bcs, ws = qf0.get_quadrature_points_and_weights()
        m = mesh.boundary_tri_face_unit_normal(bcs, index=boundary_face_index)

        # 小行星外法向, 初始为 (1, 0, 1) 绕 z 轴旋转, 这里 t 为 omega*t
        n = np.array([np.cos(t), -np.sin(t), 1]) # 指向太阳的方向
        n = n/np.sqrt(np.dot(n, n))

        mu = np.dot(m, n)
        mu[mu<0] = 0
        return mu
    
    def right_vector(self, uh, timeline):
        shape = uh.shape[0]
        f = np.zeros(shape, dtype=np.float)
        return f 
    
    @cartesian
    def neumann(self, p, n):
        gN = np.zeros((p.shape[0], p.shape[1]), dtype=np.float)
        return gN

    @cartesian
    def is_neumann_boundary(self):
        tface, qface = self.mesh.entity('face')
        NTF = len(tface)
        boundary_neumann_tface_index = np.zeros(NTF, dtype=np.bool_)
        boundary_neumann_tface_index[-NTF//2:] = True
        return boundary_neumann_tface_index 

    @cartesian    
    def robin(self, p, n, t, Phi):
        """ Robin boundary condition
        """
        mu = self.init_mu(t)
       
        shape = len(mu.shape)*(1, )
        kappa = -np.array([1.0], dtype=np.float64).reshape(shape)/Phi
        return -mu/Phi, kappa
    
    @cartesian
    def is_robin_boundary(self):
        tface, qface = self.mesh.entity('face')
        NTF = len(tface)
        boundary_robin_tface_index = np.zeros(NTF, dtype=np.bool_)
        boundary_robin_tface_index[:NTF//2] = True
        return boundary_robin_tface_index 

if __name__ == '__main__':
    p = int(sys.argv[1])
    n = int(sys.argv[2])
    NT = int(sys.argv[3])
    maxit = int(sys.argv[4])
#    h = float(sys.argv[5])
#    nh = int(sys.argv[6])
    h = 0.005
    nh = 100

    pde = TPMModel()
    mesh = pde.init_mesh(n=n, h=h, nh=nh, p=p)

    simulator = PlanetHeatConductionSimulator(pde, mesh)

    timeline = simulator.time_mesh(NT=NT)

    Ndof = np.zeros(maxit, dtype=np.float)
    
    for i in range(maxit):
        print(i)
        model = PlanetHeatConductionSimulator(pde, mesh, p=p)
        
        Ndof[i] = model.space.number_of_global_dofs()

        uh = model.init_solution(timeline)

        timeline.time_integration(uh, model, spsolve)

        if i < maxit-1:
            timeline.uniform_refine()
            
            n = n+1
            h = h/2
            nh = nh*2
            mesh = pde.init_mesh(n=n, h=h, nh=nh)
        
        uh = model.space.function(array=uh[:, -1])
        uh = uh*self.simulator.T
        print('uh:', uh)

    np.savetxt('01solution', uh)
    mesh.nodedata['uh'] = uh

    mesh.to_vtk(fname='test.vtu') 
