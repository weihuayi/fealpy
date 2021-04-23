import numpy as np
import matplotlib.pyplot as plt
import pyamg
import sys

from fealpy.functionspace import WedgeLagrangeFiniteElementSpace
from fealpy.mesh import LagrangeWedgeMesh
from fealpy.geometry.implicit_surface import SphereSurface
from fealpy.decorator import cartesian, barycentric
from scipy.sparse.linalg import spsolve
from fealpy.tools.show import showmultirate, show_error_table

from nonlinear_robin import nonlinear_robin

class PlanetHeatConductionSimulator():
    def __init__(self, pde, mesh, p=1):
        self.space = WedgeLagrangeFiniteElementSpace(mesh, p=p)
        self.mesh = self.space.mesh
        self.pde = pde
        self.nr = nonlinear_robin(space, mesh, p=p)
        
        self.M = self.space.mass_matrix()
        self.A = self.space.stiff_matrix()

    def time_mesh(self, NT=100):
        from fealpy.timeintegratoralg.timeline import UniformTimeLine
        timeline = UniformTimeLine(0, 1, NT)
        return timeline

    def init_solution(self, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        uh = np.zeros(gdof, NL), dtype=np.float)
        uh[:, 0] = 150
        return uh

    def apply_boundary_condition(self, A, b, uh, timeline):
        from fealpy.boundarycondition import RobinBC
        from fealpy.boundarycondition import DirichletBC
        t0 = timeline.current_time_level()
        t1 = timeline.next_time_level()
        i = timeline.current
        
        bc = RobinBC(self.space, lambda x, n:self.pde.robin(x, n, t), threshold=self.pde.is_robin_boundary)
        A0, b = bc.apply(A, b)
        bc = DirichletBC(self.space, lambda x:0, threshold=self.pde.is_dirichlet_boundary)
        A, b = bc.apply(A, b)
        return A, b

    def solve(self, uh, timeline):
        '''  piccard 迭代  C-N 方法 '''
        from nonlinear_robin import nonlinear_robin

        i = timeline.current
        t1 = timeline.next_time_level()
        dt = timeline.current_time_step_length()
        F = self.right_vector(uh, timeline)

        rho = self.mesh.meshdata['rho']
        c = self.mesh.meshdata['c']
        kappa = self.mesh.meshdata['kappa']
        epsilon = self.mesh.meshdata['epsilon']
        sigma = self.mesh.meshdata['sigma']

        A = kappa*self.A
        M = rho*c*self.M

        A, b = self.apply_boundary_condition(A, F, timeline)
        e = 0.000000001
        error = 1
        xi_new = self.space.function()
        xi_new[:] = copy.deepcopy(uh[:, i])
        while error > e:
            xi_tmp = copy.deepcopy(xi_new[:])
            R = nonlinear_robin.robin_bc(A, xi_new, t1, threshold=is_robin_boundary)
            r = M@uh[:, i] - 0.5*dt*R@uh[:,i] + dt*b
            R = M + 0.5*dt*R
            ml = pyamg.ruge_stuben_solver(R)
            xi_new[:] = ml.solve(r, tol=1e-12, accel='cg').reshape(-1)
#            xi_new[:] = spsolve(R, b).reshape(-1)
            error = np.max(np.abs(xi_tmp-xi_new[:]))
        uh[:, i+1] = xi_new

class TPMModel():
    def __init__(self):
        pass

    def init_mesh(self, n=0, h=0.1, nh=5, p=1):
        surface = SphereSurface()
        mesh = surface.init_mesh(meshtype='tri', p=p)
        mesh.uniform_refine(n)
        mesh = LagrangeWedgeMesh(mesh, h, nh, p=p)
        self.mesh = mesh
        return mesh

    def init_mu():
        pass
    
    def right_vector(self):
        return 0
    
    @cartesian
    def dirichlet(self):
        return 0

    @cartesian
    def is_dirichlet_boundary(self, p):
        tface, qface = self.mesh.entity('face')
        nf = len(tface)
        boundary_dir_tface_index = np.zeros(nf, dtype=np.bool_)
        boundary_dir_tface_index[-nf//2:] = True
        return boundary_dir_tface_index 

    @cartesian    
    def robin(self, p, n, t):
        """ Robin boundary condition
        """
        A = self.mesh.meshdata['A']
        q = self.mesh.meshdata['q']
        mu = self.mesh.meshdata['mu0']
        kappa = self.mesh.meshdata['kappa']
        return -(1-A)*q*mu, kappa
    
    @cartesian
    def is_robin_boundary(self):
        tface, qface = self.mesh.entity('face')
        nf = len(tface)
        boundary_robin_tface_index = np.zeros(nf, dtype=np.bool_)
        boundary_robin_tface_index[:nf//2] = True
        return boundary_robin_tface_index 


if __name__ == '__main__':
    p = int(sys.argv[1])
    n = int(sys.argv[2])
    h = int(sys.argv[3])
    nh = int(sys.argv[4])
    NT = int(sys.argv[5])
    maxit = int(sys.argv[6])

    pde = TPMModel()
    mesh = pde.init_mesh(n=n, h=h, nh=nh, p=p)

    mesh.meshdata['A'] = 0.1 # 邦德反照率
    mesh.meshdata['epsilon'] = 0.9 # 辐射率
    mesh.meshdata['rho'] = 1400 # kg/m^3 密度
    mesh.meshdata['c'] = 1200 # Jkg^-1K^-1 比热容
    mesh.meshdata['kappa'] = 0.02 # Wm^-1K^-1 热导率
    mesh.meshdata['sigma'] = 5.6367e-8 # 玻尔兹曼常数
    mesh.meshdata['q'] = 1367.5 # W/m^2 太阳辐射通量
    mesh.meshdata['mu0'] = pde.init_mu()  # max(cos beta,0) 太阳高度角参数

    simulator  = PlanetHeatConductionSimulator(pde, mesh)

    timeline = self.time_mesh(NT=NT)

#    errorType = ['$||u-u_h||_0$', '$||\\nable u - \\nable u_h ||_0$']
#    errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
    Ndof = np.zeros(maxit, dtype=np.float)
    for i in range(maxit):
        print(i)
        model = PlanetHeatConductionSimulator(pde, mesh, p=p)
        Ndof[i] = model.space.number_of_global_dofs()

        uh = model.init_solution(timeline)

        timeline.time_integration(uh, model, spsolve)

        if i < maxit-1:
            timeline.uniform_refine()
#                mesh.uniform_refine()
            n = n+1
            h = h/2
            nh = nh*2
            mesh = pde.init_mesh(n=n, h=h, nh=nh)
        uh = model.space.function(array=uh[:, -1])
#        errorMatrix[0, i] = model.space.integralalg.L2_error(u, uh)
#        errorMatrix[1, i] = model.space.integralalg.L2_error(u.grad_value,
#                uh.grad_value)
#        print(errorMatrix)

    mesh.to_vtk(fname='test.vtu') 
    fig = plt.figure()
    axes = fig.gca(projection='3d')
    uh.add_plot(axes, cmap='rainbow')

    show_error_table(Ndof, errorType, errorMatrix)
    showmultirate(plt, 0, Ndof, errorMatrix, errorType)
    plt.show()

