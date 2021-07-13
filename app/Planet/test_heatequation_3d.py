import numpy as np
import copy
import pyamg
import matplotlib.pyplot as plt

from fealpy.decorator import cartesian, barycentric
from heatequation_2d import CosCosCosExpData
from nonlinear_robin import nonlinear_robin
from fealpy.timeintegratoralg.timeline import UniformTimeLine
from fealpy.boundarycondition import RobinBC
from fealpy.boundarycondition import DirichletBC
from scipy.sparse.linalg import spsolve
from fealpy.tools.show import showmultirate, show_error_table

class ParabolicFEMModel():
    def __init__(self, pde, mesh, p=1, q=6, p0=None):
#        from fealpy.functionspace import LagrangeFiniteElementSpace
        from fealpy.functionspace.WedgeLagrangeFiniteElementSpace import WedgeLagrangeFiniteElementSpace
        from fealpy.boundarycondition import BoundaryCondition
        self.space = WedgeLagrangeFiniteElementSpace(mesh, p=p, q=q)
        self.mesh = self.space.mesh
        self.pde = pde
        self.nr = nonlinear_robin(pde, self.space, self.mesh, p=p, q=q)

        self.M = self.space.mass_matrix()
        self.A = self.space.stiff_matrix()

    def init_solution(self, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        uh = np.zeros((gdof, NL), dtype=np.float)
        uh[:, 0] = self.space.interpolation(lambda x:self.pde.solution(x, 0.0))
        return uh

    def interpolation(self, u, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        ps = self.space.interpolation_points()
        uI = np.zeros((gdof, NL), dtype=np.float)
        times = timeline.all_time_levels()
        for i, t in enumerate(times):
            uI[..., i] = u(ps, t)
        return uI

    def get_current_left_matrix(self, timeline):
        return self.A

    def get_current_right_vector(self, uh, timeline):
        t0 = timeline.current_time_level()
        t1 = timeline.next_time_level()
        f = lambda x: self.pde.source(x, t1) #+ self.pde.source(x, t0))
        F = self.space.source_vector(cartesian(f))
        return F

    def apply_boundary_condition(self, A, b, uh, timeline):
        t0 = timeline.current_time_level()
        t1 = timeline.next_time_level()
        i = timeline.current
        from fealpy.boundarycondition import NeumannBC
        bc = NeumannBC(self.space, lambda x, n:(self.pde.neumann(x, n, t1)), threshold=self.pde.is_dirichlet_boundary)
        b = bc.apply(b)
#        A = A[:-1, :][:, :-1]
#        b = b[:-1]
        
        bc = RobinBC(self.space, lambda x, n:self.pde.boundary_nonlinear_robin(x, n, t1),
                threshold=self.pde.is_robin_boundary())
        A0, b = bc.apply(A, b)
 
#        bc = DirichletBC(self.space, lambda x:(self.pde.dirichlet(x,
#            t0)+self.pde.dirichlet(x, t1)), threshold=self.pde.is_dirichlet_boundary)
        return A, b
    
    def solve(self, uh, timeline):
        '''  piccard 迭代  向后欧拉方法 '''
        t1 = timeline.next_time_level()
        i = timeline.current
        dt = timeline.current_time_step_length()
        A = self.get_current_left_matrix(timeline)
        F = self.get_current_right_vector(uh, timeline)
        A0, b = self.apply_boundary_condition(A, F, uh, timeline)
        e = 0.000000001
        error = 1
        xi_new = self.space.function()
        xi_new[:] = copy.deepcopy(uh[:,i])
        while error > e:
            xi_tmp = copy.deepcopy(xi_new[:])
            R = self.nr.robin_bc(A, xi_new, lambda x,
                    n:self.pde.boundary_nonlinear_robin(x, n, t1),
                    threshold=self.pde.is_robin_boundary())
            r = self.M@uh[:, i] + dt*b
            R = self.M + dt*R
            xi_new[:] = spsolve(R, r).reshape(-1)
#            ml = pyamg.ruge_stuben_solver(R)
#            xi_new[:] = ml.solve(r, tol=1e-12, accel='cg').reshape(-1)    
            error = np.max(np.abs(xi_tmp-xi_new[:]))
#            print(error)
#        print('i:', i+1)
        uh[:, i+1] = xi_new

#    def solve(self, uh, timeline):
#        A = self.get_current_left_matrix(timeline)
#        F = self.get_current_right_vector(uh, timeline)
#        A, b = self.apply_boundary_condition(A, F, uh, timeline)
#        dt = timeline.current_time_step_length()
#        i = timeline.current
#        b = self.M@uh[:, i] - 0.5*dt*A@uh[:, i] + 0.5*dt*b
#        A = self.M + 0.5*dt*A
#        ml = pyamg.ruge_stuben_solver(A)
#        uh[:, i+1] = ml.solve(b, tol=1e-12, accel='cg').reshape(-1)
#        uh[:, i+1] = spsolve(A, b).reshape(-1)

class TimeIntegratorAlgTest():
    def __init__(self):
        pass

    def test_ParabolicFEMModel_time(self, maxit=4):
#        pde = XYTData(0.1)
        pde = CosCosCosExpData()
        p = 1
        n = 0
        h = 0.1
        nh = 5
        mesh = pde.init_mesh(n=n, h=h, nh=nh, p=p)
        timeline = UniformTimeLine(0, 1, 100)

        errorType = ['$|| u - u_h ||_\infty$', '$||u-u_h||_0$']
        errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
        Ndof = np.zeros(maxit, dtype=np.float)
        for i in range(maxit):
            print(i)
            model = ParabolicFEMModel(pde, mesh, p=p)
            Ndof[i] = model.space.number_of_global_dofs()

            uh = model.init_solution(timeline)

            timeline.time_integration(uh, model, spsolve)
            uI = model.interpolation(pde.solution, timeline)
            errorMatrix[0, i] = np.max(np.abs(uI - uh))

            u = lambda x:model.pde.solution(x, 1.0)
            uh = model.space.function(array=uh[:, -1])
            errorMatrix[1, i] = model.space.integralalg.L2_error(u, uh)
            if i < maxit-1:
                timeline.uniform_refine()
#                mesh.uniform_refine()
                n = n+1
                h = h/2
                nh = nh*2
                mesh = pde.init_mesh(n=n, h=h, nh=nh, p=p)
            print(errorMatrix)

        mesh.nodedata['uh'] = uh
        fig = plt.figure()
        axes = fig.gca(projection='3d')
        uh.add_plot(axes, cmap='rainbow')

        show_error_table(Ndof, errorType, errorMatrix)
        showmultirate(plt, 0, Ndof, errorMatrix, errorType)
        plt.show()

        mesh.to_vtk(fname='test.vtu') 

test = TimeIntegratorAlgTest()
test.test_ParabolicFEMModel_time()
