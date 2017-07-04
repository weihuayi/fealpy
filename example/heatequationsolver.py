import numpy as np
from scipy.sparse.linalg import cg, inv, dsolve, spsolve
import matplotlib.pyplot as plt
from fealpy.form.Form import LaplaceSymetricForm, MassForm, SourceForm
from fealpy.boundarycondition import DirichletBC
from fealpy.functionspace.function import FiniteElementFunction 
from fealpy.functionspace.tools import function_space
from fealpy.timeintegratoralg.TimeIntegratorAlgorithm import TimeIntegratorAlgorithm
from fealpy.erroranalysis import L2_error


class HeatEquationSolver(TimeIntegratorAlgorithm):
    def __init__(self, model, mesh, initTime, stopTime, N, method='FM'):
        super(HeatEquationSolver, self).__init__(initTime, stopTime)
        self.model = model
        self.N = N
        self.dt = (stopTime - initTime)/N
        self.method = method
        self.V = function_space(mesh, 'Lagrange', 1)
        uh = FiniteElementFunction(self.V)
        uh[:] = model.init_value(self.V.interpolation_points())

        self.solution = [uh]
        self.maxError = 0.0 
        self.stiffMatrix = LaplaceSymetricForm(self.V, 3).get_matrix()
        self.massMatrix = MassForm(self.V, 3).get_matrix()
    
    def get_step_length(self):
        return self.dt

    def get_current_linear_system(self):
        V = self.V
        model = self.model

        t = self.currentTime

        S = self.stiffMatrix
        M = self.massMatrix
        k = model.diffusion_coefficient()
        bc = DirichletBC(self.V, lambda p: model.dirichlet(p, t), model.is_dirichlet_boundary)
        F = SourceForm(V, lambda p: model.source(p, t), 3).get_vector() 
        dt = self.dt
        if self.method is 'FM':
            b = dt*(F - k*S@self.solution[-1]) + M@self.solution[-1]
            A, b = bc.apply(M, b) 
            return A, b
        if self.method is 'BM':
            b = dt*F + M@self.solution[-1]
            A = M + dt*k*S
            A, b = bc.apply(A, b)
            return A, b
        if self.method is 'CN':
            b = dt*F + (M - 0.5*dt*k*S)@self.solution[-1]
            A = M + 0.5*dt*k*S
            A, b = bc.apply(A, b)
            return A, b
    def accept_solution(self, currentSolution):
        self.solution.append(currentSolution)

        V = self.V
        model = self.model
        t = self.currentTime
        u = model.solution(V.interpolation_points(), t)
        e = np.max(np.abs(u - currentSolution))
        self.maxError = max(e, self.maxError)
    
    def solve(self, A, b):
        uh = FiniteElementFunction(self.V)
        uh[:] = spsolve(A, b)
        return uh


## The following is the test part
from fealpy.model.heatequation_model_2d import  SinCosExpData, SinSinExpData
from fealpy.mesh.simple_mesh_generator import rectangledomainmesh

model = SinCosExpData()
#moedl = SinSinExpData()
box = [0, 1, 0, 1]                                           
initTime = 0.0                                                    
stopTime = 1.0     
N = 4000
n = np.array([16, 32, 64])
emax = np.zeros(len(n),dtype = np.float)
for j in range(len(n)):
    mesh = rectangledomainmesh(box, nx=n[j], ny=n[j], meshtype='tri')
    solver = HeatEquationSolver(model, mesh, initTime, stopTime, N, method = 'BM')
    solver.run()
    print(solver.maxError)
    emax[j] = solver.maxError
    
print(n)
print('max error:\n', emax)
order = np.log(emax[0:-1]/emax[1:])/np.log(2)
print('order:\n', order)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()
