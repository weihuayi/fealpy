
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from fealpy.mesh.level_set_function import dcircle
from fealpy.mesh.interface_mesh_generator import interfacemesh2d

from fealpy.form.Form import LaplaceSymetricForm, MassForm, SourceForm
from fealpy.boundarycondition import DirichletBC
from fealpy.functionspace.function import FiniteElementFunction 
from fealpy.functionspace.tools import function_space

from fealpy.timeintegratoralg.TimeIntegratorAlgorithm import TimeIntegratorAlgorithm

class HeatEquationSolver(TimeIntegratorAlgorithm):
    def __init__(self, model, mesh, interval, N):
        super(HeatEquationSolver, self).__init__(interval)
        self.model = model
        self.dt = (interval[1] - interval[0])/N

        V = function_space(mesh, 'Lagrange', 1)
        uh = FiniteElementFunction(V)
        uh[:] = model.init_value(V.interpolation_points())

        self.solution = [uh]

        self.BC = DirichletBC(V, model.dirichlet, model.is_dirichlet_boundary)
        self.stiff_matrix = LaplaceSymetricForm(V, 3).get_matrix()
        self.mass_matrix = MassForm(V, 3).get_matrix()

    def get_step_length(self):
        return self.dt

    def get_left_hand_matrix(self, t):
        dt = self.dt
        A = self.stiff_matrix
        M = self.mass_matrix

    def get_right_hand_vector(self, t):
        pass

    def accept_solution(self, current_solution):
        pass

    def solve(self, A, b):
        pass
n = 10
box = [-1, 1, -1, 1]
cxy = (0.0, 0.0)
r = 0.5
phi = lambda p: dcircle(p, cxy, r)
mesh = interfacemesh2d(box, phi, n)

# plot 
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, pointcolor=mesh.pointMarker) 

circle = Circle(cxy, r, edgecolor='g', fill=False, linewidth=2)
axes.add_patch(circle)
plt.show()
