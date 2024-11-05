
import numpy as np
import matplotlib.pyplot as plt
from fealpy.model.linear_elasticity_model import Model2d
from fealpy.functionspace.mixed_fem_space import HuZhangFiniteElementSpace

model = Model2d()
mesh = model.init_mesh(0)
V = HuZhangFiniteElementSpace(mesh, 4)

sI = V.interpolation(model.stress)

point = V.interpolation_points()
val = model.stress(point)
cell = mesh.ds.cell
cell2dof = V.dof.cell2dof
ldof = V.dof.number_of_local_dofs()


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_point(axes, point=point, showindex=True)
print(V.number_of_global_dofs())
print(cell)
print(cell2dof)
print(V.cell_to_dof().reshape(2, -1, 3))

plt.show()
