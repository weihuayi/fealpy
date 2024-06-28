from fealpy.np.functionspace import LagrangeFESpace as Space
from fealpy.np.mesh.triangle_mesh import TriangleMesh


mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=2, ny=2)
# import matplotlib.pyplot as plt
# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes)
# mesh.find_cell(axes, showindex=True, color='k', marker='s', markersize=2, fontsize=8, fontcolor='k')
# mesh.find_node(axes, showindex=True, color='r', marker='o', markersize=2, fontsize=8, fontcolor='r')
# plt.show()
space = Space(mesh, p=1, ctype='C')
GD = 2
vspace = GD*(space, )
uh = vspace[0].function(dim=GD)
print("uh:", uh.shape, "\n", uh)
gdof = vspace[0].number_of_global_dofs()
vgdof = gdof * GD
ldof = vspace[0].number_of_local_dofs()
vldof = ldof * GD
cell2dof = space.cell_to_dof()
print("gdof", gdof)
print("vldof", ldof)
print("cell2dof:", cell2dof)