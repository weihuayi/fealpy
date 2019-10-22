import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import QuadrangleMesh
from fealpy.functionspace import QuadBilinearFiniteElementSpace 
from fealpy.pde.poisson_2d import CosCosData


def show_mesh(space):
    mesh = space.mesh
    cell = mesh.entity('cell')

    A0 = np.einsum('ijk, ij->ik', space.bvector, space.pvector + 1)

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    mesh.find_node(axes, node=space.ccenter, showindex=True)
    axes.quiver(
            space.ccenter[:, 0],
            space.ccenter[:, 1],
            space.bvector[:, 0, 0],
            space.bvector[:, 0, 1], angles='xy', scale_units='xy', scale=1)
    axes.quiver(
            space.ccenter[:, 0],
            space.ccenter[:, 1],
            space.bvector[:, 1, 0],
            space.bvector[:, 1, 1], angles='xy', scale_units='xy', scale=1)

    axes.quiver(
            space.ccenter[:, 0],
            space.ccenter[:, 1],
            A0[:, 0],
            A0[:, 1], angles='xy', scale_units='xy', scale=1)
    plt.show()

node = np.array([
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1),
    (0.5, 0),
    (1, 0.4),
    (0.3, 1),
    (0, 0.6),
    (0.5, 0.45)], dtype=np.float)

cell = np.array([
    (0, 4, 8, 7), (4, 1, 5, 8),
    (7, 8, 6, 3), (8, 5, 2, 6)], dtype=np.int)

pde = CosCosData()
mesh = QuadrangleMesh(node, cell)
mesh.uniform_refine(7)
space = QuadBilinearFiniteElementSpace(mesh)

uI = space.interpolation(pde.solution)

L2 = space.integralalg.L2_error(pde.solution, uI.value)
print("L2:", L2)
H1 = space.integralalg.L2_error(pde.gradient, uI.grad_value)
print("H1:", H1)


