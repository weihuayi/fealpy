import numpy as np

from fealpy.mesh.tree_data_structure import Quadtree
from fealpy.mesh.QuadrangleMesh import QuadrangleMesh 
from fealpy.mesh.PolygonMesh import PolygonMesh


from fealpy.model.poisson_model_2d import LShapeRSinData
from fealpy.form.vem import LaplaceSymetricForm
from fealpy.form.vem import SourceForm
from fealpy.boundarycondition import DirichletBC

from fealpy.solver import solve

def lshape_mesh(r=1):
    point = np.array([
        (-1, -1),
        (0, -1),
        (-1, 0),
        (0, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1)], dtype=np.float)
    cell = np.array([
        (0, 1, 3, 2),
        (2, 3, 6, 5),
        (3, 4, 7, 6)], dtype=np.int)
    return point, cell


def vem_solve(model, quadtree):
    mesh = PolygonMesh.from_quadtree(quadtree)
    V = VirtualElementSpace2d(mesh, 1) 
    a = LaplaceSymetricForm(V)
    L = SourceForm(V, model.source)

    uh = FiniteElementFunction(V)
    Ndof[i] = V.number_of_global_dofs() 

    BC = DirichletBC(V, model.dirichlet)
    solve(a, L, uh, dirichlet=BC, solver='direct')
    return uh

point, cell = lshape_mesh(4)
quadtree = Quadtree(point, cell)
quadtree.uniform_refine(4)
model = LShapeRSinData() 

maxit = 4  
error = np.zeros((maxit,), dtype=np.float)
Ndof = np.zeros((maxit,), dtype=np.int)

for i in range(maxit):
    uh = vem_solve(model, quadtree)
    uI = uh.V.interpolation(model.solution)
    error[i] = np.sqrt(np.sum((uh - uI)**2)/Ndof[i])
    if i < mqxit - 1:
        quadtree.uniform_refine()

print(Ndof)
print(error)
print(error[:-1]/error[1:])
