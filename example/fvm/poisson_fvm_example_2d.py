from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.mesh import TetrahedronMesh
from fealpy.model import PDEDataManager
from fealpy.fvm import ScalarDiffusionIntegrator
from fealpy.fvm import BilinearForm
from fealpy.fvm import DirichletBC
from scipy.sparse.linalg import spsolve

def iterative_solution(A, b0, Tf, max_iter=1, tol=1e-6):
    uh1 = spsolve(A, b0) 
    for it in range(max_iter):
        Cross_diffusion = Integrator.Cross_diffusion(uh1, Tf)
        b = b0 + Cross_diffusion
        uh2 = spsolve(A, b)
        error = bm.max(bm.abs(uh2 - uh1))
        # print(f"Iteration {it+1}, error: {error}")
        if error < tol:
            print(f"Converged after {it+1} iterations, error: {error}")
            break
        uh1 = uh2
    return uh2  

pde = PDEDataManager('poisson').get_example('sinsin')
#pde = PDEDataManager('poisson').get_example('coscos')
#pde = PDEDataManager('poisson').get_example('sinsinsin')
#pde = PDEDataManager('poisson').get_example('coscoscos')
domain = pde.domain()
mesh = TriangleMesh.from_box(domain, nx=64, ny=64)
#mesh = TetrahedronMesh.from_box(domain, nx=18, ny=18, nz=18)
Integrator = ScalarDiffusionIntegrator(mesh)
bform = BilinearForm(mesh, pde)
dbc = DirichletBC(mesh, pde.dirichlet)
NC = mesh.number_of_cells()

flux_coeff, Tf = Integrator.vector_decomposition()
A = bform.matrix_assembly(flux_coeff)
b0 = mesh.integral(pde.source, q=3, celltype=True).reshape((NC, 1))
b0 = dbc.apply(b0)  
uh = iterative_solution(A, b0, Tf, max_iter=6, tol=1e-6)

cell_centers = mesh.bc_to_point(bm.array([1/3, 1/3, 1/3]))
#cell_centers = mesh.bc_to_point(bm.array([1/4, 1/4, 1/4, 1/4]))
uI = pde.solution(cell_centers)
cell_areas = mesh.entity_measure("cell")
l2_error = bm.sqrt(bm.sum(cell_areas * (uI - uh)**2))
print(f"L2 error: {l2_error}")
print(NC)

