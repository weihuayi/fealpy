 
from fealpy.experimental.functionspace import LagrangeFESpace
from fealpy.experimental.mesh import TriangleMesh
from fealpy.experimental.pde.semilinear_2d import SemilinearData
from fealpy.experimental.backend import backend_manager as bm
from functools import partial

bm.set_backend('pytorch')
domain = [0, 1, 0, 2]
nx = 4
ny = 4
pde = SemilinearData(domain)
mesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)
space = LagrangeFESpace(mesh, p=1)
uh = space.function()

q=3
uh_ = uh[space.cell_to_dof()]
qf = mesh.quadrature_formula(q, 'cell')
cm = mesh.entity_measure('cell')
bcs, ws = qf.get_quadrature_points_and_weights()
phi = space.basis(bcs) #(1, NQ, ldof)

def kernel_fdunc(u):
    return u**3

def cell_integral(ws, phi, val):
    return bm.einsum(f'q, qi, qj, ...j -> ...i', ws, phi[0], phi[0], val) * cm[0]

def auto_grad(ws, phi, val):
    fn = bm.vmap(bm.jacfwd(
          partial(cell_integral, ws, phi)
          ))
    return fn(val)

def test():
    print(auto_grad(ws, phi, kernel_fdunc(uh_)))
    
if __name__ == "__main__":
    test()
