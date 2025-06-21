 
from fealpy.functionspace import LagrangeFESpace
from fealpy.mesh import TriangleMesh
from fealpy.pde.semilinear_2d import SemilinearData
from fealpy.backend import backend_manager as bm
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
coef = 1

def kernel_func(u):
    return u**3 + u + 1

def cell_integral(u, cm, phi, ws, coef):
    val = kernel_func(bm.einsum('i, qi -> q', u, phi[0]))
    return bm.einsum('q, qi, q -> i', ws, phi[0], val) * cm * coef

def auto_grad(uh_, cm, coef, ws, phi):
    fn_A = bm.vmap(bm.jacfwd(                         
        partial(cell_integral, coef=coef, ws=ws, phi=phi)
        ))
    fn_F = bm.vmap(
        partial(cell_integral, coef=coef, ws=ws, phi=phi)
    )
    return  fn_A(uh_, cm), -fn_F(uh_, cm)

def test():
    print(auto_grad(uh_, cm, coef, ws, phi)[0],\
          auto_grad(uh_, cm, coef, ws, phi)[1])
    
if __name__ == "__main__":
    test()
