 
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
gphi = space.grad_basis(bcs) #(NC, NQ, ldof, dof_numel)
coef = 1

def kernel_func(u):
    return u**3 + u + 1

def cell_integral(u, gphi, cm, coef, ws):
    val = kernel_func(bm.einsum('i, qid -> qd', u, gphi))
    return bm.einsum('q, qid, qd -> i', ws, gphi, val) * cm * coef

def auto_grad(uh_, gphi, cm, coef, ws):
    fn_A = bm.vmap(bm.jacfwd(                         
        partial(cell_integral, coef=coef, ws=ws)
        ))
    fn_F = bm.vmap(
        partial(cell_integral, coef=coef, ws=ws)
    )
    return  fn_A(uh_, gphi, cm), -fn_F(uh_, gphi, cm)

def test():
    print(auto_grad(uh_, gphi, cm, coef, ws)[0],\
          auto_grad(uh_, gphi, cm, coef, ws)[1])
    
if __name__ == "__main__":
    test()
