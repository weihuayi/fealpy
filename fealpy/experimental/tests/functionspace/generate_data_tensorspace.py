import torch
from torch import Tensor, einsum
from fealpy.torch.mesh import TriangleMesh, TetrahedronMesh, QuadrangleMesh
from fealpy.torch.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.torch.fem import LinearElasticityIntegrator, BilinearForm, \
                             LinearForm, VectorSourceIntegrator, DirichletBC
from fealpy.torch.fem.integrator import CellSourceIntegrator, _S, Index, CoefLike, enable_cache

# import numpy as np
# from fealpy.decorator import cartesian
# from fealpy.fem import LinearElasticityOperatorIntegrator as LEOI
# from fealpy.fem import BilinearForm as BF
# from fealpy.fem import LinearForm as LF
# from fealpy.fem import VectorSourceIntegrator as VSI
# from fealpy.functionspace import LagrangeFESpace as LFS
# from fealpy.mesh import TriangleMesh as TMD

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.set_printoptions(precision=7)
def source(points: Tensor) -> Tensor:
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype, 'device': points.device}
    
    val = torch.zeros(points.shape, **kwargs)
    val[..., 0] = 35/13 * y - 35/13 * y**2 + 10/13 * x - 10/13 * x**2
    val[..., 1] = -25/26 * (-1 + 2 * y) * (-1 + 2 * x)
    
    return val

# @cartesian
# def source_old(p):
#     x = p[..., 0]
#     y = p[..., 1]
#     val = np.zeros(p.shape, dtype=np.float64)
#     val[..., 0] = 35/13*y - 35/13*y**2 + 10/13*x - 10/13*x**2
#     val[..., 1] = -25/26*(-1+2*y) * (-1+2*x)

#     return val

def source1(points: Tensor) -> Tensor:
        x = points[..., 0]
        y = points[..., 1]
        val = 2*torch.pi*torch.pi*torch.cos(torch.pi*x)*torch.cos(torch.pi*y)

        return val

def solution(points: Tensor) -> Tensor:
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype, 'device': points.device}
    
    val = torch.zeros(points.shape, **kwargs)
    val[..., 0] = x * (1 - x) * y * (1 - y)
    val[..., 1] = 0
    
    return val

NX = 2
NY = 2
mesh_tri = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=NX, ny=NY, device=device)
# mesh_tri_old = TMD.from_box(box=[0, 1, 0, 1], nx=NX, ny=NY)

GD_tri = mesh_tri.geo_dimension()
#print("GD_tri:", GD_tri)
qf = mesh_tri.quadrature_formula(2, 'cell')
# bcs-(NQ, BC)
bcs, ws = qf.get_quadrature_points_and_weights()
#print("bcs_tri:", bcs.shape, "\n", bcs)
# (NC, LDOF, GD)
glambda_x_tri = mesh_tri.grad_lambda()
#print("glambda_x_tri:", glambda_x_tri.shape)
# (NC, NI)
ipoints_tri = mesh_tri.cell_to_ipoint(p=2)
#print("ipoints_tri:", ipoints_tri.shape, "\n", ipoints_tri)

# space_tri_old = LFS(mesh_tri_old, p=1, ctype='C', doforder='vdims')
space_tri = LagrangeFESpace(mesh_tri, p=1, ctype='C')
# print("ldof_tri-(p+1)*(p+2)/2:", space_tri.number_of_local_dofs())
# print("gdof_tri:", space_tri.number_of_global_dofs())
# (NC, LDOF)
cell2dof_tri = space_tri.cell_to_dof()
#print("cell2dof_tri:", cell2dof_tri)
# (NQ, LDOF, BC)
gphi_lambda_tri = space_tri.grad_basis(bcs, index=_S, variable='u')
#print("gphi_lambda_tri:", gphi_lambda_tri.shape)
# (NC, NQ, LDOF, GD)
gphi_x = space_tri.grad_basis(bcs, index=_S, variable='x')
#print("gphi_x:", gphi_x.shape)
#  phi-(1, NQ, LDOF)
phi = space_tri.basis(bcs, index=_S, variable='x')
#print("phi:", phi.shape, "\n", phi)

tensor_space_tri_node = TensorFunctionSpace(space_tri, shape=(GD_tri, -1))
#print("tldof_tri-ldof_tri*GD:", tensor_space_tri_node.number_of_local_dofs())
#print("tgdof:", tensor_space_tri_node.number_of_global_dofs())
# tcell2dof-(NC, TLDOF)
tdofnumel = tensor_space_tri_node.dof_numel 
#print("tdofnumel:", tdofnumel)
tGD = tensor_space_tri_node.dof_ndim 
#print("GD:", tGD)    
tld = space_tri.number_of_local_dofs()
#print("ld:", tld)
tcell2dof = tensor_space_tri_node.cell_to_dof()
#print("tcell2dof:", tcell2dof.shape, "\n", tcell2dof)
# tphi-(1, NQ, TLDOF, GD)
tphi = tensor_space_tri_node.basis(bcs, index=_S, variable='x')
print("tphi:", tphi.shape, "\n", tphi)
tgrad_phi = tensor_space_tri_node.grad_basis(bcs, index=_S, variable='x')
#print("tgrad_phi:", tgrad_phi.shape, "\n", tgrad_phi)
tface2dof = tensor_space_tri_node.face_to_dof() 
#print("tface2dof:", tface2dof.shape, "\n", tface2dof)
