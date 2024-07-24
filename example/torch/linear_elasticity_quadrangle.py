import torch

from torch import Tensor

from fealpy.torch.mesh import QuadrangleMesh
from fealpy.torch.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.torch.fem import LinearElasticityIntegrator, BilinearForm,\
                             LinearForm, VectorSourceIntegrator

def source(points: Tensor) -> Tensor:
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype, 'device': points.device}
    
    val = torch.zeros(points.shape, **kwargs)
    val[..., 0] = 35/13 * y - 35/13 * y**2 + 10/13 * x - 10/13 * x**2
    val[..., 1] = -25/26 * (-1 + 2 * y) * (-1 + 2 * x)
    
    return val

def solution(points: Tensor) -> Tensor:
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype, 'device': points.device}
    
    val = torch.zeros(points.shape, **kwargs)
    val[..., 0] = x * (1 - x) * y * (1 - y)
    val[..., 1] = 0
    
    return val

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

NX = 2
NY = 2
mesh_quad = QuadrangleMesh.from_box(box=[0, 1, 0, 1], nx=NX, ny=NY, device=device)
NN_quad = mesh_quad.number_of_nodes()
NC_quad = mesh_quad.number_of_cells()
NE_quad = mesh_quad.number_of_edges()
NF_quad = mesh_quad.number_of_faces()
print("NN_quad:", NN_quad)
print("NC_quad:", NC_quad)
print("NE_quad:", NE_quad)
print("NF_quad:", NF_quad)

cell_quad = mesh_quad.cell
print("cell_quad:", cell_quad)
edge_quad = mesh_quad.edge
print("edge_quad:", edge_quad)
node_quad = mesh_quad.node
print("node_quad:", node_quad)
face_quad = mesh_quad.face
print("face_quad:", face_quad)

GD_quad = mesh_quad.geo_dimension()
print("GD_quad:", GD_quad)

qf = mesh_quad.quadrature_formula(3, 'cell')
# bcs-(NQ, BC), ws-(NQ, )
bcs, ws = qf.get_quadrature_points_and_weights()
print("bcs:\n", bcs)
print("ws:", ws.shape, "\n", ws)

# (NC, LDOF, GD)
# glambda_x_quad = mesh_quad.grad_lambda()
# print("glambda_x_quad:", glambda_x_quad.shape)
# (NC, (p+1)*(p+2)/2) 先排 y 再排 x
ipoints_quad = mesh_quad.cell_to_ipoint(p=2)
print("ipoints_quad:", ipoints_quad.shape, "\n", ipoints_quad)

space_quad = LagrangeFESpace(mesh_quad, p=1, ctype='C')
print("ldof_quad-(p+1)**2:", space_quad.number_of_local_dofs())
print("gdof_quad:", space_quad.number_of_global_dofs())
# (NC, LDOF)
cell2dof_quad = space_quad.cell_to_dof()
print("cell2dof_quad:", cell2dof_quad)
# # (NQ, LDOF, BC)
# gphi_lambda_quad = space_quad.grad_basis(bcs, variable='u')
# print("gphi_lambda_quad:", gphi_lambda_quad.shape, "\n", gphi_lambda_quad)
# # (NC, NQ, LDOF, GD)
# gphi_x = space_quad.grad_basis(bcs, variable='x')
# print("gphi_x:", gphi_x.shape, "\n", gphi_x)
#  phi-(1, NQ, LDOF)
phi = space_quad.basis(bcs, variable='x')
print("phi:", phi.shape, "\n", phi)

tensor_space_quad_node = TensorFunctionSpace(space_quad, shape=(2, -1))
print("tldof_quad:", tensor_space_quad_node.number_of_local_dofs())
print("tgdof_quad:", tensor_space_quad_node.number_of_global_dofs())
# integrator_strain = LinearElasticityIntegrator(E=1.0, nu=0.3, \
#                                         device=device, method='fast_strain')
# KK_quad_strain = integrator_strain.fast_assembly_strain_constant(space=tensor_space_quad_node)
# print("KK_quad_strain:", KK_quad_strain.shape, "\n", KK_quad_strain[0])
# integrator_stress = LinearElasticityIntegrator(E=1.0, nu=0.3, \
#                                         device=device, method='fast_stress')
# KK_quad_stress = integrator_stress.fast_assembly_stress_constant(space=tensor_space_node)
# print("KK_quad_stress:", KK_quad_stress.shape, "\n", KK_quad_stress[0])