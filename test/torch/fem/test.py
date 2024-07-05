import pytest
import torch
from torch import Tensor, einsum
from fealpy.torch.mesh import TriangleMesh, TetrahedronMesh, QuadrangleMesh
from fealpy.torch.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.torch.fem import LinearElasticityIntegrator, BilinearForm, \
                             LinearForm, VectorSourceIntegrator
from fealpy.torch.fem.integrator import CellSourceIntegrator, _S, Index, CoefLike, enable_cache

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

NX = 1
NY = 1
mesh_tri = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=NX, ny=NY, device=device)
NN_tri = mesh_tri.number_of_nodes()
NE_tri = mesh_tri.number_of_edges()
NC_tri = mesh_tri.number_of_cells()
NF_tri = mesh_tri.number_of_faces()
print("NN:", NN_tri)
print("NE:", NE_tri)
print("NC:", NC_tri)
print("NF:", NF_tri)
cell_tri = mesh_tri.cell
print("cell_tri:", cell_tri)
edge_tri = mesh_tri.edge
print("edge_tri:", edge_tri)
node_tri = mesh_tri.node
print("node_tri:", node_tri)
space_tri = LagrangeFESpace(mesh_tri, p=1, ctype='C')
print("ldof_tri:", space_tri.number_of_local_dofs())
print("gdof_tri:", space_tri.number_of_global_dofs())
qf = mesh_tri.quadrature_formula(3, 'cell')
# bcs-(NQ, BC)
bcs, ws = qf.get_quadrature_points_and_weights()
print("bcs_tri:", bcs.shape)
# (NC, LDOF, GD)
glambda_x_tri = mesh_tri.grad_lambda()
print("glambda_x_tri:", glambda_x_tri.shape)
#  phi-(1, NQ, ldof)
phi = space_tri.basis(bcs, index=_S, variable='x')
print("phi:", phi.shape)
tensor_space_node = TensorFunctionSpace(space_tri, shape=(2, -1))
print("tldof:", tensor_space_node.number_of_local_dofs())
print("tgdof:", tensor_space_node.number_of_global_dofs())
# tphi-(1, NQ, tldof, GD)
tphi = tensor_space_node.basis(bcs, index=_S, variable='x')
print("tphi:", tphi.shape)
tcell2dof = tensor_space_node.cell_to_dof()
print("tcell2dof:\n", tcell2dof)
cell_tri = mesh_tri.cell
print("cell_tri:", cell_tri)
edge_tri = mesh_tri.edge
print("edge_tri:", edge_tri)
node_tri = mesh_tri.node
print("node_tri:", node_tri)
integrator_strain = LinearElasticityIntegrator(E=1.0, nu=0.3, \
                                        device=device, method='fast_strain')
KK_tri_strain = integrator_strain.fast_assembly_strain_constant(space=tensor_space_node)
print("KK_tri_strain:", KK_tri_strain.shape, "\n", KK_tri_strain[0])
integrator_stress = LinearElasticityIntegrator(E=1.0, nu=0.3, \
                                        device=device, method='fast_stress')
KK_tri_stress = integrator_stress.fast_assembly_stress_constant(space=tensor_space_node)
print("KK_tri_stress:", KK_tri_stress.shape, "\n", KK_tri_stress[0])
bform = BilinearForm(tensor_space_node)
bform.add_integrator(integrator_stress)
K_tri_stress = bform.assembly()
print("K_tri_stress:", K_tri_stress.shape, "\n", K_tri_stress.to_dense())

# integrator_source = VectorSourceIntegrator(source=source, batched=False)
# FF_tri = integrator_source.assembly(space=tensor_space_node)
# lform = LinearForm(tensor_space_node, batch_size=10)
# lform.add_integrator(integrator_source)
# print("FF_tri:", FF_tri)
# F_tri = lform.assembly()
# print("F_tri:", F_tri.shape, "\n", F_tri)


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
# qf_quad = mesh_quad.integrator(3, 'cell')
# print("qf_quad:", qf_quad)
space_quad = LagrangeFESpace(mesh_quad, p=1, ctype='C')
print("ldof:", space_quad.number_of_local_dofs())
print("gdof:", space_quad.number_of_global_dofs())
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

NZ = 1
mesh_tet = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=NX, ny=NY, nz=NZ, device=device)
NN_tet = mesh_tet.number_of_nodes()
NC_tet = mesh_tet.number_of_cells()
NE_tet = mesh_tet.number_of_edges()
NF_tet = mesh_tet.number_of_faces()
print("NN_tet:", NN_tet)
print("NC_tet:", NC_tet)
print("NE_tet:", NE_tet)
print("NF_tet:", NF_tet)
cell_tet = mesh_tet.cell
print("cell_tet:", cell_tet.shape, "\n", cell_tet)
face_tet = mesh_tet.face
print("face_tet:", face_tet.shape, "\n", face_tet)
edge_tet = mesh_tet.edge
print("edge_tet:", edge_tet.shape, "\n", edge_tet)
node_tet = mesh_tet.node
print("node_tet:", node_tet.shape, "\n", node_tet)
qf_tet = mesh_tet.integrator(3, 'cell')
# bcs-(NQ, BC)
bcs, ws = qf_tet.get_quadrature_points_and_weights()
print("bcs:", bcs.shape)
# (NC, LDOF, GD)
glambda_x_tet = mesh_tet.grad_lambda()
print("glambda_x_tet:", glambda_x_tet.shape)
space_tet = LagrangeFESpace(mesh_tet, p=1, ctype='C')
print("ldof_tet:", space_tet.number_of_local_dofs())
print("gdof_tet:", space_tet.number_of_global_dofs())
tensor_space_tet_node = TensorFunctionSpace(space_tet, shape=(2, -1))
print("tldof_tet:", tensor_space_tet_node.number_of_local_dofs())
print("tgdof_tet:", tensor_space_tet_node.number_of_global_dofs())
integrator_strain = LinearElasticityIntegrator(E=1.0, nu=0.3, \
                                        device=device, method='fast_strain')
KK_tet_strain = integrator_strain.fast_assembly_strain_constant(space=tensor_space_tet_node)
print("KK_tet_strain:", KK_tet_strain.shape, "\n", KK_tet_strain[0])



# tensor_space = TensorFunctionSpace(space, shape=(-1, 2))

# integrator = LinearElasticityIntegrator(E=1.0, nu=0.3, \
#                                         device=device, method='fast_strain')
# KK_torch = integrator.assembly(space=tensor_space)
# bform = BilinearForm(tensor_space)
# K = bform.assembly()

# test = integrator.fast_assembly_strain_constant(space=tensor_space)
# print("test:", test.shape, "\n", test)


# integrator1 = LinearElasticityIntegrator(E=1.0, nu=0.3, 
#                                          device=device, method='fast_stress')
# test1 = integrator1.fast_assembly_stress_constant(space=tensor_space)
# print("test1:", test1.shape, "\n", test1)

