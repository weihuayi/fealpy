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
NN_tri = mesh_tri.number_of_nodes()
NE_tri = mesh_tri.number_of_edges()
NC_tri = mesh_tri.number_of_cells()
NF_tri = mesh_tri.number_of_faces()
print("NN_tri:", NN_tri)
print("NE_tri:", NE_tri)
print("NC_tro:", NC_tri)
print("NF_tri:", NF_tri)
cell_tri = mesh_tri.cell
print("cell_tri:", cell_tri)
edge_tri = mesh_tri.edge
print("edge_tri:", edge_tri)
node_tri = mesh_tri.node
print("node_tri:", node_tri)
face_tri = mesh_tri.face
print("face_tri:", face_tri)
GD_tri = mesh_tri.geo_dimension()
print("GD_tri:", GD_tri)
qf = mesh_tri.quadrature_formula(3, 'cell')
# bcs-(NQ, BC)
bcs, ws = qf.get_quadrature_points_and_weights()
print("bcs_tri:", bcs.shape, "\n", bcs)
# (NC, LDOF, GD)
glambda_x_tri = mesh_tri.grad_lambda()
print("glambda_x_tri:", glambda_x_tri.shape)
# (NC, NI)
ipoints_tri = mesh_tri.cell_to_ipoint(p=3)
print("ipoints_tri:", ipoints_tri.shape, "\n", ipoints_tri)

# space_tri_old = LFS(mesh_tri_old, p=1, ctype='C', doforder='vdims')
space_tri = LagrangeFESpace(mesh_tri, p=1, ctype='C')
print("ldof_tri-(p+1)*(p+2)/2:", space_tri.number_of_local_dofs())
print("gdof_tri:", space_tri.number_of_global_dofs())
# (NC, LDOF)
cell2dof_tri = space_tri.cell_to_dof()
print("cell2dof_tri:", cell2dof_tri)
# (NQ, LDOF, BC)
gphi_lambda_tri = space_tri.grad_basis(bcs, index=_S, variable='u')
print("gphi_lambda_tri:", gphi_lambda_tri.shape)
# (NC, NQ, LDOF, GD)
gphi_x = space_tri.grad_basis(bcs, index=_S, variable='x')
print("gphi_x:", gphi_x.shape)
#  phi-(1, NQ, LDOF)
phi = space_tri.basis(bcs, index=_S, variable='x')
print("phi:", phi.shape, "\n", phi)

tensor_space_tri_node = TensorFunctionSpace(space_tri, shape=(GD_tri, -1))
print("tldof_tri-ldof_tri*GD:", tensor_space_tri_node.number_of_local_dofs())
print("tgdof:", tensor_space_tri_node.number_of_global_dofs())
# tcell2dof-(NC, TLDOF)
tcell2dof = tensor_space_tri_node.cell_to_dof()
print("tcell2dof:", tcell2dof.shape, "\n", tcell2dof)
# tphi-(1, NQ, TLDOF, GD)
tphi = tensor_space_tri_node.basis(bcs, index=_S, variable='x')
print("tphi:", tphi.shape, "\n", tphi)

# E0 = 1.0  # Elastic modulus
# nu = 0.3  # Poisson's ratio
# lambda_ = (E0 * nu) / ((1 + nu) * (1 - 2 * nu))
# mu = E0 / (2 * (1 + nu))
# integrator_strain_old = LEOI(lam=lambda_, mu=mu, q=3)
# vspace = 2 * (space_tri_old,)
# bform_strain_old = BF(vspace)
# bform_strain_old.add_domain_integrator(integrator_strain_old)
# KK_tri_strain_old = integrator_strain_old.assembly_cell_matrix(space=vspace)
# print("KK_tri_strain_old:", KK_tri_strain_old.shape, "\n", KK_tri_strain_old[0])
# K_tri_strain_old = bform_strain_old.assembly()
# print("K_tri_strain_old:", K_tri_strain_old.shape, "\n", K_tri_strain_old.toarray())

integrator_strain = LinearElasticityIntegrator(E=1.0, nu=0.3, \
                                        device=device, method='fast_strain')
# KK_tri_strain - (NC, TLDOF, TLDOF)
KK_tri_strain = integrator_strain.fast_assembly_strain_constant(space=tensor_space_tri_node)
print("KK_tri_strain:", KK_tri_strain.shape, "\n", KK_tri_strain[0])
integrator_stress = LinearElasticityIntegrator(E=1.0, nu=0.3, \
                                        device=device, method='fast_stress')
# KK_tri_stress - (NC, TLDOF, TLDOF)
KK_tri_stress = integrator_stress.fast_assembly_stress_constant(space=tensor_space_tri_node)
print("KK_tri_stress:", KK_tri_stress.shape, "\n", KK_tri_stress[0])

bform_strain = BilinearForm(tensor_space_tri_node)
bform_strain.add_integrator(integrator_strain)
K_tri_strain = bform_strain.assembly()
print("K_tri_strain:", K_tri_strain.shape, "\n", K_tri_strain.to_dense())
bform_stress = BilinearForm(tensor_space_tri_node)
bform_stress.add_integrator(integrator_stress)
K_tri_stress = bform_stress.assembly()
print("K_tri_stress:", K_tri_stress.shape, "\n", K_tri_stress.to_dense())

integrator_source = VectorSourceIntegrator(source=source, batched=False)
# FF_tri - (NC, TLDOF)
FF_tri = integrator_source.assembly(space=tensor_space_tri_node)
print("FF_tri:", FF_tri.shape, "\n", FF_tri)
lform = LinearForm(tensor_space_tri_node)
lform.add_integrator(integrator_source)
# F_tri - (TGDOF)
F_tri = lform.assembly()
print("F_tri:", F_tri.shape, "\n", F_tri)

# A, F = DirichletBC(tensor_space_tri_node).apply(K_tri_strain, F_tri, gd=solution)
# asd

uh_tri = tensor_space_tri_node.function(dim=2)
print("uh_tri:", uh_tri.shape)

isDDof = tensor_space_tri_node.is_boundary_dof()
print("is_DDof:", isDDof.shape, "\n", isDDof)
uh_tri_interpolate = space_tri.interpolate(source=source, uh=uh_tri)
print("uh_tri_interpolate:", uh_tri_interpolate.shape, "\n", uh_tri_interpolate)
uh_tri[isDDof] = solution(uh_tri_interpolate[isDDof])
print("uh_tri:", uh_tri.shape, "\n", uh_tri)



asd

## ---------------------------------------------------------------------------------------
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
qf_quad = mesh_quad.integrator(3, 'cell')
# bcs-(NQ, BC)
bcs, ws = qf_quad.get_quadrature_points_and_weights()
print("bcs_quad:", bcs, "\n", bcs[0].shape, "-", bcs[1].shape)
# # (NC, LDOF, GD)
# glambda_x_quad = mesh_quad.grad_lambda()
# print("glambda_x_quad:", glambda_x_quad.shape)
# (NC, NI) 先排 y 再排 x
ipoints_quad = mesh_quad.cell_to_ipoint(p=2)
print("ipoints_quad:", ipoints_quad.shape, "\n", ipoints_quad)

space_quad = LagrangeFESpace(mesh_quad, p=1, ctype='C')
print("ldof_quad-(p+1)**2:", space_quad.number_of_local_dofs())
print("gdof_quad:", space_quad.number_of_global_dofs())
# (NC, LDOF)
cell2dof_quad = space_quad.cell_to_dof()
print("cell2dof_quad:", cell2dof_quad)
# phi_quad = space_quad.basis(bcs, index=_S, variable='x')
# print("phi_quad:", phi_quad.shape)
# # (NQ, LDOF, BC)
# gphi_lambda_quad = space_quad.grad_basis(bcs, index=_S, variable='u')
# print("gphi_lambda_quad:", gphi_lambda_quad.shape)

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



## ---------------------------------------------------------------------------------------
NX = 1
NY = 1
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
GD_tet = mesh_tet.geo_dimension()
print("GD_tet:", GD_tet)
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
print("ldof_tet-(p+1)*(p+2)*(p+3)/6:", space_tet.number_of_local_dofs())
print("gdof_tet:", space_tet.number_of_global_dofs())
# (NQ, LDOF, BC)
gphi_lambda_tet = space_tet.grad_basis(bcs, index=_S, variable='u')
print("gphi_lambda_tet:", gphi_lambda_tet.shape)
# (NC, NQ, LDOF, GD)
gphi_x_tet = space_tet.grad_basis(bcs, index=_S, variable='x')
print("gphi_x_tet:", gphi_x_tet.shape)
#  phi-(1, NQ, ldof)
phi_tet = space_tet.basis(bcs, index=_S, variable='x')
print("phi_tet:", phi_tet.shape)

tensor_space_tet_node = TensorFunctionSpace(space_tet, shape=(3, -1))
print("tldof_tet-ldof_tet*GD:", tensor_space_tet_node.number_of_local_dofs())
print("tgdof_tet:", tensor_space_tet_node.number_of_global_dofs())
integrator_3d = LinearElasticityIntegrator(E=1.0, nu=0.3, \
                                        device=device, method='fast_3d')
KK_tet_3d = integrator_3d.fast_assembly_constant(space=tensor_space_tet_node)
print("KK_tet_3d:", KK_tet_3d.shape, "\n", KK_tet_3d[0])



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

