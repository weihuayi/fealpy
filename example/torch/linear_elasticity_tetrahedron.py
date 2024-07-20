import torch

from torch import Tensor

from fealpy.torch.mesh import TetrahedronMesh
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

cell_tet = mesh_tet.cell
print("cell_tet:", cell_tet.shape, "\n", cell_tet)
face_tet = mesh_tet.face
print("face_tet:", face_tet.shape, "\n", face_tet)
edge_tet = mesh_tet.edge
print("edge_tet:", edge_tet.shape, "\n", edge_tet)
node_tet = mesh_tet.node
print("node_tet:", node_tet.shape, "\n", node_tet)

GD_tet = mesh_tet.geo_dimension()
print("GD_tet:", GD_tet)

qf_tet = mesh_tet.integrator(3, 'cell')
# bcs-(NQ, BC), ws-(NQ, )
bcs, ws = qf_tet.get_quadrature_points_and_weights()
print("bcs_tri:", bcs.shape, "\n", bcs)
print("ws:", ws.shape, "\n", ws)

# (NC, BC, GD)
glambda_x_tet = mesh_tet.grad_lambda()
print("glambda_x_tet:", glambda_x_tet.shape, "\n", glambda_x_tet)
# (NC, (p+1)*(p+2)/2)
ipoints_tet = mesh_tet.cell_to_ipoint(p=3)
print("ipoints_tet:", ipoints_tet.shape, "\n", ipoints_tet)

space_tet = LagrangeFESpace(mesh_tet, p=1, ctype='C')
print("ldof_tet-(p+1)*(p+2)*(p+3)/6:", space_tet.number_of_local_dofs())
print("gdof_tet:", space_tet.number_of_global_dofs())
# (NC, LDOF)
cell2dof_tet = space_tet.cell_to_dof()
print("cell2dof_tet:", cell2dof_tet.shape, "\n", cell2dof_tet)
# (NQ, LDOF, BC)
gphi_lambda_tri = space_tri.grad_basis(bcs, variable='u')
print("gphi_lambda_tri:", gphi_lambda_tri.shape, "\n", gphi_lambda_tri)
# (NC, NQ, LDOF, GD)
gphi_x = space_tri.grad_basis(bcs, variable='x')
print("gphi_x:", gphi_x.shape, "\n", gphi_x)
#  phi-(1, NQ, LDOF)
phi = space_tri.basis(bcs, variable='x')
print("phi:", phi.shape, "\n", phi)

tensor_space_tri_node = TensorFunctionSpace(space_tri, shape=(GD_tri, -1))
print("tldof_tri-ldof_tri*GD:", tensor_space_tri_node.number_of_local_dofs())
print("tgdof:", tensor_space_tri_node.number_of_global_dofs())
# tcell2dof-(NC, TLDOF)
tcell2dof = tensor_space_tri_node.cell_to_dof()
print("tcell2dof:", tcell2dof.shape, "\n", tcell2dof)
# tphi-(1, NQ, TLDOF, GD)
tphi = tensor_space_tri_node.basis(bcs, variable='x')
print("tphi:", tphi.shape, "\n", tphi)

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

integrator_source = VectorSourceIntegrator(source=source)
# FF_tri - (NC, TLDOF)
FF_tri = integrator_source.assembly(space=tensor_space_tri_node)
print("FF_tri:", FF_tri.shape, "\n", FF_tri)
lform = LinearForm(tensor_space_tri_node)
lform.add_integrator(integrator_source)
# F_tri - (TGDOF)
F_tri = lform.assembly()
print("F_tri:", F_tri.shape, "\n", F_tri)