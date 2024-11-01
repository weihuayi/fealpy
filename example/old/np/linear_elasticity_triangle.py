import numpy as np

from fealpy.decorator import cartesian

from fealpy.np.mesh import TriangleMesh
from fealpy.np.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.np.fem import LinearElasticityIntegrator, BilinearForm, \
                          LinearForm, VectorSourceIntegrator

@cartesian
def source(points):
    x = points[..., 0]
    y = points[..., 1]
    val = np.zeros(points.shape, dtype=np.float64)
    val[..., 0] = 35/13*y - 35/13*y**2 + 10/13*x - 10/13*x**2
    val[..., 1] = -25/26*(-1+2*y) * (-1+2*x)

    return val

NX = 2
NY = 2
mesh_tri = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=NX, ny=NY)
NN_tri = mesh_tri.number_of_nodes()
NE_tri = mesh_tri.number_of_edges()
NC_tri = mesh_tri.number_of_cells()
NF_tri = mesh_tri.number_of_faces()
print("NN_tri:", NN_tri)
print("NE_tri:", NE_tri)
print("NC_tro:", NC_tri)
print("NF_tri:", NF_tri)

cell_tri = mesh_tri.cell
print("cell_tri:\n", cell_tri)
edge_tri = mesh_tri.edge
print("edge_tri:\n", edge_tri)
node_tri = mesh_tri.node
print("node_tri:\n", node_tri)
face_tri = mesh_tri.face
print("face_tri:\n", face_tri)

GD_tri = mesh_tri.geo_dimension()
print("GD_tri:", GD_tri)

qf = mesh_tri.quadrature_formula(3, 'cell')
# bcs-(NQ, BC), ws-(NQ, )
bcs, ws = qf.get_quadrature_points_and_weights()
print("bcs_tri:", bcs.shape, "\n", bcs)
print("ws:", ws.shape, "\n", ws)

# (NC, BC, GD)
glambda_x_tri = mesh_tri.grad_lambda()
print("glambda_x_tri:", glambda_x_tri.shape, "\n", glambda_x_tri)
# (NC, (p+1)*(p+2)/2)
ipoints_tri = mesh_tri.cell_to_ipoint(p=3)
print("ipoints_tri:", ipoints_tri.shape, "\n", ipoints_tri)

space_tri = LagrangeFESpace(mesh_tri, p=1, ctype='C')
print("ldof_tri-(p+1)*(p+2)/2:", space_tri.number_of_local_dofs())
print("gdof_tri:", space_tri.number_of_global_dofs())
# (NC, LDOF)
cell2dof_tri = space_tri.cell_to_dof()
print("cell2dof_tri:", cell2dof_tri.shape, "\n", cell2dof_tri)
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

integrator_strain = LinearElasticityIntegrator(E=1.0, nu=0.3, method='fast_strain')
# KK_tri_strain - (NC, TLDOF, TLDOF)
KK_tri_strain = integrator_strain.fast_assembly_strain_constant(space=tensor_space_tri_node)
print("KK_tri_strain:", KK_tri_strain.shape, "\n", KK_tri_strain[0].round(4))
integrator_stress = LinearElasticityIntegrator(E=1.0, nu=0.3, method='fast_stress')
# KK_tri_stress - (NC, TLDOF, TLDOF)
KK_tri_stress = integrator_stress.fast_assembly_stress_constant(space=tensor_space_tri_node)
print("KK_tri_stress:", KK_tri_stress.shape, "\n", KK_tri_stress[0].round(4))

bform_strain = BilinearForm(tensor_space_tri_node)
bform_strain.add_integrator(integrator_strain)
K_tri_strain = bform_strain.assembly()
print("K_tri_strain:", K_tri_strain.shape, "\n", K_tri_strain.toarray().round(4))
bform_stress = BilinearForm(tensor_space_tri_node)
bform_stress.add_integrator(integrator_stress)
K_tri_stress = bform_stress.assembly()
print("K_tri_stress:", K_tri_stress.shape, "\n", K_tri_stress.toarray().round(4))

integrator_source = VectorSourceIntegrator(source=source)
# FF_tri - (NC, TLDOF)
FF_tri = integrator_source.assembly(space=tensor_space_tri_node)
print("FF_tri:", FF_tri.shape, "\n", FF_tri.round(4))
lform = LinearForm(tensor_space_tri_node)
lform.add_integrator(integrator_source)
# F_tri - (TGDOF)
F_tri = lform.assembly()
print("F_tri:", F_tri.shape, "\n", F_tri.round(4))