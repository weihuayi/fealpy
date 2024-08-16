from fealpy.experimental.mesh import UniformMesh3d, HexahedronMesh

from fealpy.experimental.fem import LinearElasticityIntegrator, \
                                    BilinearForm, LinearForm, \
                                    VectorSourceIntegrator

from fealpy.fem import LinearElasticityOperatorIntegrator as LinearElasticityIntegrator_old
from fealpy.fem import VectorSourceIntegrator as VectorSourceIntegrator_old
from fealpy.fem import BilinearForm as BilinearForm_old
from fealpy.fem import LinearForm as LinearForm_old
from fealpy.fem import DirichletBC as DirichletBC_old

from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.experimental.typing import TensorLike
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.solver import cg
from fealpy.experimental.fem import DirichletBC as DBC
from fealpy.experimental.sparse import COOTensor


bm.set_backend('numpy')

# 平面应变问题定义
def source(points: TensorLike) -> TensorLike:
    x = points[..., 0]
    y = points[..., 1]
    
    val = bm.zeros(points.shape, dtype=points.dtype)
    val[..., 0] = 35/13 * y - 35/13 * y**2 + 10/13 * x - 10/13 * x**2
    val[..., 1] = -25/26 * (-1 + 2 * y) * (-1 + 2 * x)
    
    return val
def solution(points: TensorLike) -> TensorLike:
    x = points[..., 0]
    y = points[..., 1]
    
    val = bm.zeros(points.shape, dtype=points.dtype)
    val[..., 0] = x * (1 - x) * y * (1 - y)
    val[..., 1] = 0
    
    return val
def dirichlet(points: TensorLike) -> TensorLike:

    return solution(points)

extent = [0, 1, 0, 1, 0, 1]
# extent = [0, 2, 0, 2, 0, 2]
h = [1, 1, 1]
origin = [0, 0, 0]
mesh = UniformMesh3d(extent, h, origin)
mesh_text = HexahedronMesh.from_box(extent, nx=1, ny=1, nz=1)


# edge2ipoint = mesh.edge_to_ipoint(p=1)
# print("edge2ipoint:", edge2ipoint.shape, "\n", edge2ipoint)
# face2cell = mesh.face_to_cell()
# print("face2cell:", face2cell.shape, "\n", face2cell)
# face2edge = mesh.face_to_edge()
# print("face2edge:", face2edge.shape, "\n", face2edge)
print("node:", mesh.entity('node'))
print("face:", mesh.entity('face'))
print("cell:", mesh.entity('cell'))
print("face2cell:", mesh.face_to_cell().shape, "\n", mesh.face_to_cell())
face2ipoint = mesh.face_to_ipoint(p=2)
print("face2ipoint:", face2ipoint.shape, "\n", face2ipoint)
cell2ipoint = mesh.cell_to_ipoint(p=2)
print("cell2ipoint:", cell2ipoint.shape, "\n", cell2ipoint)
print("node_hex:", mesh_text.entity('node'))
print("face_hex:", mesh_text.entity('face'))
print("cell_hex:", mesh_text.entity('cell'))
face2ipoint_hex = mesh_text.face_to_ipoint(p=2)
print("face2ipoint_hex:", face2ipoint_hex.shape, "\n", face2ipoint_hex)
cell2ipoint_hex = mesh_text.cell_to_ipoint(p=2)
print("cell2ipoint_hex:", cell2ipoint_hex.shape, "\n", cell2ipoint_hex)
import matplotlib.pyplot as plt

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
# mesh.add_plot(axes)
# mesh.find_node(axes, showindex=True)
# mesh.find_edge(axes, showindex=True)
# mesh.find_face(axes, showindex=True)
# mesh.find_cell(axes, showindex=True)
mesh_text.add_plot(axes)
mesh_text.find_node(axes, showindex=True)
mesh_text.find_edge(axes, showindex=True)
mesh_text.find_face(axes, showindex=True)
mesh_text.find_cell(axes, showindex=True)

plt.show()
asd
maxit = 5
errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
import numpy as np
errorMatrix_old = np.zeros((2, maxit), dtype=np.float64)
errorMatrix_new_old = np.zeros((2, maxit), dtype=np.float64)
for i in range(maxit):
    p = 1
    space = LagrangeFESpace(mesh, p=p, ctype='C')
    tensor_space = TensorFunctionSpace(space, shape=(2, -1))

    # (tgdof, )
    uh_dependent = tensor_space.function()

    # 与单元有关的组装方法
    integrator_bi_dependent = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                            elasticity_type=None, q=5)
    
    cell2tldof = tensor_space.cell_to_dof()
    print("cell2ldof:", cell2tldof)
    # 与单元有关的组装方法
    KK_dependent = integrator_bi_dependent.assembly(space=tensor_space)
    print("KK_dependent:\n", KK_dependent[0])

    # 与单元有关的组装方法 
    bform_dependent = BilinearForm(tensor_space)
    bform_dependent.add_integrator(integrator_bi_dependent)
    K_dependent = bform_dependent.assembly()

    
    integrator_li = VectorSourceIntegrator(source=source, q=5)
    integrator_li_old = VectorSourceIntegrator_old(f=source, q=5)

    FF = integrator_li.assembly(space=tensor_space)
    FF_old = integrator_li_old.assembly_cell_vector(space=tensor_space_old)

    lform = LinearForm(tensor_space)
    lform.add_integrator(integrator_li)
    F = lform.assembly()
    lform_old = LinearForm_old(tensor_space_old)
    lform_old.add_domain_integrator(integrator_li_old)
    F_old = lform_old.assembly()

    dbc = DBC(space=tensor_space, gd=dirichlet, left=False)
    isDDof = tensor_space.is_boundary_dof(threshold=None)

    F = dbc.check_vector(F)
    uh_dependent = tensor_space.boundary_interpolate(gD=dirichlet, uh=uh_dependent)
    F_dependent = F - K_dependent.matmul(uh_dependent[:])
    F_dependent[isDDof] = uh_dependent[isDDof]
    
    K_dependent = dbc.check_matrix(K_dependent)
    kwargs = K_dependent.values_context()
    indices = K_dependent.indices()
    new_values = bm.copy(K_dependent.values())
    IDX = isDDof[indices[0, :]] | isDDof[indices[1, :]]
    new_values[IDX] = 0

    K_dependent = COOTensor(indices, new_values, K_dependent.sparse_shape)
    index, = bm.nonzero(isDDof, as_tuple=True)
    one_values = bm.ones(len(index), **kwargs)
    one_indices = bm.stack([index, index], axis=0)
    K1_dependent = COOTensor(one_indices, one_values, K_dependent.sparse_shape)
    K_dependent = K_dependent.add(K1_dependent).coalesce()
    K_dependent_dense = K_dependent.to_dense()

    dflag_old = space_old.boundary_interpolate(gD=dirichlet, uh=uh_old)
    K_old, F_old = DirichletBC_old(space=space_old, gD=dirichlet).apply(K_old, F_old)
    K_old_dense = K_old.toarray()

    errorMatrix_new_old[0, i] = np.max(np.abs(bm.to_numpy(K_dependent.to_dense()) - K_old.toarray()))
    errorMatrix_new_old[1, i] = np.max(np.abs(bm.to_numpy(F_dependent) - F_old))

    uh_dependent[:] = cg(K_dependent, F_dependent, maxiter=5000, atol=1e-14, rtol=1e-14)
    from scipy.sparse.linalg import spsolve
    uh_old.flat[:] = spsolve(K_old, F_old)

    u_exact = tensor_space.interpolate(solution)
    errorMatrix[0, i] = bm.max(bm.abs(bm.array(uh_dependent) - u_exact))
    errorMatrix[1, i] = bm.max(bm.abs(bm.array(uh_dependent) - u_exact)[isDDof])
    errorMatrix_old[0, i] = np.max(np.abs(uh_old.ravel() - u_exact))
    errorMatrix_old[1, i] = np.max(np.abs(uh_old.ravel() - u_exact)[dflag_old.flat])
    
    if i < maxit-1:
        mesh.uniform_refine()
        mesh_old.uniform_refine()

print("errorMatrix:\n", errorMatrix)
print("errorMatrix_old:\n", errorMatrix_old)
print("errorMatrix_new_old:\n", errorMatrix_new_old)
print("order:\n ", bm.log2(errorMatrix[0,:-1]/errorMatrix[0,1:]))
print("order_old:\n ", np.log2(errorMatrix_old[0,:-1]/errorMatrix_old[0,1:]))