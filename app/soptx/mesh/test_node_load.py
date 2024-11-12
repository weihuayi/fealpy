from fealpy.backend import backend_manager as bm

from fealpy.mesh import HexahedronMesh
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator

from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.linear_form import LinearForm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.sparse import COOTensor, CSRTensor


bm.set_backend('numpy')
nx, ny, nz = 3, 3, 3 
mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=nx, ny=ny, nz=nz)
NC = mesh.number_of_cells()
cm = mesh.entity_measure('cell')
# import matplotlib.pyplot as plt
# fig = plt.figure()
# axes = fig.add_subplot(111, projection='3d')
# mesh.add_plot(axes)
# mesh.find_node(axes, showindex=True)
# mesh.find_cell(axes, showindex=True)
# plt.show()

cell = mesh.entity("cell")
cell_indices = [10, 13, 16]
cell_target = cell[cell_indices, :]

node = mesh.entity("node")

# 载荷点 a 的坐标
a = bm.array([0.5, 0.16666667, 0.16666667], dtype=bm.float64)

# 载荷点 b 的坐标
b = bm.array([0.5, 0.5, 0.16666667], dtype=bm.float64)

# 载荷点 c 的坐标
c = bm.array([0.5, 0.83333333, 0.16666667], dtype=bm.float64)

# 如果您想将三个点放在一个数组中
load_points = bm.array([a, b, c], dtype=bm.float64)

load_values = bm.array([2.0, 3.0, 4.0], dtype=bm.float64)  # 对应的载荷大小

def physical_to_local_coords(point, cell_coords):
    """
    将物理坐标转换为局部坐标（参考单元坐标）。

    参数：
    - point: ndarray，形状为 (3,)，待转换的物理坐标 (x, y, z)。
    - cell_coords: ndarray，形状为 (8, 3)，单元的8个节点的物理坐标。

    返回：
    - xi, eta, zeta: float，局部坐标。

    注意：
    该函数假定单元是线性的八节点六面体单元，参考单元的局部坐标范围为 [0, 1]。
    """
    # 计算单元在各个方向的最小值和最大值
    x_min, y_min, z_min = cell_coords.min(axis=0)
    x_max, y_max, z_max = cell_coords.max(axis=0)

    # 计算局部坐标 ξ, η, ζ
    xi = (point[0] - x_min) / (x_max - x_min)
    eta = (point[1] - y_min) / (y_max - y_min)
    zeta = (point[2] - z_min) / (z_max - z_min)

    return xi, eta, zeta

cell_coords_a = node[cell[9]]  
cell_coords_b = node[cell[12]]  
cell_coords_c = node[cell[15]]  
# 计算局部坐标
xi_a, eta_a, zeta_a = physical_to_local_coords(a, cell_coords_a)
xi_b, eta_b, zeta_b = physical_to_local_coords(b, cell_coords_b)
xi_c, eta_c, zeta_c = physical_to_local_coords(c, cell_coords_c)    

q = 1
qf = mesh.quadrature_formula(q)
bcs_a = (bm.tensor([[xi_a, 1-xi_a]]), bm.tensor([[eta_a, 1-eta_a]]), bm.tensor([[zeta_a, 1-zeta_a]]))
bcs_b = (bm.tensor([[xi_b, 1-xi_b]]), bm.tensor([[eta_b, 1-eta_b]]), bm.tensor([[zeta_b, 1-zeta_b]]))
bcs_c = (bm.tensor([[xi_c, 1-xi_c]]), bm.tensor([[eta_c, 1-eta_c]]), bm.tensor([[zeta_c, 1-zeta_c]]))

# bcs, ws = qf.get_quadrature_points_and_weights() # bcs ((1, 2), (1, 2), (1, 2)), ws (NQ, )

space = LagrangeFESpace(mesh, p=1, ctype='C')
tensor_space_dof = TensorFunctionSpace(space, shape=(3, -1))
tgdof = tensor_space_dof.number_of_global_dofs()
tldof = tensor_space_dof.number_of_local_dofs()
GD = 3
cell2dof_dof = tensor_space_dof.cell_to_dof() # (NC, tldof)

phi_a_dof = tensor_space_dof.basis(bcs_a) # (1, 1, tldof, GD)
phi_b_dof = tensor_space_dof.basis(bcs_b) # (1, 1, tldof, GD)
phi_c_dof = tensor_space_dof.basis(bcs_c) # (1, 1, tldof, GD)

FE_a_dof = load_values[0] * phi_a_dof # (1, 1, tldof, GD)
FE_b_dof = load_values[1] * phi_b_dof # (1, 1, tldof, GD)
FE_c_dof = load_values[2] * phi_c_dof # (1, 1, tldof, GD)

FE_a_dof = bm.einsum("qcid -> qci", FE_a_dof) # (1, 1, tldof)
FE_b_dof = bm.einsum("qcid -> qci", FE_b_dof) # (1, 1, tldof)
FE_c_dof = bm.einsum("qcid -> qci", FE_c_dof) # (1, 1, tldof)

FE_dof = bm.concatenate([FE_a_dof, FE_b_dof, FE_c_dof], axis=1) # (1, 3, tldof)
FE_all_dof = bm.zeros((1, NC, tldof), dtype=bm.float64)
FE_all_dof[0, cell_indices, :] = FE_dof[0, :, :] # (1, NC, tldof)

F_dof = COOTensor(
            indices = bm.empty((1, 0), dtype=bm.int32, device=bm.get_device(space)),
            values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(space)),
            spshape = (tgdof, ))
indices = cell2dof_dof.reshape(1, -1)
test = FE_all_dof.reshape(-1)
F_dof = F_dof.add(COOTensor(indices, FE_all_dof.reshape(-1), (tgdof, ))).to_dense()

def is_dirichlet_boundary_dof(points: TensorLike) -> TensorLike:
        domain = [0, 1, 0, 1, 0, 1]
        eps = 1e-12
        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < eps
        
        return coord

isDDof_dof = tensor_space_dof.is_boundary_dof(threshold=is_dirichlet_boundary_dof, method='interp')

print('---------------------------------------------')

