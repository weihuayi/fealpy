from fealpy.experimental.mesh import UniformMesh2d
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.experimental.fem import LinearElasticityIntegrator, BilinearForm
from fealpy.experimental.fem import DirichletBC as DBC

from fealpy.experimental.typing import TensorLike, Tuple
from builtins import float

from fealpy.experimental.sparse import COOTensor

from fealpy.experimental.solver import cg

from fealpy.experimental.backend import backend_manager as bm

# bm.set_backend('numpy')
# bm.set_backend('pytorch')
# bm.set_backend('jax')

def material_model_SIMP(rho: TensorLike, penal: float, E0: float, 
                            Emin: float=None) -> TensorLike:
    if Emin is None:
        E = rho ** penal * E0
    else:
        E = Emin + rho ** penal * (E0 - Emin)
    return E

def material_model_SIMP_derivative(rho: TensorLike, penal: float, E0: float, 
                                    Emin: float=None) -> TensorLike:
    if Emin is None:
        dE = -penal * rho ** (penal - 1) * E0
    else:
        dE = -penal * rho ** (penal - 1) * (E0 - Emin)
    return dE

# TODO 可以使用稀疏矩阵存储
def compute_filter(rmin: int) -> Tuple[TensorLike, TensorLike]:
    H = bm.zeros((NC, NC), dtype=bm.float64)

    # 确定哪些单元在滤波器半径范围内
    for i1 in range(nx):
        for j1 in range(ny):
            e1 = (i1) * ny + j1
            # 确定滤波器半径 rmin 的整数边界
            imin = max(i1 - (bm.ceil(rmin) - 1), 0.)
            imax = min(i1 + (bm.ceil(rmin)), nx)
            for i2 in range(int(imin), int(imax)):
                jmin = max(j1 - (bm.ceil(rmin) - 1), 0.)
                jmax = min(j1 + (bm.ceil(rmin)), ny)
                for j2 in range(int(jmin), int(jmax)):
                    e2 = i2 * ny + j2
                    H[e1, e2] = max(0., rmin - bm.sqrt((i1-i2)**2 + (j1-j2)**2))

    Hs = bm.sum(H, axis=1)

    return H, Hs

def apply_filter(ft, rho, dc, dv):

    if ft == 0:
        dc = bm.matmul(H, bm.multiply(rho, dc) / Hs / bm.maximum(1e-3, rho))
        dv = dv
    elif ft == 1:
        dc = bm.matmul(H, (dc / Hs))
        dv = bm.matmul(H, (dv / Hs))
    
    return dc, dv


# def check(nx, ny, rmin, rho, dc):

#     dcn_e = bm.zeros((nx, ny), dtype=bm.float64)

#     dc_e = bm.reshape(dc, (nx, ny))
#     rho_e = bm.reshape(rho, (nx, ny))

#     # 计算过滤器半径
#     r = int(rmin)

#     for i in range(nx):
#         for j in range(ny):
#             sum_val = 0.0
#             # 确定邻域的范围
#             min_x = max(i - r, 0)
#             max_x = min(i + r + 1, nx)
#             min_y = max(j - r, 0)
#             max_y = min(j + r + 1, ny)

#             for k in range(min_x, max_x):
#                 for l in range(min_y, max_y):

#                     # Calculate convolution operator value for the element (k,l) with respect to (i,j)
#                     fac = rmin - bm.sqrt((i - k)**2 + (j - l)**2)

#                     # Accumulate the convolution sum
#                     sum_val += max(0, fac)

#                     # 基于 convolution 算子的值修改单元的灵敏度
#                     dcn_e[j, i] += max(0, fac) * rho_e[l, k] * dc_e[l, k]

#             # Normalize the modified sensitivity for element (i, j)
#             dcn_e[j, i] /= (rho_e[j, i] * sum_val)
    
#     dcn = bm.reshape(bm.flip(dcn_e, axis=0), (nx, ny)).T.reshape(-1)

#     return dcn


# Optimality criterion
def oc(volfrac, nx, ny, rho, dc, dv, passive=None):
        # 初始化 Lagrange 乘子
        l1, l2 = 0, 1e5

        # 定义设计变量中允许的最大变化
        move = 0.2

        rho_new = bm.copy(rho)

        # 二分法以找到满足体积约束的 Lagrange 乘子
        while (l2 - l1) > 1e-4:
            # 计算当前 Lagrange 乘子区间的中点
            lmid = 0.5 * (l2 + l1)
            # Lower limit move restriction
            tmp0 = rho - move
            # Upper limit move restriction
            tmp1 = rho + move

            # Design variable update (intermediate step) using OC update scheme
            be = -dc / dv / lmid
            tmp2 = rho * bm.sqrt(be)
            tmp3 = bm.minimum(tmp1, tmp2)
            tmp4 = bm.minimum(1, tmp3) 
            tmp5 = bm.maximum(tmp0, tmp4)

            rho_new = bm.maximum(0.001, tmp5)

            # 寻找 Passive 单元，并将其密度设置为最小密度
            if passive is not None:
                rho_new[passive == 1] = 0.001

            # 检查当前设计是否满足体积约束
            if bm.sum(rho_new) - volfrac * nx * ny > 0:
                l1 = lmid
            else:
                l2 = lmid

        return rho_new


# Short Cantilever
def source(points: TensorLike) -> TensorLike:
    
    val = bm.zeros(points.shape, dtype=points.dtype)
    val[nx*(ny+1), 1] = -1
    
    return val

def dirichlet(points: TensorLike) -> TensorLike:

    return bm.zeros(points.shape, dtype=points.dtype)

def is_dirichlet_boundary_edge(points: TensorLike) -> TensorLike:
    """
    Determine which boundary edges satisfy the given property.

    Args:
        points (TensorLike): The coordinates of the points defining the edges.

    Returns:
        TensorLike: A boolean array indicating which boundary edges satisfy the property. 
        The length of the array is NBE, which represents the number of boundary edges.
    """

    temp = (points[:, 0] == 0.0)

    return temp


# Default input parameters
nx = 3
ny = 2
volfrac = 0.5
penal = 3.0
rmin = 1.5

extent = [0, nx, 0, ny]
h = [1, 1]
origin = [0, 0]
mesh = UniformMesh2d(extent, h, origin)
# import matplotlib.pyplot as plt
# fig = plt.figure()
# axes = fig.add_subplot(111)
# mesh.add_plot(axes)
# mesh.find_node(axes, showindex=True)
# mesh.find_cell(axes, showindex=True)
# plt.show()
NC = mesh.number_of_cells()
p = 1
space = LagrangeFESpace(mesh, p=p, ctype='C')
tensor_space = TensorFunctionSpace(space, shape=(-1, 2))

# Allocate design variables, initialize and allocate sens.
rho = volfrac * bm.ones(NC, dtype=bm.float64)

# element stiffness matrix
integrator_bi = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                        elasticity_type='stress', coef=None, q=5)
KE = integrator_bi.assembly(space=tensor_space)

# Filter
H, Hs = compute_filter(rmin)

import matplotlib.pyplot as plt
plt.ion()
fig, ax = plt.subplots()
image = ax.imshow(-rho.reshape(nx, ny).T, cmap='gray', vmin=-1, vmax=0)
ax.axis('off')

# Set loop counter and gradient vectors
loop = 0
change = 1
dv = bm.zeros(NC, dtype=bm.float64)
dc = bm.ones(NC, dtype=bm.float64)
c = bm.ones(NC, dtype=bm.float64)
while change > 0.01 and loop < 2000:
    loop += 1

    rho_old = bm.copy(rho)

    # Setup and solve FE problem
    uh = tensor_space.function()
    E = material_model_SIMP(rho=rho, penal=penal, E0=1.0, Emin=None)
    integrator_bi = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                            elasticity_type='stress', coef=E, q=5)
    KK = integrator_bi.assembly(space=tensor_space)
    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator_bi)
    K = bform.assembly()

    F = tensor_space.interpolate(source)
    
    dbc = DBC(space=tensor_space, gd=dirichlet, left=False)
    isDDof = tensor_space.is_boundary_dof(threshold=is_dirichlet_boundary_edge)

    F = dbc.check_vector(F)
    uh = tensor_space.boundary_interpolate(gD=dirichlet, uh=uh, threshold=is_dirichlet_boundary_edge)
    F = F - K.matmul(uh[:])
    F[isDDof] = uh[isDDof]
    
    K = dbc.check_matrix(K)
    kwargs = K.values_context()
    indices = K.indices()
    new_values = bm.copy(K.values())
    IDX = isDDof[indices[0, :]] | isDDof[indices[1, :]]
    new_values[IDX] = 0

    K = COOTensor(indices, new_values, K.sparse_shape)
    index, = bm.nonzero(isDDof)
    one_values = bm.ones(len(index), **kwargs)
    one_indices = bm.stack([index, index], axis=0)
    K1 = COOTensor(one_indices, one_values, K.sparse_shape)
    K = K.add(K1).coalesce()
    K_dense = K.to_dense()

    uh[:] = cg(K, F, maxiter=5000, atol=1e-14, rtol=1e-14)

    # Objective and sensitivity
    obj = 0
    cell2ldof = tensor_space.cell_to_dof()
    uhe = uh[cell2ldof]
    c[:] = bm.einsum('ci, cik, ck -> c', uhe, KE, uhe)
    obj += bm.einsum('c, c -> ', E, c)

    dE = material_model_SIMP_derivative(rho=rho, penal=penal, E0=1.0, Emin=None)
    dc[:] = bm.einsum('c, c -> c', dE, c)

    dv[:] = bm.ones(NC, dtype=bm.float64)

    # 灵敏度过滤
    dcn, dvn = apply_filter(ft=0, H=H, Hs=Hs, rho=rho, dc=dc, dv=dv)

    # Optimality criteria
    rho = oc(volfrac=volfrac, nx=nx, ny=ny, rho=rho, dc=dc, dv=dv)

    # Compute the change by the inf. norm
    change = bm.linalg.norm(rho.reshape(NC, 1) - rho_old.reshape(NC, 1), bm.inf)
    
    # Write iteration history to screen
    print("iter.: {0} , object.: {1:.3f} Volfrac.: {2:.3f}, change.: {3:.3f}".format(
        loop, obj, bm.sum(rho) / NC, change))

    # Plot
    image.set_data(-rho.reshape(nx, ny).T)
    plt.draw()
    plt.pause(1e-5)