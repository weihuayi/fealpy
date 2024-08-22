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

def check(rmin, rho, dce):
    dcn = bm.zeros((nx, ny), dtype=bm.float64).T
    rho = rho.reshape(nx, ny).T
    dce = dce.reshape(nx, ny).T
    # 计算过滤器半径
    r = int(rmin)

    for i in range(nx):
        for j in range(ny):
            sum_val = 0.0
            # 确定邻域的范围
            min_x = max(i - r, 0)
            max_x = min(i + r + 1, nx)
            min_y = max(j - r, 0)
            max_y = min(j + r + 1, ny)

            for k in range(min_x, max_x):
                for l in range(min_y, max_y):

                    # Calculate convolution operator value for the element (k,l) with respect to (i,j)
                    fac = rmin - bm.sqrt((i - k)**2 + (j - l)**2)

                    # Accumulate the convolution sum
                    sum_val += max(0, fac)

                    # 基于 convolution 算子的值修改单元的灵敏度
                    dcn[j, i] += max(0, fac) * rho[l, k] * dce[l, k]

            # Normalize the modified sensitivity for element (i,j)
            dcn[j, i] /= (rho[j, i] * sum_val)

    return dcn.reshape(-1)

# Optimality criterion
def oc(volfrac, rho, dce, dve, passive=None):
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
            be = -dce / dve / lmid
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


def source(points: TensorLike) -> TensorLike:
    
    val = bm.zeros(points.shape, dtype=points.dtype)
    val[nx*(ny+1), 1] = -1
    
    return val

def dirichlet(points: TensorLike) -> TensorLike:

    return bm.zeros(points.shape, dtype=points.dtype)

def is_dirichlet_boundary(points):
    return points[:, 0] == 0.0


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
import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.add_subplot(111)
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, color='r', fontsize=20)
plt.show()
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

# Set loop counter and gradient vectors
loop = 0
change = 1
dve = bm.zeros(NC, dtype=bm.float64)
dce = bm.ones(NC, dtype=bm.float64)
ce = bm.ones(NC, dtype=bm.float64)
while change > 0.01 and loop < 2000:
    loop += 1

    rho_old = bm.copy(rho)

    # Setup and solve FE problem
    uh = tensor_space.function()
    E = material_model_SIMP(rho=rho, penal=penal, E0=1.0, Emin=1e-9)
    integrator_bi = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                            elasticity_type='stress', coef=E, q=5)
    KK = integrator_bi.assembly(space=tensor_space)
    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator_bi)
    K = bform.assembly()

    F = tensor_space.interpolate(source)
    
    dbc = DBC(space=tensor_space, gd=dirichlet, left=False)
    isDDof = tensor_space.is_boundary_dof(threshold=is_dirichlet_boundary)
    isDDofs = tensor_space.is_boundary_dof(threshold=None)

    F = dbc.check_vector(F)
    uh = tensor_space.boundary_interpolate(gD=dirichlet, uh=uh, threshold=is_dirichlet_boundary)
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
    c = 0
    cell2ldof = tensor_space.cell_to_dof()
    uhe = uh[cell2ldof]
    ce[:] = bm.einsum('ci, cik, ck -> c', uhe, KE, uhe)
    c += bm.einsum('c, c -> ', E, ce)

    dE = material_model_SIMP_derivative(rho=rho, penal=penal, E0=1.0, Emin=None)
    dce[:] = bm.einsum('c, c -> c', dE, ce)

    dve[:] = bm.ones(NC, dtype=bm.float64)

    # 灵敏度过滤
    dce = check(rmin=rmin, rho=rho, dce=dce)

    # Optimality criteria
    # TODO 更新后的设计变量不对

    rho = oc(volfrac=volfrac, rho=rho, dce=dce, dve=dve)

    # Compute the change by the inf. norm
    change = bm.linalg.norm(rho.reshape(NC, 1) - rho_old.reshape(NC, 1), bm.inf)
    
    # Write iteration history to screen
    print("it.: {0} , c.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(
        loop, c, bm.sum(rho) / NC, change))
