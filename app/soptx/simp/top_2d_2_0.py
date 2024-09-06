from fealpy.experimental.mesh import UniformMesh2d
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.experimental.fem import LinearElasticityIntegrator, BilinearForm
from fealpy.experimental.fem import DirichletBC as DBC

from fealpy.experimental.typing import TensorLike, Tuple
from builtins import float

from fealpy.experimental.sparse import COOTensor

from fealpy.experimental.solver import cg

from fealpy.experimental.backend import backend_manager as bm

from math import ceil, sqrt


# bm.set_backend('numpy')
bm.set_backend('pytorch')
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

def compute_filter(rmin: int) -> Tuple[TensorLike, TensorLike]:
    nfilter = int(nx * ny * ((2 * (ceil(rmin) - 1) + 1) ** 2))
    iH = bm.zeros(nfilter)
    jH = bm.zeros(nfilter)
    sH = bm.zeros(nfilter)
    cc = 0
    for i in range(nx):
        for j in range(ny):
            row = i * ny + j
            kk1 = int(max(i - (ceil(rmin) - 1), 0))
            kk2 = int(min(i + ceil(rmin), nx))
            ll1 = int(max(j - (ceil(rmin) - 1), 0))
            ll2 = int(min(j + ceil(rmin), ny))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k * ny + l
                    fac = rmin - sqrt(((i-k) * (i-k) + (j-l) * (j-l)))
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = max(0.0, fac)
                    cc = cc + 1
    # Finalize assembly and convert to csc format
    H = COOTensor(indices=bm.stack((iH, jH), axis=0), values=sH, spshape=(nx*ny, nx*ny)).tocsr()
    Hs = H @ bm.ones(H.shape[1])
    # TODO 目前 COOTensor 没有实现 sum 方法
    # Hs = bm.sum(H, axis=1)

    return H, Hs

# Optimality criterion
def oc(nx, ny, rho, dce, dve, g):
	l1 = 0
	l2 = 1e9
	move = 0.2
	# reshape to perform vector operations
	rho_new = bm.zeros(nx*ny, dtype=bm.float64)
	while (l2 - l1) / (l2 + l1)> 1e-3:
		lmid = 0.5 * (l2 + l1)
		rho_new[:] = bm.maximum(0.0, bm.maximum(rho-move, 
                                           bm.minimum(1.0, 
                                                    bm.minimum(rho+move, rho*bm.sqrt(-dce/dve/lmid)))))
        # 检查当前设计是否满足体积约束
		gt = g + bm.sum((dve * (rho_new - rho)))
		if gt > 0 :
			l1 = lmid
		else:
			l2 = lmid
               
	return rho_new, gt


# Half-MBB 
def source(points: TensorLike) -> TensorLike:
    
    val = bm.zeros(points.shape, dtype=points.dtype)
    val[ny, 1] = -1
    
    return val

def dirichlet(points: TensorLike) -> TensorLike:

    return bm.zeros(points.shape, dtype=points.dtype)

# def is_dirichlet_boundary(points: TensorLike) -> TensorLike:
#     """
#     Determine which boundary edges satisfy the given property.

#     Args:
#         points (TensorLike): The coordinates of the points defining the edges.

#     Returns:
#         TensorLike: A boolean array indicating which boundary edges satisfy the property. 
#         The length of the array is NBE, which represents the number of boundary edges.
#     """
#     temp1 = (points[:, 0] == 0.0)
#     temp2 = (bm.arange(len(points)) % 2 == 0)
#     temp = temp1 & temp2

#     return temp

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
nx = 160
ny = 80
volfrac = 0.5
penal = 3.0
rmin = 1.5
ft = 0 # ft==0 -> sens, ft==1 -> dens


extent = [0, nx, 0, ny]
h = [1, 1]
origin = [0, 0]
mesh = UniformMesh2d(extent, h, origin)
NC = mesh.number_of_cells()
p = 1
space = LagrangeFESpace(mesh, p=p, ctype='C')
tensor_space = TensorFunctionSpace(space, shape=(-1, 2))

# Allocate design variables, initialize and allocate sens.
rho = volfrac * bm.ones(NC, dtype=bm.float64)
rho_old = bm.copy(rho)
rho_Phys = bm.copy(rho_old)
g = 0 # must be initialized to use the NGuyen/Paulino OC approach

# element stiffness matrix
integrator_bi = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                        elasticity_type='stress', coef=None, q=5)
KE = integrator_bi.assembly(space=tensor_space)

# Filter
H, Hs = compute_filter(rmin)

# Set loop counter and gradient vectors
loop = 0
change = 1
dve = bm.zeros(NC, dtype=bm.float64)
dce = bm.ones(NC, dtype=bm.float64)
ce = bm.ones(NC, dtype=bm.float64)
while change > 0.01 and loop < 2000:
    loop += 1

    # Setup and solve FE problem
    uh = tensor_space.function()
    E = material_model_SIMP(rho=rho_Phys, penal=penal, E0=1.0, Emin=1e-9)
    integrator_bi = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                            elasticity_type='stress', coef=E, q=5)
    KK = integrator_bi.assembly(space=tensor_space)
    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator_bi)
    K = bform.assembly()

    F = tensor_space.interpolate(source)
    
    dbc = DBC(space=tensor_space, gd=dirichlet, left=False)
    isDDof = tensor_space.is_boundary_dof(threshold=is_dirichlet_boundary_edge)
    isDDof[1::2] = False

    F = dbc.check_vector(F)
    uh = tensor_space.boundary_interpolate(gD=dirichlet, uh=uh, 
                                           threshold=is_dirichlet_boundary_edge)
    uh[1::2] = 0
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

    dE = material_model_SIMP_derivative(rho=rho_Phys, penal=penal, E0=1.0, Emin=1e-9)
    dce[:] = bm.einsum('c, c -> c', dE, ce)

    dve[:] = bm.ones(NC, dtype=bm.float64)

    # Sensitivity filtering
    # asarray 函数的作用是将输入数据转换为一个数组,它的一个重要特性是: 
    # 如果输入已经是一个数组, 它不会复制数据, 而是返回输入的原始数组.
    # 这种行为可以节省内存和提高效率, 特别是在处理大数据时.
    if ft == 0:
        dce[:] = bm.asarray(H * (rho * dce)[bm.newaxis].T / Hs)[:, 0] / bm.maximum(0.001, rho)
    elif ft == 1:
        dce[:] = bm.asarray(H * (dce[bm.newaxis].T / Hs))[:, 0]
        dve[:] = bm.asarray(H * (dve[bm.newaxis].T / Hs))[:, 0]

    # Optimality criteria
    rho_old[:] = rho
    rho, g = oc(rho=rho, dce=dce, dve=dve, g=g)

    # Filter design variables
    if ft == 0:
        rho_Phys[:] = rho
    elif ft == 1:
        rho_Phys[:] = bm.asarray(H * rho[bm.newaxis].T / Hs)[:, 0]

    # Compute the change by the inf. norm
    change = bm.linalg.norm(rho.reshape(NC, 1) - rho_old.reshape(NC, 1), bm.inf)
    
    # Write iteration history to screen
    print("it.: {0} , c.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(
        loop, c, (g + volfrac * NC) / NC, change))
