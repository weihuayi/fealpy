from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
bm.set_backend('numpy')
class PDE:
    def solution(self, p):
        """解析解 u = sin(pi*x)*sin(pi*y)"""
        x, y = p[..., 0], p[..., 1]
        return bm.sin(bm.pi * x) * bm.sin(bm.pi * y)

    def source(self, p):
        """源项 f = 2*pi^2*sin(pi*x)*sin(pi*y)"""
        x, y = p[..., 0], p[..., 1]
        return 2 * bm.pi**2 * bm.sin(bm.pi * x) * bm.sin(bm.pi * y)

pde = PDE()

# 网格生成
mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=2)
cell2cell = mesh.cell_to_cell()
num_cells = mesh.number_of_cells()
cell_centers = mesh.bc_to_point(bm.array([1/3, 1/3, 1/3]))
edge_centers = mesh.bc_to_point(bm.array([0.5, 0.5]))
NC = mesh.number_of_cells()
def vector_decomposition(mesh):
    #计算网格边向量edge_vec和法向向量en
    cell = mesh.entity("cell")
    node = mesh.entity("node")
    p = node[cell]
    edge_vec = bm.stack([p[:, 2] - p[:, 1], 
                     p[:, 0] - p[:, 2], 
                     p[:, 1] - p[:, 0]], axis=1)
    x = bm.array([[0, -1], [1, 0]])
    edge_vec_unit = edge_vec / bm.linalg.norm(edge_vec, axis=-1, keepdims=True)
    en = edge_vec_unit @ x
    #edge_vec_norm为边向量的模长，即每条边的边长，NC*3
    edge_vec_norm = bm.linalg.norm(edge_vec, axis=-1, keepdims=True)
    #Sf为带模长的边法向量，模长为边的长度
    Sf = bm.einsum('ijl,ijk->ijk', edge_vec_norm, en)
    #计算内心连线向量和边中点连线向量
    e = bm.zeros((NC, 3, 2))
    for j in range(3):
        nbr_ids = cell2cell[:, j]
        mask = nbr_ids >= 0
        e[mask, j, :] = cell_centers[nbr_ids[mask]] - cell_centers[mask]
    edge_middle_point = mesh.bc_to_point(bm.array([0.5, 0.5]))
    e2c = mesh.edge_to_cell()
    boundary_edge = mesh.boundary_face_index()
    boundary_e2c = e2c[boundary_edge]
    boundary_meshcenter = cell_centers[boundary_e2c[..., 0]]
    boundary_edge_middle_point = edge_middle_point[boundary_edge]
    #e2为处理边界的向量，计算边界的单元到边界中点的向量
    e2 = boundary_edge_middle_point - boundary_meshcenter
    cell_idx = boundary_e2c[:, 0]
    local_edge_idx = boundary_e2c[:, 2]
    #e是内心连线的向量，形状为NC*3*2，在每个控制体中，起点为本控制体内心，若在边界则计算内心到控制体边界边中点连线的向量
    e[cell_idx, local_edge_idx] = e2
    #计算向量分解Sf = Ef + Tf ,计算Ef和距离之比
    d = bm.linalg.norm(e, axis=-1, keepdims=True)
    numerator = bm.einsum('ijk,ijk->ij', Sf, Sf)[..., None]
    denominator = bm.einsum('ijk,ijk->ij', e, Sf)[..., None]
    Ef = (numerator / denominator) * e
    Ef_norm = bm.linalg.norm(Ef, axis=-1)[..., None]
    Tf = Sf - Ef
    flux_coeff = Ef_norm / d
    flux_coeff = flux_coeff.reshape((NC, 3))
    return flux_coeff, Sf, Ef, Tf

def matrix_assembly(mesh, flux_coeff):
    A = bm.zeros((NC, NC))
    # 初始右端项
    b0 = mesh.integral(pde.source, q=3, celltype=True).reshape((NC, 1))
    I = []  
    J = []  
    V = []  
    for i in range(3):
        src = bm.arange(NC)
        tgt = cell2cell[:, i]
        cval = -flux_coeff[:, i]
        mask = (tgt != src)  # 非对角项（处理非边界或边界有邻居的）
        I.extend(src[mask])
        J.extend(tgt[mask])
        V.extend(cval[mask])
    diag = bm.sum(flux_coeff, axis=1)
    I.extend(bm.arange(NC))
    J.extend(bm.arange(NC))
    V.extend(diag)
    A = coo_matrix((V, (I, J)), shape=(NC, NC))
    return A, b0

def iterative_solution(mesh, A, b0, Tf, Sf, max_iter=1, tol=1e-6):
    """迭代求解非正交修正的数值解"""
    uh1 = bm.zeros((NC))  
    S = mesh.entity_measure("cell")
    for it in range(max_iter):
        # 边中值
        uh_nb = uh1[cell2cell]           
        uh_f = 0.5 * (uh1[:, None] + uh_nb) 
        # 2. 平均梯度 grad[i] = sum_j uh_f[i,j] * Sf[i,j] / S[i]
        grad = bm.sum(uh_f[:, :, None] * Sf, axis=1) / S[:, None]
        # 3. 边上梯度 grad_f[i,j] = 0.5*(grad[i] + grad[nb])
        grad_nb = grad[cell2cell]  # (NC, 3, 2)
        cell_mask = (cell2cell == bm.arange(NC)[:, None])  # (NC, 3)
        grad_f = 0.5 * (grad[:, None, :] + grad_nb)  # (NC, 3, 2)

        grad_f = bm.where(cell_mask[..., None], 0.0, grad_f)  # 边界交叉扩散为0
        Cross_diffusion = bm.einsum('ijk,ijk->i', Tf, grad_f)[..., None]
        b = b0 + Cross_diffusion
        uh2 = spsolve(A, b)
        # 收敛性检查
        error = bm.max(bm.abs(uh2 - uh1))
        print(f"Iteration {it+1}, error: {error}")
        if error < tol:
            print(f"Converged after {it+1} iterations, error: {error}")
            break
        uh1 = uh2
    return uh2  

flux_coeff, Sf, Ef, Tf = vector_decomposition(mesh)
A, b0 = matrix_assembly(mesh, flux_coeff)
uh = iterative_solution(mesh, A, b0, Tf, Sf, max_iter=6, tol=1e-6)

# 计算误差
print(f"Number of cells: {num_cells}")
uI = pde.solution(cell_centers)
# cell_areas = mesh.entity_measure("cell")
# l2_error = bm.sqrt(bm.sum(cell_areas * (uI - uh)**2))
# print(f"L2 error: {l2_error}")
e = bm.max(bm.abs(uI-uh))
print("error:",e)