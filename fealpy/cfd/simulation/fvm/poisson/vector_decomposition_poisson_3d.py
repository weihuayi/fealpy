from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')
from fealpy.mesh import TetrahedronMesh
from fealpy.model import PDEModelManager
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

pde = PDEModelManager('poisson').get_example('sinsinsin')
domain = pde.domain()
mesh = TetrahedronMesh.from_box(domain, nx=16, ny=16, nz=16)
cell2cell = mesh.cell_to_cell()
cell_centers = mesh.bc_to_point(bm.array([1/4, 1/4, 1/4, 1/4]))
face_centers = mesh.bc_to_point(bm.array([1/3, 1/3, 1/3]))
NC = mesh.number_of_cells()
cell2face = mesh.cell_to_face()    
def vector_decomposition(mesh):
    cell = mesh.entity("cell")
    node = mesh.entity("node")
    p = node[cell]
    local_faces = bm.array([
        [1, 2, 3],  # face 0
        [0, 3, 2],  # face 1
        [0, 1, 3],  # face 2
        [0, 2, 1],  # face 3
    ])
    face_coords = p[:, local_faces]
    vec1 = face_coords[:, :, 1] - face_coords[:, :, 0]  # (NC, 4, 3)
    vec2 = face_coords[:, :, 2] - face_coords[:, :, 0]  # (NC, 4, 3)
    Sf = 0.5 * bm.cross(vec1, vec2)  # 面的面积法向量，shape: (NC, 4, 3)
    e = bm.zeros((NC, 4, 3))
    for j in range(4):
            nbr_ids = cell2cell[:, j]
            mask = nbr_ids >= 0
            e[mask, j, :] = cell_centers[nbr_ids[mask]] - cell_centers[mask]
    f2c = mesh.face_to_cell()
    boundary_face = mesh.boundary_face_index()
    boundary_f2c = f2c[boundary_face]
    boundary_cellcenter = cell_centers[boundary_f2c[..., 0]]
    boundary_facecenter = face_centers[boundary_face]
    #e2为处理边界的向量，计算边界的单元到边界中点的向量
    e2 = boundary_facecenter - boundary_cellcenter
    cell_idx = boundary_f2c[:, 0]
    local_face_idx = boundary_f2c[:, 2]
    #e是内心连线的向量，形状为NC*3*2，在每个控制体中，起点为本控制体内心，若在边界则计算内心到控制体边界边中点连线的向量
    e[cell_idx, local_face_idx] = e2
    d = bm.linalg.norm(e, axis=-1, keepdims=True)
    numerator = bm.einsum('ijk,ijk->ij', Sf, Sf)[..., None]
    denominator = bm.einsum('ijk,ijk->ij', e, Sf)[..., None]
    Ef = (numerator / denominator) * e
    Ef_norm = bm.linalg.norm(Ef, axis=-1)[..., None]
    Tf = Sf - Ef
    flux_coeff = Ef_norm / d
    flux_coeff = flux_coeff.reshape((NC, 4))
    return flux_coeff, Sf, Ef, Tf

def matrix_assembly(mesh, flux_coeff):
    A = bm.zeros((NC, NC))
    # 初始右端项
    b0 = mesh.integral(pde.source, q=3, celltype=True).reshape((NC, 1))
    I = []  
    J = []  
    V = []  
    for i in range(4):
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
    V = mesh.entity_measure("cell")
    for it in range(max_iter):
        # 边中值
        uh_nb = uh1[cell2cell]           
        uh_f = 0.5 * (uh1[:, None] + uh_nb) 
        # 2. 平均梯度 grad[i] = sum_j uh_f[i,j] * Sf[i,j] / S[i]
        grad = bm.sum(uh_f[:, :, None] * Sf, axis=1) / V[:, None]
        # 3. 边上梯度 grad_f[i,j] = 0.5*(grad[i] + grad[nb])
        grad_nb = grad[cell2cell]  # (NC, 3, 2)
        grad_f = 0.5 * (grad[:, None, :] + grad_nb)  # (NC, 3, 2)
        # grad_f = bm.where(cell_mask[..., None], 0.0, grad_f)  # 边界交叉扩散为0
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
uI = pde.solution(cell_centers)
cell_areas = mesh.entity_measure("cell")
l2_error = bm.sqrt(bm.sum(cell_areas * (uI - uh)**2))
print(f"L2 error: {l2_error}")
# error = bm.max(bm.abs(uI-uh))
# print("error:",error)
