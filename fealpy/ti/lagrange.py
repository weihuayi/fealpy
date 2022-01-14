import numpy as np
import taichi as ti

def multi_index_matrix0d(p):
    multiIndex = 1
    return multiIndex 

def multi_index_matrix1d(p):
    ldof = p+1
    multiIndex = np.zeros((ldof, 2), dtype=np.int_)
    multiIndex[:, 0] = np.arange(p, -1, -1)
    multiIndex[:, 1] = p - multiIndex[:, 0]
    return multiIndex

def multi_index_matrix2d(p):
    ldof = (p+1)*(p+2)//2
    idx = np.arange(0, ldof)
    idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
    multiIndex = np.zeros((ldof, 3), dtype=np.int_)
    multiIndex[:,2] = idx - idx0*(idx0 + 1)/2
    multiIndex[:,1] = idx0 - multiIndex[:,2]
    multiIndex[:,0] = p - multiIndex[:, 1] - multiIndex[:, 2]
    return multiIndex

def multi_index_matrix3d(p):
    ldof = (p+1)*(p+2)*(p+3)//6
    idx = np.arange(1, ldof)
    idx0 = (3*idx + np.sqrt(81*idx*idx - 1/3)/3)**(1/3)
    idx0 = np.floor(idx0 + 1/idx0/3 - 1 + 1e-4) # a+b+c
    idx1 = idx - idx0*(idx0 + 1)*(idx0 + 2)/6
    idx2 = np.floor((-1 + np.sqrt(1 + 8*idx1))/2) # b+c
    multiIndex = np.zeros((ldof, 4), dtype=np.int_)
    multiIndex[1:, 3] = idx1 - idx2*(idx2 + 1)/2
    multiIndex[1:, 2] = idx2 - multiIndex[1:, 3]
    multiIndex[1:, 1] = idx0 - idx2
    multiIndex[:, 0] = p - np.sum(multiIndex[:, 1:], axis=1)
    return multiIndex

multi_index_matrix = [multi_index_matrix0d, multi_index_matrix1d, multi_index_matrix2d, multi_index_matrix3d]


def lagrange_shape_function(bc, p):
    """

    Notes
    -----
    
    计算形状为 (..., TD+1) 的重心坐标数组 bc 中的每一个重心坐标处的 p 次
    Lagrange 形函数值, 以及关于 TD+1 个重心坐标处的 1 阶导数值.

    """
    TD = bc.shape[-1] - 1
    multiIndex = multi_index_matrix[TD](p) 
    ldof = multiIndex.shape[0] # p 次 Lagrange 形函数的个数 

    c = np.arange(1, p+1, dtype=np.int_)
    P = 1.0/np.multiply.accumulate(c)
    t = np.arange(0, p)
    shape = bc.shape[:-1]+(p+1, TD+1) # (NQ, p+1, TD+1)
    A = np.ones(shape, dtype=bc.dtype)
    A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)

    FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
    FF[..., range(p), range(p)] = p
    np.cumprod(FF, axis=-2, out=FF)
    F = np.zeros(shape, dtype=bc.dtype)
    F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
    F[..., 1:, :] *= P.reshape(-1, 1)

    np.cumprod(A, axis=-2, out=A)
    A[..., 1:, :] *= P.reshape(-1, 1)

    idx = np.arange(TD+1)
    Q = A[..., multiIndex, idx]
    M = F[..., multiIndex, idx]

    shape = bc.shape[:-1]+(ldof, TD+1) # (NQ, ldof, TD+1)
    R1 = np.zeros(shape, dtype=bc.dtype)
    for i in range(TD+1):
        idx = list(range(TD+1))
        idx.remove(i)
        R1[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)

    R0 = np.prod(Q, axis=-1)
    return R0, R1 


@ti.kernel
def tri_lagrange_cell_mass_matrix_1(
        node : ti.template(), 
        cell : ti.template(),
        S : ti.template()):
    """
    Note:
        三角形网格上的线性拉格朗日有限元空间, 组装**单元质量矩阵**
    """
    c0 = 1.0/12.0
    c1 = 1.0/24.0
    for c in range(cell.shape[0]):
        x0 = node[cell[c, 0], 0]
        y0 = node[cell[c, 0], 1]

        x1 = node[cell[c, 1], 0]
        y1 = node[cell[c, 1], 1]

        x2 = node[cell[c, 2], 0]
        y2 = node[cell[c, 2], 1]

        l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 

@ti.kernel
def tri_lagrange_cell_stiff_matrix_1(
        node : ti.template(), 
        cell : ti.template(),
        S : ti.template()):
    """
    Note:
        三角形网格上的 1 次拉格朗日有限元空间, 组装**单元刚度矩阵**
    """
    for c in range(cell.shape[0]):
        x0 = node[cell[c, 0], 0]
        y0 = node[cell[c, 0], 1]

        x1 = node[cell[c, 1], 0]
        y1 = node[cell[c, 1], 1]

        x2 = node[cell[c, 2], 0]
        y2 = node[cell[c, 2], 1]

        l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 
        gphi = ti.Matrix([
            [(y1 - y2)/l, (x2 - x1)/l], 
            [(y2 - y0)/l, (x0 - x2)/l],
            [(y0 - y1)/l, (x1 - x0)/l]])
        l *= 0.5
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                S[c, i, j] = l*(gphi[i, 0]*gphi[j, 0] + gphi[i, 1]*gphi[j, 1])

@ti.kernel
def tri_lagrange_cell_mass_matrix_p(
        node : ti.template(), 
        cell : ti.template(),
        S : ti.template()):
    """
    Note:
        三角形网格上的 p 次拉格朗日有限元空间, 组装**单元质量矩阵**
    """
    for c in range(cell.shape[0]):
        x0 = node[cell[c, 0], 0]
        y0 = node[cell[c, 0], 1]

        x1 = node[cell[c, 1], 0]
        y1 = node[cell[c, 1], 1]

        x2 = node[cell[c, 2], 0]
        y2 = node[cell[c, 2], 1]

        l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 

@ti.kernel
def tri_lagrange_cell_stiff_matrix_p(
        node : ti.template(),  # (NN, 2)
        cell : ti.template(),  # (NC, 3)
        R : ti.template(), # (NQ, ldof, 3)
        ws : ti.template(), # (NQ, )
        S : ti.template() # (NC, ldof, ldof)
        ):

    """
    Note:
        三角形网格上的 p 次拉格朗日有限元空间, 组装**单元刚度矩阵**
    """

    NC = cell.shape[0]
    NQ = R.shape[0]
    ldof = R.shape[1]
    for c in range(NC):
        x0 = node[cell[c, 0], 0]
        y0 = node[cell[c, 0], 1]

        x1 = node[cell[c, 1], 0]
        y1 = node[cell[c, 1], 1]

        x2 = node[cell[c, 2], 0]
        y2 = node[cell[c, 2], 1]

        l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 
        gphi = ti.Matrix([
            [(y1 - y2)/l, (x2 - x1)/l], 
            [(y2 - y0)/l, (x0 - x2)/l],
            [(y0 - y1)/l, (x1 - x0)/l]])
        l *= 0.5
        for i in range(ldof):
            for j in range(ldof):
                S[c, i, j] = 0.0
                for q in range(NQ):
                    vix = R[q, i, 0]*gphi[0, 0] + R[q, i, 1]*gphi[1, 0] + R[q, i, 2]*gphi[2, 0]
                    viy = R[q, i, 0]*gphi[0, 1] + R[q, i, 1]*gphi[1, 1] + R[q, i, 2]*gphi[2, 1]
                    vjx = R[q, j, 0]*gphi[0, 0] + R[q, j, 1]*gphi[1, 0] + R[q, j, 2]*gphi[2, 0]
                    vjy = R[q, j, 0]*gphi[0, 1] + R[q, j, 1]*gphi[1, 1] + R[q, j, 2]*gphi[2, 1]
                    S[c, i, j] += ws[q]*(vix*vjx + viy*vjy)

                S[c, i, j] *= l

@ti.kernel
def tet_lagrange_cell_mass_matrix_1(
        node : ti.template(), # (NN, 3)
        cell : ti.template(), # (NC, 4)
        S : ti.template()): # (NC,4,, 4)
    """
    Note:
        四面体网格上的线性拉格朗日有限元空间, 组装**单元质量矩阵**
    """
    NC = cell.shape[0]

    for c in range(NC):
        m = ti.Matrix([
            [node[cell[c, 1], 0] - node[cell[c, 0], 0], node[cell[c, 1], 1] - node[cell[c, 0], 1], node[cell[c, 1], 2] - node[cell[c, 0], 2]],
            [node[cell[c, 2], 0] - node[cell[c, 0], 0], node[cell[c, 2], 1] - node[cell[c, 0], 1], node[cell[c, 2], 2] - node[cell[c, 0], 2]],
            [node[cell[c, 3], 0] - node[cell[c, 0], 0], node[cell[c, 3], 1] - node[cell[c, 0], 1], node[cell[c, 3], 2] - node[cell[c, 0], 2]]
            )
        l = m.determinant()/6.0
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                S[c, i, j] = l*(gphi[i, 0]*gphi[j, 0] + gphi[i, 1]*gphi[j, 1])

@ti.kernel
def tet_lagrange_cell_stiff_matrix_1(
        node : ti.template(), # (NN, 3)
        cell : ti.template(), # (NC, 4)
        S : ti.template()): # (NC, ldof, ldof)
    """
    Note:
        四面体网格上的线性拉格朗日有限元空间, 组装单元刚度矩阵
    """
    for c in range(cell.shape[0]):
        x0 = node[cell[c, 0], 0]
        y0 = node[cell[c, 0], 1]

        x1 = node[cell[c, 1], 0]
        y1 = node[cell[c, 1], 1]

        x2 = node[cell[c, 2], 0]
        y2 = node[cell[c, 2], 1]

        l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 
        gphi = ti.Matrix([
            [(y1 - y2)/l, (x2 - x1)/l], 
            [(y2 - y0)/l, (x0 - x2)/l],
            [(y0 - y1)/l, (x1 - x0)/l]])
        l *= 0.5
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                S[c, i, j] = l*(gphi[i, 0]*gphi[j, 0] + gphi[i, 1]*gphi[j, 1])

@ti.kernel
def tet_lagrange_cell_mass_matrix_p(
        node : ti.template(),  # (NN, 3)
        cell : ti.template(),  # (NC, 4)
        R : ti.template(), # (NQ, ldof, 4)
        ws : ti.template(), # (NQ, )
        S : ti.template() # (NC, ldof, ldof)
        ):
    """
    Note:
        四面体网格上的 p 次拉格朗日元, 组装单元刚度矩阵
    """
    NC = cell.shape[0]
    NQ = R.shape[0]
    ldof = R.shape[1]
    for c in range(NC):
        x0 = node[cell[c, 0], 0]
        y0 = node[cell[c, 0], 1]

        x1 = node[cell[c, 1], 0]
        y1 = node[cell[c, 1], 1]

        x2 = node[cell[c, 2], 0]
        y2 = node[cell[c, 2], 1]

        l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 
        gphi = ti.Matrix([
            [(y1 - y2)/l, (x2 - x1)/l], 
            [(y2 - y0)/l, (x0 - x2)/l],
            [(y0 - y1)/l, (x1 - x0)/l]])
        l *= 0.5
        for i in range(ldof):
            for j in range(ldof):
                S[c, i, j] = 0.0
                for q in range(NQ):
                    vix = R[q, i, 0]*gphi[0, 0] + R[q, i, 1]*gphi[1, 0] + R[q, i, 2]*gphi[2, 0]
                    viy = R[q, i, 0]*gphi[0, 1] + R[q, i, 1]*gphi[1, 1] + R[q, i, 2]*gphi[2, 1]
                    vjx = R[q, j, 0]*gphi[0, 0] + R[q, j, 1]*gphi[1, 0] + R[q, j, 2]*gphi[2, 0]
                    vjy = R[q, j, 0]*gphi[0, 1] + R[q, j, 1]*gphi[1, 1] + R[q, j, 2]*gphi[2, 1]
                    S[c, i, j] += ws[q]*(vix*vjx + viy*vjy)

                S[c, i, j] *= l
@ti.kernel
def tet_lagrange_cell_stiff_matrix_p(
        node : ti.template(),  # (NN, 3)
        cell : ti.template(),  # (NC, 4)
        R : ti.template(), # (NQ, ldof, 4)
        ws : ti.template(), # (NQ, )
        S : ti.template() # (NC, ldof, ldof)
        ):
    """
    Note:
        四面体网格上的 p 次拉格朗日元, 组装单元刚度矩阵
    """
    NC = cell.shape[0]
    NQ = R.shape[0]
    ldof = R.shape[1]
    for c in range(NC):
        x0 = node[cell[c, 0], 0]
        y0 = node[cell[c, 0], 1]

        x1 = node[cell[c, 1], 0]
        y1 = node[cell[c, 1], 1]

        x2 = node[cell[c, 2], 0]
        y2 = node[cell[c, 2], 1]

        l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 
        gphi = ti.Matrix([
            [(y1 - y2)/l, (x2 - x1)/l], 
            [(y2 - y0)/l, (x0 - x2)/l],
            [(y0 - y1)/l, (x1 - x0)/l]])
        l *= 0.5
        for i in range(ldof):
            for j in range(ldof):
                S[c, i, j] = 0.0
                for q in range(NQ):
                    vix = R[q, i, 0]*gphi[0, 0] + R[q, i, 1]*gphi[1, 0] + R[q, i, 2]*gphi[2, 0]
                    viy = R[q, i, 0]*gphi[0, 1] + R[q, i, 1]*gphi[1, 1] + R[q, i, 2]*gphi[2, 1]
                    vjx = R[q, j, 0]*gphi[0, 0] + R[q, j, 1]*gphi[1, 0] + R[q, j, 2]*gphi[2, 0]
                    vjy = R[q, j, 0]*gphi[0, 1] + R[q, j, 1]*gphi[1, 1] + R[q, j, 2]*gphi[2, 1]
                    S[c, i, j] += ws[q]*(vix*vjx + viy*vjy)

                S[c, i, j] *= l

@ti.kernel
def quad_lagrange_cell_stiff_matrix_1(
        node : ti.template(), 
        cell : ti.template(),
        S : ti.template()):
    """
    Note:
        四边形网格上的双线性拉格朗日有限元空间, 组装单元刚度矩阵
    """
    for c in range(cell.shape[0]):
        x0 = node[cell[c, 0], 0]
        y0 = node[cell[c, 0], 1]

        x1 = node[cell[c, 1], 0]
        y1 = node[cell[c, 1], 1]

        x2 = node[cell[c, 2], 0]
        y2 = node[cell[c, 2], 1]

        l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 
        gphi = ti.Matrix([
            [(y1 - y2)/l, (x2 - x1)/l], 
            [(y2 - y0)/l, (x0 - x2)/l],
            [(y0 - y1)/l, (x1 - x0)/l]])
        l *= 0.5
        for i in ti.static(range(4)):
            for j in ti.static(range(4)):
                S[c, i, j] = l*(gphi[i, 0]*gphi[j, 0] + gphi[i, 1]*gphi[j, 1])

@ti.kernel
def quad_lagrange_cell_stiff_matrix_p(
        node : ti.template(),  # (NN, 2)
        cell : ti.template(),  # (NC, 3)
        ws : ti.template(), # (NQ, )
        R : ti.template(), # (NQ, ldof, 3)
        S : ti.template() # (NC, ldof, ldof)
        ):
    """
    Note:
        四边形网格上的 p 次拉格朗日元, 组装刚度矩阵
    """
    NC = cell.shape[0]
    NQ = R.shape[0]
    ldof = R.shape[1]
    for c in range(cell.shape[0]):
        x0 = node[cell[c, 0], 0]
        y0 = node[cell[c, 0], 1]

        x1 = node[cell[c, 1], 0]
        y1 = node[cell[c, 1], 1]

        x2 = node[cell[c, 2], 0]
        y2 = node[cell[c, 2], 1]

        l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 
        gphi = ti.Matrix([
            [(y1 - y2)/l, (x2 - x1)/l], 
            [(y2 - y0)/l, (x0 - x2)/l],
            [(y0 - y1)/l, (x1 - x0)/l]])
        l *= 0.5
        for i in range(ldof):
            for j in range(ldof):
                S[c, i, j] = 0.0
                for q in range(NQ):
                    vix = R[q, i, 0]*gphi[0, 0] + R[q, i, 1]*gphi[1, 0] + R[q, i, 2]*gphi[2, 0]
                    viy = R[q, i, 0]*gphi[0, 1] + R[q, i, 1]*gphi[1, 1] + R[q, i, 2]*gphi[2, 1]
                    vjx = R[q, j, 0]*gphi[0, 0] + R[q, j, 1]*gphi[1, 0] + R[q, j, 2]*gphi[2, 0]
                    vjy = R[q, j, 0]*gphi[0, 1] + R[q, j, 1]*gphi[1, 1] + R[q, j, 2]*gphi[2, 1]
                    S[c, i, j] += ws[q]*(vix*vjx + viy*vjy)

                S[c, i, j] *= l

@ti.kernel
def lagrange_cell_stiff_matrix_cnode(cnode : ti.template(), S : ti.template()):
    for c in range(cnode.shape[0]):
        x0 = cnode[c, 0, 0] 
        y0 = cnode[c, 0, 1]

        x1 = cnode[c, 1, 0] 
        y1 = cnode[c, 1, 1]

        x2 = cnode[c, 2, 0] 
        y2 = cnode[c, 2, 1]


        l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 
        gphi = ti.Matrix([
            [(y1 - y2)/l, (x2 - x1)/l], 
            [(y2 - y0)/l, (x0 - x2)/l],
            [(y0 - y1)/l, (x1 - x0)/l]])
        l *= 0.5
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                S[c, i, j] = l*(gphi[i, 0]*gphi[j, 0] +
                        gphi[i, 1]*gphi[j, 1])
