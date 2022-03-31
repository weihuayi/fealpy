import taichi as ti
import numpy as np

from scipy.sparse import csr_matrix

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

@ti.data_oriented
class LagrangeFiniteElementSpace():
    """
    单纯型网格上的任意次拉格朗日空间，这里的单纯型网格是指
    * 区间网格(1d)
    * 三角形网格(2d)
    * 四面体网格(3d)
    """
    def __init__(self, mesh, p=1, spacetype='C', q=None):
        self.mesh = mesh

        self.itype = mesh.itype
        self.ftype = mesh.ftype

        self.p = p

        TD = self.mesh.top_dimension()
        mi = multi_index_matrix[TD](p)
        self.multiIndex = ti.field(self.itype, shape=mi.shape)
        self.multiIndex.from_numpy(mi)

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells()

        self.edof = p+1
        self.fdof = (p+1)*(p+2)//2
        self.vdof = (p+1)*(p+2)*(p+3)//6

        self.edge2dof = ti.field(self.itype, shape=(NE, self.edof))
        if TD == 2:
            self.cell2dof = ti.field(self.itype, shape=(NC, self.fdof))
        elif TD == 3:
            self.face2dof = ti.field(self.itype, shape=(NC, self.fdof))
            self.cell2dof = ti.field(self.itype, shape=(NC, self.vdof))

    @ti.kernel
    def edge_to_dof(self):
        n = self.p - 1
        start = self.mesh.node.shape[0]
        for i in range(self.mesh.edge.shape[0]):
            self.edge2dof[i, 0] = self.mesh.edge[i, 0]
            self.edge2dof[i, self.p] = self.mesh.edge[i, 1]
            for j in ti.static(range(1, self.p)):
                self.edge2dof[i, j] = start + i*n + j - 1

    @ti.kernel
    def tri_to_dof(self):
        NN = self.mesh.node.shape[0]
        NE = self.mesh.edge.shape[0]
        for c in range(self.mesh.cell.shape[0]): 
            # 三个顶点 
            self.cell2dof[c, 0] = self.mesh.cell[c, 0]
            self.cell2dof[c, self.fdof - self.p - 1] = self.mesh.cell[c, 1] # 不支持负数索引
            self.cell2dof[c, self.fdof - 1] = self.mesh.cell[c, 2]

            # 第 0 条边
            ei = self.mesh.cell2edge[c, 0]
            v0 = self.mesh.edge[ei, 0]
            s0 = self.fdof - self.p
            if v0 == self.mesh.cell[c, 1]:
                for i in range(1, self.p):
                    self.cell2dof[c, s0] = self.edge2dof[ei, i]
                    s0 += 1
            else:
                for i in range(1, self.p):
                    self.cell2dof[c, s0] = self.edge2dof[ei, self.p-i] 
                    s0 += 1

            # 第 1 条边
            ei = self.mesh.cell2edge[c, 1]
            v0 = self.mesh.edge[ei, 0]
            s0 = 2
            if v0 == self.mesh.cell[c, 0]:
                for i in range(1, self.p):
                    self.cell2dof[c, s0] = self.edge2dof[ei, i]
                    s0 += i + 2
            else:
                for i in range(1, self.p):
                    self.cell2dof[c, s0] = self.edge2dof[ei, self.p-i]
                    s0 += i + 2 

            # 第 2 条边
            ei = self.mesh.cell2edge[c, 2]
            v0 = self.mesh.edge[ei, 0]
            s0 = 1
            if v0 == self.mesh.cell[c, 0]:
                for i in range(1, self.p):
                    self.cell2dof[c, s0] = self.edge2dof[ei, i]
                    s0 += i + 1
            else:
                for i in range(1, self.p):
                    self.cell2dof[c, s0] = self.edge2dof[ei, self.p-i]
                    s0 += i + 1 

            # 内部点
            if self.p >= 3:
                start = NN + (self.p-1)*NE
                s0 = 4
                level = self.p - 2 
                nd = (self.p - 2)*(self.p - 1)//2
                s1 = 0
                for l in range(0, level):
                    for i in range(0, l+1):
                        print("s0:", s0, c, start, NN, NE)
                        self.cell2dof[c, s0] = start + c*nd + s1 
                        s0 += 1
                        s1 += 1 
                    s0 += 3
                    
    @ti.kernel
    def tet_to_dof(self):
        pass


    def geo_dimension(self):
        return self.mesh.node.shape[0]

    def top_dimension(self):
        return self.multiIndex.shape[1] - 1

    def number_of_local_dofs(self):
        return self.multiIndex.shape[0]

    def shape_function(self, bc):
        """

        @brief 给定一个或一组重心坐标，计算所有单元基函数在重心坐标处的值,
        以及关于重心坐标的 1 阶导数值。

        @param[in] bc numpy.ndarray 形状为 (..., TD+1)

        """

        p = self.p
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
    def cell_mass_matrices(self, 
            S: ti.template(), 
            B: ti.template(), 
            W: ti.template()
            ):
        """
        计算网格上的所有单元质量矩阵
        """
       
        # 积分单元矩阵
        val = ti.Matrix.zero(self.f64, B.shape[1], B.shape[1])
        for k in ti.static(range(B.shape[0])):
            for i in ti.static(range(B.shape[1])):
                val[i, i] += W[k]*B[k, i]*B[k, i]
                for j in ti.static(range(i+1, B.shape[1])):
                    val[i, j] += W[k]*B[k, i]*B[k, j] 

        for c in range(self.mesh.cell.shape[0]):
            cm = self.mesh.cellmeasure[c]
            for i in ti.static(range(B.shape[1])):
                S[c, i, i] = cm*val[i, i]
                for j in ti.static(range(i+1, B.shape[1])):
                    S[c, i, j] = cm*val[i, j]
                    S[c, j, i] = S[c, i, j]


    def mass_matrix(self, bc, ws, c=None):
        """
        组装总体质量矩阵
        """

        NC = self.number_of_cells() 
        ldof = self.number_of_local_dofs()
        NQ = len(ws)

        B = ti.field(ti.f64, shape=(NQ, ldof))
        R0, _ = self.lagrange_shape_function(bc)
        B.from_numpy(R0)

        W = ti.field(ti.f64, shape=(NQ, ))
        W.from_numpy(ws)

        K = ti.field(ti.f64, shape=(NC, ldof, ldof))
        self.cell_mass_matrices(K, B, W)

        M = K.to_numpy()
        if c is not None:
            M *= c # 目前假设 c 是常数

        cell = self.cell.to_numpy()
        I = np.broadcast_to(cell[:, :, None], shape=M.shape)
        J = np.broadcast_to(cell[:, None, :], shape=M.shape)

        NN = self.number_of_nodes() 
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(NN, NN))
        return M

    def pytest(self, bc):
        self.edge_to_dof()
        print("tiedge2dof:", self.edge2dof)
        self.tri_to_dof()
        print("ticell2dof:", self.cell2dof)

    @ti.kernel
    def titest(self, bc: ti.template()):
        pass


