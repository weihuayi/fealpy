import taichi as ti
import numpy as np

from scipy.sparse import csr_matrix

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

        mi = mesh.multi_index_matrix(p)
        self.multiIndex = ti.field(self.itype, shape=mi.shape)
        self.multiIndex.from_numpy(mi)

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells()

        self.edof = p+1
        self.fdof = (p+1)*(p+2)//2
        self.vdof = (p+1)*(p+2)*(p+3)//6
        
        TD = self.top_dimension() 

        self.edge2dof = ti.field(self.itype, shape=(NE, self.edof))
        if TD == 2:
            self.cell2dof = ti.field(self.itype, shape=(NC, self.fdof))
        elif TD == 3:
            self.face2dof = ti.field(self.itype, shape=(NC, self.fdof))
            self.cell2dof = ti.field(self.itype, shape=(NC, self.vdof))


                    
    @ti.kernel
    def tet_to_dof(self):
        pass


    def geo_dimension(self):
        return self.mesh.node.shape[0]

    def top_dimension(self):
        return self.multiIndex.shape[1] - 1

    def number_of_nodes(self):
        return self.mesh.node.shape[0]
    
    def number_of_local_dofs(self):
        return self.multiIndex.shape[0]

    def number_of_cells(self):
        return self.mesh.cell.shape[0]
    
    def number_of_global_dofs(self):
        return self.mesh.number_of_global_interpolation_points(self.p)
   
    
    def shape_function(self, bc):
        """

        @brief 给定一个或一组重心坐标，计算所有单元基函数在重心坐标处的值,
        以及关于重心坐标的 1 阶导数值。

        @param[in] bc numpy.ndarray 形状为 (..., TD+1)

        """

        p = self.p
        TD = bc.shape[-1] - 1
        multiIndex = self.mesh.multi_index_matrix(p)
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
    def cell_stiff_matrices(self, 
            S: ti.template(), 
            B: ti.template(),
            W: ti.template()
            ):
        """
        计算网格上的所有单元刚度矩阵
        """ 
        ldof = B.shape[1]
        NQ = W.shape[0]
        NC = self.mesh.cell.shape[0]
     
        for c in range(NC):
            L,cm = self.mesh.grad_lambda(c)
            for k in range(NQ):
                for i in range(ldof):
                    vix = B[k,i,0]*L[0,0]+B[k,i,1]*L[1,0]+B[k,i,2]*L[2,0]
                    viy = B[k,i,0]*L[0,1]+B[k,i,1]*L[1,1]+B[k,i,2]*L[2,1]
                    S[c,i,i] += cm*W[k]*(vix*vix+viy*viy)
                    for j in range(i+1,ldof):
                        vjx = B[k,j,0]*L[0,0]+B[k,j,1]*L[1,0]+B[k,j,2]*L[2,0]
                        vjy = B[k,j,0]*L[0,1]+B[k,j,1]*L[1,1]+B[k,j,2]*L[2,1]
                        S[c,i,j] += cm*W[k]*(vix*vjx+viy*vjy)
                        S[c,j,i] = S[c,i,j]
    
    @ti.kernel
    def cell_pTgpx_matrices(self, 
            S: ti.template(), 
            B: ti.template(), 
            G: ti.template(),
            W: ti.template(),
            C: ti.template()
            ):
        """
        计算网格上的所有单元刚度矩阵
        """ 
        ldofi = B.shape[1]
        ldofj = G.shape[1]
        NQ = W.shape[0]
        NC = self.mesh.cell.shape[0]
        if ti.static(C == None):
            for c in range(NC):
                L,cm = self.mesh.grad_lambda(c)
                for k in range(NQ):
                    for i in range(ldofi):
                        for j in range(ldofj):
                            vjx = G[k,j,0]*L[0,0]+G[k,j,1]*L[1,0]+G[k,j,2]*L[2,0]
                            S[c,i,j] += W[k]*cm*vjx*B[k,i]
        else:
            for c in range(NC):
                L,cm = self.mesh.grad_lambda(c)
                for k in range(NQ):
                    for i in range(ldofi):
                        for j in range(ldofj):
                            vjx = G[k,j,0]*L[0,0]+G[k,j,1]*L[1,0]+G[k,j,2]*L[2,0]
                            S[c,i,j] += C[k,c]*W[k]*cm*vjx*B[k,i]
    
    @ti.kernel
    def cell_pTgpy_matrices(self, 
            S: ti.template(), 
            B: ti.template(), 
            G: ti.template(),
            W: ti.template(),
            C: ti.template()
            ):
        """
        计算网格上的所有单元刚度矩阵
        """ 
        NQ = W.shape[0]
        NC = self.mesh.cell.shape[0]
        ldofi = B.shape[1]
        ldofj = G.shape[1]
        if ti.static(C == None):
            for c in range(NC):
                L,cm = self.mesh.grad_lambda(c)
                for k in range(NQ):
                    for i in range(ldofi):
                        for j in range(ldofj):
                            vjy = G[k,j,0]*L[0,1]+G[k,j,1]*L[1,1]+G[k,j,2]*L[2,1]
                            S[c,i,j] += W[k]*cm*vjy*B[k,i]
        else :
            for c in range(NC):
                L,cm = self.mesh.grad_lambda(c)
                for k in range(NQ):
                    for i in range(ldofi):
                        for j in range(ldofj):
                            vjy = G[k,j,0]*L[0,1]+G[k,j,1]*L[1,1]+G[k,j,2]*L[2,1]
                            S[c,i,j] += C[k,c]*W[k]*cm*vjy*B[k,i]
    
    @ti.kernel
    def cell_gpxTp_matrices(self, 
            S: ti.template(), 
            B: ti.template(), 
            G: ti.template(), 
            W: ti.template(),
            C: ti.template()
            ):
        """
        计算网格上的所有单元刚度矩阵
        """ 
        NQ = W.shape[0]
        NC = self.mesh.cell.shape[0]
     
        ldofi = G.shape[1]
        ldofj = B.shape[1]
        if ti.static(C == None):
            for c in range(NC):
                L,cm = self.mesh.grad_lambda(c)
                for k in range(NQ):
                    for i in range(ldofi):
                        vix = G[k,i,0]*L[0,0]+G[k,i,1]*L[1,0]+G[k,i,2]*L[2,0]
                        for j in range(ldofj):
                            S[c,i,j] += W[k]*cm*vix*B[k,j]
        else :
            for c in range(NC):
                L,cm = self.mesh.grad_lambda(c)
                for k in range(NQ):
                    for i in range(ldofi):
                        vix = G[k,i,0]*L[0,0]+G[k,i,1]*L[1,0]+G[k,i,2]*L[2,0]
                        for j in range(ldofj):
                            S[c,i,j] += C[k,c]*W[k]*cm*vix*B[k,j]
    
    @ti.kernel
    def cell_gpyTp_matrices(self, 
            S: ti.template(), 
            B: ti.template(), 
            G: ti.template(), 
            W: ti.template(),
            C: ti.template()
            ):
        """
        计算网格上的所有单元刚度矩阵
        """ 
        ldof = B.shape[1]
        NQ = W.shape[0]
        NC = self.mesh.cell.shape[0]
     
        ldofi = G.shape[1]
        ldofj = B.shape[1]
        if ti.static(C == None):
            for c in range(NC):
                L,cm = self.mesh.grad_lambda(c)
                for k in range(NQ):
                    for i in range(ldofi):
                        viy = G[k,i,0]*L[0,1]+G[k,i,1]*L[1,1]+G[k,i,2]*L[2,1]
                        for j in range(ldofj):
                            S[c,i,j] += W[k]*cm*viy*B[k,j]
        else :
            for c in range(NC):
                L,cm = self.mesh.grad_lambda(c)
                for k in range(NQ):
                    for i in range(ldofi):
                        viy = G[k,i,0]*L[0,1]+G[k,i,1]*L[1,1]+G[k,i,2]*L[2,1]
                        for j in range(ldofj):
                            S[c,i,j] += C[k,c]*W[k]*cm*vix*B[k,j]
    
    @ti.kernel
    def cell_mass_matrices(self, 
            S: ti.template(), 
            B: ti.template(), 
            W: ti.template(),
            val: ti.template()
            ):
        """
        计算网格上的所有单元质量矩阵
        """ 
        # 积分单元矩阵
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
        R0,_ = self.shape_function(bc)
        B.from_numpy(R0)

        W = ti.field(ti.f64, shape=(NQ, ))
        W.from_numpy(ws)
        
        val = ti.field(ti.f64, shape=(ldof,ldof))
        K = ti.field(ti.f64, shape=(NC, ldof, ldof))
        self.cell_mass_matrices(K, B, W,val)

        M = K.to_numpy()
        if c is not None:
            M *= c # 目前假设 c 是常数
        
        
        cell2dof = ti.field(ti.u32,shape=(NC,ldof))
        self.mesh.cell_to_dof(self.p,cell2dof)
        cell2dof = cell2dof.to_numpy()
        
        NN = self.number_of_global_dofs()
        
        I = np.broadcast_to(cell2dof[:, :, None], shape=M.shape)
        J = np.broadcast_to(cell2dof[:, None, :], shape=M.shape)
        
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


