import numpy as np
from numpy.linalg import inv
from scipy.sparse import bmat, coo_matrix, csc_matrix, csr_matrix, spdiags, eye
import scipy.io as sio

from .Function import Function
from .ScaledMonomialSpace2d import ScaledMonomialSpace2d
from ..quadrature import GaussLegendreQuadrature
from ..quadrature import PolygonMeshIntegralAlg
from ..common import ranges
from ..common import block, block_diag

class RDFNCVEMDof2d():
    """
    The dof manager of Stokes Div-Free Non Conforming VEM 2d space.
    """
    def __init__(self, mesh, p):
        """

        Parameter
        ---------
        mesh : the polygon mesh
        p : the order the space with p>=2
        """
        self.p = p
        self.mesh = mesh
        # 注意这里只包含每个单元边上的标量自由度 
        self.cell2dof, self.cell2dofLocation = self.cell_to_dof()

    def boundary_dof(self):
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        edge2dof = self.edge_to_dof()
        isBdEdge = self.mesh.ds.boundary_edge_flag()
        isBdDof[edge2dof[isBdEdge]] = True
        return isBdDof

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        edge2dof = np.arange(NE*p).reshape(NE, p)
        return edge2dof

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cellLocation = mesh.ds.cellLocation
        cell2edge = mesh.ds.cell_to_edge()

        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        cell2dofLocation = np.zeros(NC+1, dtype=np.int)
        cell2dofLocation[1:] = np.add.accumulate(ldof)
        cell2dof = np.zeros(cell2dofLocation[-1], dtype=np.int)

        edge2dof = self.edge_to_dof()
        edge2cell = mesh.ds.edge_to_cell()
        idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
        cell2dof[idx] = edge2dof

        idx = cell2dofLocation[edge2cell[:, [1]]] + edge2cell[:, [3]]*p + np.arange(p)
        cell2dof[idx] = edge2dof
        return cell2dof, cell2dofLocation

    def number_of_global_dofs(self):
        # 这里只有单元每个边上的自由度
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        gdof = NE*p
        return gdof

    def number_of_local_dofs(self):
        # 这里只有单元每个边上的自由度,  
        p = self.p
        mesh = self.mesh
        NCE = mesh.number_of_edges_of_cells()
        ldofs = NCE*p
        return ldofs


class ReducedDivFreeNonConformingVirtualElementSpace2d:
    def __init__(self, mesh, p=2, q=None):
        """
        Parameter
        ---------
        mesh : polygon mesh
        p : the space order, p>=2
        """
        self.p = p
        self.smspace = ScaledMonomialSpace2d(mesh, p, q=q)
        self.mesh = mesh
        self.dof = RDFNCVEMDof2d(mesh, p) # 注意这里只是标量的自由度管理
        self.integralalg = self.smspace.integralalg

        self.CM = self.smspace.cell_mass_matrix()
        self.EM = self.smspace.edge_mass_matrix()

        smldof = self.smspace.number_of_local_dofs()
        ndof = p*(p+1)//2
        self.H0 = inv(self.CM[:, 0:ndof, 0:ndof])
        self.H1 = inv(self.EM[:, 0:p, 0:p])

        self.ftype = self.mesh.ftype
        self.itype = self.mesh.itype

        self.Q, self.L = self.matrix_Q_L()
        self.G, self.B = self.matrix_G_B()
        self.E = self.matrix_E()
        self.U = self.matrix_U()
        self.R, self.J = self.matrix_R_J()
        self.D = self.matrix_D()
        self.PI0 = self.matrix_PI0()

    def project(self, u):
        p = self.p # p >= 2
        NE = self.mesh.number_of_edges()
        uh = self.function()
        def u0(x):
            return np.einsum('ij..., ijm->ijm...', u(x),
                    self.smspace.edge_basis(x, p=p-1))
        eh = self.mesh.entity_measure('edge')
        uh0 = self.integralalg.edge_integral(
                u0, edgetype=True)/eh[:, None, None]
        uh[0:2*NE*p] = uh0.reshape(-1, 2).T.flat
        if p > 2:
            idx = self.smspace.diff_index_1(p=p-2) # 一次求导后的非零基函数编号及求导系数
            x = idx['x']
            y = idx['y']
            def u1(x, index):
                return np.einsum('ij..., ijm->ijm...', u(x),
                        self.smspace.basis(x, p=p-2, index=index))
            area = self.smspace.cellmeasure
            uh1 = self.integralalg.integral(u1, celltype=True)/area[:, None, None]
            uh[2*NE*p:] += uh1[:, y[0], 0].flat
            uh[2*NE*p:] -= uh1[:, x[0], 1].flat
        return uh

    def project_to_complete_space(self, f, uh, ph, cuh, cph):
        """

        Parameters
        ----------
        uh : 
        cuh : 

        Notes
        -----
            把缩减虚单元空间中的函数投影回完全的虚单元空间, 其中边界和梯度正交空
        间的自由度直接设置, 但梯度空间的自由度要重新计算.
        """
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        ch = self.smspace.cellsize # 单元尺寸 

        cell2dof = self.dof.cell2dof # 这里只是边上的标量自由度管理
        cell2dofLocation = self.dof.cell2dofLocation

        idof = p*(p-1)
        idof0 = (p+1)*p//2 - 1 # G_{k-2}^\nabla 梯度空间对应的自由度个数

        c2d = cuh.space.cell_to_dof(doftype='cell') # 完全向量虚单元空间单元内部自由度全局编号
        cuh[:2*NE*p] = uh[:2*NE*p] # 边上自由度直接赋值 

        # 下面代码 cuh[c2d[:, idof0:]] 返回的是 copy 
        # cuh[c2d[:, idof0:]].flat[:] = uh[2*NE*p:] # 这里是 Bug
        if p > 2:
            cuh[c2d[:, idof0:]] = uh[2*NE*p:].reshape(-1, idof-idof0)  # 单元内部梯度正交空间对应的自由度全局编号

        # 下面处理梯度空间对应虚单元自由度
        n = mesh.edge_unit_normal()
        eh = mesh.entity_measure('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isBdEdge = (edge2cell[:, 0] == edge2cell[:, 1])

        area = self.smspace.cellmeasure # 单元面积
        Q0 = self.CM[:, 0, 1:idof0+1]/area[:, None]

        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = mesh.edge_bc_to_point(bcs) 
        phi = self.smspace.edge_basis(ps, p=p-1)


        # left element
        phi0 = self.smspace.basis(ps, index=edge2cell[:, 0], p=p-1)[..., 1:idof0+1]
        phi0 -= Q0[None, edge2cell[:, 0]]
        h = 1/ch[edge2cell[:, 0]]
        h *= eh # 积分的尺度
        h *= eh # 自由定义的尺度
        F = np.einsum('i, ijm, ijn, j->jmn', ws, phi0, phi, h)
        F = F@self.H1

        edge2dof = self.dof.edge_to_dof()
        val = np.einsum('jmn, jn, j->jm', F, uh[edge2dof], n[:, 0])
        np.add.at(cuh, c2d[edge2cell[:, 0], :idof0], val)
        val = np.einsum('jmn, jn, j->jm', F, uh[NE*p:][edge2dof], n[:, 1])
        np.add.at(cuh, c2d[edge2cell[:, 0], :idof0], val)

        # right element
        phi0 = self.smspace.basis(ps, index=edge2cell[:, 1], p=p-1)[..., 1:idof0+1]
        phi0 -= Q0[None, edge2cell[:, 1]]
        h = 1/ch[edge2cell[:, 1]]
        h *= eh
        h *= eh
        F = np.einsum('i, ijm, ijn, j->jmn', ws, phi0, phi, h)
        F = F@self.H1
        F[isBdEdge] = 0.0 # 注意边界边的右边单元的贡献要设为 0

        val = np.einsum('jmn, jn, j->jm', F, uh[edge2dof], n[:, 0])
        np.subtract.at(cuh, c2d[edge2cell[:, 1], :idof0], val)
        val = np.einsum('jmn, jn, j->jm', F, uh[NE*p:][edge2dof], n[:, 1])
        np.subtract.at(cuh, c2d[edge2cell[:, 1], :idof0], val)


        A = cuh.space.stiff_matrix(celltype=True)
        F = cuh.space.cell_grad_source_vector(f) # (NC, ldof0)
        c2d = cuh.space.cell_to_dof(doftype='cell')

        cell2dof = self.dof.cell2dof
        cell2dofLocation = self.dof.cell2dofLocation

        val = cph.reshape((NC, (p+1)*p//2))

        def f0(i):
            n = cell2dofLocation[i+1] - cell2dofLocation[i]
            s = slice(cell2dofLocation[i], cell2dofLocation[i+1])
            x = np.r_['0', cuh[cell2dof[s]], cuh[NE*p:][cell2dof[s]], cuh[c2d[i, :]]] 
            val[i, 1:] = A[i][2*n:2*n+idof0, :]@x 
            val[i, 1:] -= F[i]
            val[i, 1:] /= ch[i]
            val[i, 0] = ph[i]  
            val[i, 0] -= sum(val[i, 1:]*Q0[i, :])

        list(map(f0, range(NC)))



    def project_to_smspace(self, uh):
        p = self.p
        idof = (p-2)*(p-1)//2
        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells()
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        smldof = self.smspace.number_of_local_dofs()
        sh = self.smspace.function(dim=2)
        c2d = self.smspace.cell_to_dof()
        def f(i):
            PI0 = self.PI0[i]
            idx = cell2dof[cell2dofLocation[i]:cell2dofLocation[i+1]]
            x0 = uh[idx]
            x1 = uh[idx+NE*p]
            x2 = np.zeros(idof, dtype=self.ftype)
            if p > 2:
                start = 2*NE*p + i*idof
                x2[:] = uh[start:start+idof]
            x = np.r_[x0, x1, x2]
            y = (PI0[:2*smldof]@x).flat
            sh[c2d[i], 0] = y[:smldof]
            sh[c2d[i], 1] = y[smldof:]
        list(map(f, range(NC)))
        return sh

    def verify_matrix(self):
        p = self.p
        NC = self.mesh.number_of_cells()
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        def f(i):
            up = np.array([1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0.])
            G = block([
                [self.G[0][i]  ,   self.G[2][i]],
                [self.G[2][i].T,   self.G[1][i]]])
            B = block([
                [self.B[0][i]], 
                [self.B[1][i]]])
            s = slice(cell2dofLocation[i], cell2dofLocation[i+1])
            R = block([
                [self.R[0][0][:, s], self.R[0][1][:, s], self.R[0][2][i]],
                [self.R[1][0][:, s], self.R[1][1][:, s], self.R[1][2][i]]])
            m, n = self.J[0].shape[0], self.R[0][2][i].shape[1]
            J = block([[self.J[0][:, s], self.J[1][:, s], np.zeros((m,n))]])
          
            D = block([
                [self.D[0][s],            0],
                [           0, self.D[0][s]],
                [self.D[1][i], self.D[2][i]]])
            A = block([
                [  G, B],
                [B.T, 0]])
            B = block([[R], [J]])

            S = inv(A)@B
            smldof = (p+1)*(p+2)//2 
            I = S@D
            sio.savemat('I'+str(i)+'.mat', {'I'+str(i):I})

        PI0 = list(map(f, range(NC)))

    def matrix_PI0(self):
        p = self.p
        NC = self.mesh.number_of_cells()
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        def f(i):
            G = block([
                [self.G[0][i]  ,   self.G[2][i]  , self.B[0][i]],
                [self.G[2][i].T,   self.G[1][i]  , self.B[1][i]],
                [self.B[0][i].T,   self.B[1][i].T,            0]]
                )
            s = slice(cell2dofLocation[i], cell2dofLocation[i+1])
            R = block([
                [self.R[0][0][:, s], self.R[0][1][:, s], self.R[0][2][i]],
                [self.R[1][0][:, s], self.R[1][1][:, s], self.R[1][2][i]],
                [   self.J[0][:, s],    self.J[1][:, s],     0]])
            PI = inv(G)@R
            return PI
        PI0 = list(map(f, range(NC)))
        return PI0

    def matrix_Q_L(self):
        p = self.p
        if p > 2:
            mesh = self.mesh
            NC = mesh.number_of_cells()
            area = self.smspace.cellmeasure

            smldof = self.smspace.number_of_local_dofs() # 标量 p 次单元缩放空间的维数
            ndof = (p-2)*(p-1)//2 # 标量 p - 3 次单元缩放单项项式空间的维数
            cell2dof =  self.dof.cell2dof
            cell2dofLocation = self.dof.cell2dofLocation # 边上的标量的自由度信息
            CM = self.CM

            idx = self.smspace.diff_index_1(p=p-2)
            x = idx['x']
            y = idx['y']
            Q = CM[:, x[0][:, None], x[0]] + CM[:, y[0][:, None], y[0]]

            L0 = np.zeros((ndof, len(cell2dof)), dtype=self.ftype)
            L1 = np.zeros((ndof, len(cell2dof)), dtype=self.ftype)
            L2 = np.zeros((NC, ndof, ndof), dtype=self.ftype)
            idx = np.arange(ndof)
            L2[:, idx, idx] = area[:, None]
            return Q, [L0, L1, L2]
        else:
            return None, None

    def matrix_G_B(self):
        """
        计算单元投投影算子的左端矩阵
        """
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()

        smldof = self.smspace.number_of_local_dofs()
        ndof = smldof - p - 1

        area = self.smspace.cellmeasure
        ch = self.smspace.cellsize
        CM = self.CM
        
        smldof = self.smspace.number_of_local_dofs()
        ndof = smldof - p - 1
        # 分块矩阵 G = [[G00, G01], [G10, G11]]
        G00 = np.zeros((NC, smldof, smldof), dtype=self.ftype)
        G01 = np.zeros((NC, smldof, smldof), dtype=self.ftype)
        G11 = np.zeros((NC, smldof, smldof), dtype=self.ftype)

        idx = self.smspace.diff_index_1()
        x = idx['x']
        y = idx['y']
        L = x[1][None, ...]/ch[..., None]
        R = y[1][None, ...]/ch[..., None]
        mxx = np.einsum('ij, ijk, ik->ijk', L, CM[:, 0:ndof, 0:ndof], L)
        myx = np.einsum('ij, ijk, ik->ijk', R, CM[:, 0:ndof, 0:ndof], L)
        myy = np.einsum('ij, ijk, ik->ijk', R, CM[:, 0:ndof, 0:ndof], R)

        G00[:, x[0][:, None], x[0]] += mxx
        G00[:, y[0][:, None], y[0]] += 0.5*myy

        G01[:, y[0][:, None], x[0]] += 0.5*myx

        G11[:, x[0][:, None], x[0]] += 0.5*mxx
        G11[:, y[0][:, None], y[0]] += myy

        mx = L*CM[:, 0, 0:ndof]/area[:, None]
        my = R*CM[:, 0, 0:ndof]/area[:, None]
        G00[:, y[0][:, None], y[0]] += np.einsum('ij, ik->ijk', my, my)
        G01[:, y[0][:, None], x[0]] -= np.einsum('ij, ik->ijk', my, mx)
        G11[:, x[0][:, None], x[0]] += np.einsum('ij, ik->ijk', mx, mx)

        m = CM[:, 0, :]/area[:, None]
        val = np.einsum('ij, ik->ijk', m, m)
        G00[:, 0:smldof, 0:smldof] += val
        G11[:, 0:smldof, 0:smldof] += val
        G = [G00, G11, G01]

        # 分块矩阵 B = [B0, B1]
        B0 = np.zeros((NC, smldof, ndof), dtype=self.ftype)
        B1 = np.zeros((NC, smldof, ndof), dtype=self.ftype)
        B0[:, x[0]] = np.einsum('ij, ijk->ijk', L, CM[:, 0:ndof, 0:ndof])
        B1[:, y[0]] = np.einsum('ij, ijk->ijk', R, CM[:, 0:ndof, 0:ndof])

        B = [B0, B1]
        return G, B

    def matrix_T(self):
        """

        $\\bfQ_0^K(\\bfM_k) : \\bfQ_0^K(\\tilde\\bPhi_k)$
        """

        p = self.p
        mesh = self.mesh
        ch = self.smspace.cellsize # 单元尺寸 

        smldof = self.smspace.number_of_local_dofs(p=p) # 标量 p 次单元缩放空间的维数
        cell2dof = self.dof.cell2dof
        cell2dofLocation = self.dof.cell2dofLocation # 标量的单元边自由度信息

        T00 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype) 
        T01 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype) 
        T10 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype) 
        T11 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype) 

        n = mesh.edge_unit_normal()
        eh = mesh.entity_measure('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = mesh.edge_bc_to_point(bcs) 
        phi = self.smspace.edge_basis(ps, p=p-1)

        area = self.smspace.cellmeasure # 单元面积
        Q0 = self.CM[:, 0, :]/area[:, None] # (NC, smdof) \bfQ_0^K(\bfm_k)

        phi0 = self.smspace.basis(ps, index=edge2cell[:, 0], p=1) #(NQ, NC, 3)
        # only need m_1 and m_2
        phi0 = phi0[:, :, 1:3] 
        phi0 -= Q0[None, edge2cell[:, 0], 1:3]

        # self.H1 is the inverse of Q_{p-1}^F
        # i->quadrature, j->edge, m-> basis on domain, n->basis on edge
        a = 1/ch[edge2cell[:, 0]]
        F0 = np.einsum('i, ijm, ijn, j, j, j->jmn', ws, phi0, phi, eh, eh, a)
        F0 = F0@self.H1

        idx0 = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
        val = np.einsum('jm, jn, j->mjn', Q0[edge2cell[:, 0]], F0[:, 0, :], n[:, 0]) 
        np.add.at(T00, (np.s_[:], idx0), val)
        val = np.einsum('jm, jn, j->mjn', Q0[edge2cell[:, 0]], F0[:, 1, :], n[:, 0]) 
        np.add.at(T10, (np.s_[:], idx0), val)

        val = np.einsum('jm, jn, j->mjn', Q0[edge2cell[:, 0]], F0[:, 0, :], n[:, 1]) 
        np.add.at(T01, (np.s_[:], idx0), val)
        val = np.einsum('jm, jn, j->mjn', Q0[edge2cell[:, 0]], F0[:, 1, :], n[:, 1]) 
        np.add.at(T11, (np.s_[:], idx0), val)

        if isInEdge.sum() > 0:
            phi1 = self.smspace.basis(ps, index=edge2cell[:, 1], p=1)
            phi1 = phi1[:, :, 1:3] 
            phi1 -= Q0[None, edge2cell[:, 1], 1:3]
            a = 1/ch[edge2cell[:, 1]]
            F1 = np.einsum('i, ijm, ijn, j, j, j->jmn', ws, phi1, phi, eh, eh, a)
            F1 = F1@self.H1
            idx0 = cell2dofLocation[edge2cell[:, [1]]] + edge2cell[:, [3]]*p + np.arange(p)
            val = np.einsum('jm, jn, j->mjn', Q0[edge2cell[:, 1]], F1[:, 0, :], n[:, 0]) 
            np.subtract.at(T00, (np.s_[:], idx0[isInEdge]), val[:, isInEdge, :])
            val = np.einsum('jm, jn, j->mjn', Q0[edge2cell[:, 1]], F1[:, 1, :], n[:, 0]) 
            np.subtract.at(T10, (np.s_[:], idx0[isInEdge]), val[:, isInEdge, :])

            val = np.einsum('jm, jn, j->mjn', Q0[edge2cell[:, 1]], F1[:, 0, :], n[:, 1]) 
            np.subtract.at(T01, (np.s_[:], idx0[isInEdge]), val[:, isInEdge, :])
            val = np.einsum('jm, jn, j->mjn', Q0[edge2cell[:, 1]], F1[:, 1, :], n[:, 1]) 
            np.subtract.at(T11, (np.s_[:], idx0[isInEdge]), val[:, isInEdge, :])
            
        return [[T00, T01], [T10, T11]]

    def matrix_E(self):
        """
        计算 M_{k-2} 和 \\tilde\\bPhi 的矩阵, 这是其它右端矩阵计算的基础。
        """
        p = self.p

        mesh = self.mesh

        ch = self.smspace.cellsize # 单元尺寸 

        smldof = self.smspace.number_of_local_dofs(p=p-2) # 标量 p-2 次单元缩放空间的维数
        cell2dof = self.dof.cell2dof
        cell2dofLocation = self.dof.cell2dofLocation # 标量的单元边自由度信息

        E00 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype) 
        E01 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype) 

        E10 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype) 
        E11 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype) 

        n = mesh.edge_unit_normal()
        eh = mesh.entity_measure('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        ndof = self.smspace.number_of_local_dofs(p=p-1)
        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights # bcs: (NQ, 2), ws: (NQ, )
        ps = mesh.edge_bc_to_point(bcs) 
        phi = self.smspace.edge_basis(ps, p=p-1)

        area = self.smspace.cellmeasure # 单元面积
        Q0 = self.CM[:, 0, 0:ndof]/area[:, None] # (NC, ndof)

        # (NQ, NE, ndof)
        phi0 = self.smspace.basis(ps, index=edge2cell[:, 0], p=p-1)
        phi0 -= Q0[None, edge2cell[:, 0], :]
        c = np.repeat(range(1, p), range(1, p))
        F0 = np.einsum('i, ijm, ijn, j, j, j->jmn', 
                ws, phi0, phi, eh, eh, ch[edge2cell[:, 0]])
        F0 = F0@self.H1
        # F0 /= c[:, None, None] here is a bug

        idx = self.smspace.diff_index_1(p=p-1)
        x = idx['x']
        y = idx['y']
        idx0 = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
        val = np.einsum('jmn, j->mjn', F0, n[:, 0]) 
        np.add.at(E00, (np.s_[:], idx0), val[x[0]]/c[:, None, None])
        np.add.at(E10, (np.s_[:], idx0), val[y[0]]/c[:, None, None])

        val = np.einsum('jmn, j->mjn', F0, n[:, 1])
        np.add.at(E01, (np.s_[:], idx0), val[x[0]]/c[:, None, None])
        np.add.at(E11, (np.s_[:], idx0), val[y[0]]/c[:, None, None])

        if np.any(isInEdge):
            phi1 = self.smspace.basis(ps, index=edge2cell[:, 1], p=p-1)
            phi1 -= Q0[None, edge2cell[:, 1], :]
            F1 = np.einsum('i, ijm, ijn, j, j, j->jmn', 
                    ws, phi1, phi, eh, eh, ch[edge2cell[:, 1]])
            F1 = F1@self.H1
            idx0 = cell2dofLocation[edge2cell[:, [1]]] + edge2cell[:, [3]]*p + np.arange(p)
            val = np.einsum('jmn, j->mjn', F1, n[:, 0])
            np.subtract.at(E00, 
                    (np.s_[:], idx0[isInEdge]), 
                    val[x[0][:, None], isInEdge]/c[:, None, None])
            np.subtract.at(E10, 
                    (np.s_[:], idx0[isInEdge]), 
                    val[y[0][:, None], isInEdge]/c[:, None, None])

            val = np.einsum('jmn, j->mjn', F1, n[:, 1])
            np.subtract.at(E01, 
                    (np.s_[:], idx0[isInEdge]), 
                    val[x[0][:, None], isInEdge]/c[:, None, None])
            np.subtract.at(E11, 
                    (np.s_[:], idx0[isInEdge]), 
                    val[y[0][:, None], isInEdge]/c[:, None, None])
            
        NC = mesh.number_of_cells()
        idof = (p-2)*(p-1)//2
        E02 = np.zeros((NC, smldof, idof), dtype=self.ftype) # 单元内部自由度
        E12 = np.zeros((NC, smldof, idof), dtype=self.ftype) 
        if p > 2: # idof > 0
            idx = self.smspace.diff_index_1(p=p-2)
            x = idx['x']
            y = idx['y']
            I = np.arange(idof)
            E02[:, y[0], I] =  area[:, None]*y[1]/c[y[0]]
            E12[:, x[0], I] = -area[:, None]*x[1]/c[x[0]]

        return [[E00, E01, E02], [E10, E11, E12]]

    def matrix_R_J(self):
        """
        计算单元投投影算子的右端矩阵
        """
        p = self.p

        mesh = self.mesh
        NC = mesh.number_of_cells()

        area = self.smspace.cellmeasure # 单元面积
        ch = self.smspace.cellsize # 单元尺寸 

        smldof = self.smspace.number_of_local_dofs(p=p) # 标量 p 次单元缩放空间的维数
        ndof = smldof - p - 1 # 标量 p - 1 次单元缩放单项项式空间的维数
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation # 标量的单元边自由度信息

        CM = self.CM # 单元质量矩阵

        # 构造分块矩阵 
        #  R = [
        #       [R00, R01, R02], 
        #       [R10, R11, R12]
        #  ]

        idof = (p-2)*(p-1)//2
        R00 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        R01 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        R02 = np.zeros((NC, smldof, idof), dtype=self.ftype)

        R10 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        R11 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        R12 = np.zeros((NC, smldof, idof), dtype=self.ftype)

        idx = self.smspace.diff_index_2()
        xx = idx['xx']
        yy = idx['yy']
        xy = idx['xy']

        ldofs = self.dof.number_of_local_dofs()
        a = np.repeat(area, ldofs)
        R00[xx[0], :] -=     xx[1][:, None]*self.E[0][0]
        R00[yy[0], :] -= 0.5*yy[1][:, None]*self.E[0][0]
        R00[xy[0], :] -= 0.5*xy[1][:, None]*self.E[1][0]
        R00 /= a

        R01[xx[0], :] -=     xx[1][:, None]*self.E[0][1]
        R01[yy[0], :] -= 0.5*yy[1][:, None]*self.E[0][1]
        R01[xy[0], :] -= 0.5*xy[1][:, None]*self.E[1][1]
        R01 /= a


        R10[xx[0], :] -= 0.5*xx[1][:, None]*self.E[1][0]
        R10[yy[0], :] -=     yy[1][:, None]*self.E[1][0]
        R10[xy[0], :] -= 0.5*xy[1][:, None]*self.E[0][0]
        R10 /= a

        R11[xx[0], :] -= 0.5*xx[1][:, None]*self.E[1][1]
        R11[yy[0], :] -=     yy[1][:, None]*self.E[1][1]
        R11[xy[0], :] -= 0.5*xy[1][:, None]*self.E[0][1]
        R11 /= a

        if p > 2:
            R02[:, xx[0], :] -=     xx[1][None, :, None]*self.E[0][2]
            R02[:, yy[0], :] -= 0.5*yy[1][None, :, None]*self.E[0][2]
            R02[:, xy[0], :] -= 0.5*xy[1][None, :, None]*self.E[1][2]
            R02 /= area[:, None, None]

            R12[:, xx[0], :] -= 0.5*xx[1][None, :, None]*self.E[1][2]
            R12[:, yy[0], :] -=     yy[1][None, :, None]*self.E[1][2]
            R12[:, xy[0], :] -= 0.5*xy[1][None, :, None]*self.E[0][2]
            R12 /= area[:, None, None]
            
        n = mesh.edge_unit_normal()
        eh = mesh.entity_measure('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights # bcs: (NQ, 2)
        ps = mesh.edge_bc_to_point(bcs)
        # phi0: (NQ, NE, ndof)
        phi0 = self.smspace.basis(ps, index=edge2cell[:, 0], p=p-1) 
        # phi: (NQ, NE, p)
        phi = self.smspace.edge_basis(ps, p=p-1)
        # F0: (NE, ndof, p) 
        F0 = np.einsum('i, ijm, ijn, j, j->jmn', ws, phi0, phi, eh, eh)
        # F0: (NE, ndof, p)
        F0 = F0@self.H1

        idx = self.smspace.diff_index_1() # 一次求导后的非零基函数编号及求导系数
        x = idx['x']
        y = idx['y']
        # idx0: (NE, p)
        idx0 = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
        # h2: (NE, ndof)
        h2 = x[1][None, :]/ch[edge2cell[:, [0]]]
        h3 = y[1][None, :]/ch[edge2cell[:, [0]]]

        # val: (ndof, NE, p)
        val = np.einsum('jm, jmn, j-> mjn', h2, F0, n[:, 0])
        # x[0]: (ndof, 1, 1)  idx0: (NE, p) --> x[0] and idx0: (ndof, NE, p)
        np.add.at(R00, (x[0][:, None, None], idx0), val)
        val = np.einsum('jm, jmn, j-> mjn', h3, F0, 0.5*n[:, 1])
        np.add.at(R00, (y[0][:, None, None], idx0), val)

        val = np.einsum('jm, jmn, j-> mjn', h2, F0, 0.5*n[:, 0])
        np.add.at(R11, (x[0][:, None, None], idx0), val)
        val = np.einsum('jm, jmn, j-> mjn', h3, F0, n[:, 1])
        np.add.at(R11, (y[0][:, None, None], idx0), val)

        val = np.einsum('jm, jmn, j-> mjn', h3, F0, 0.5*n[:, 0])
        np.add.at(R01, (y[0][:, None, None], idx0), val)
        val = np.einsum('jm, jmn, j-> mjn', h2, F0, 0.5*n[:, 1])
        np.add.at(R10, (x[0][:, None, None], idx0), val)

        a2 = area**2
        start = cell2dofLocation[edge2cell[:, 0]] + edge2cell[:, 2]*p
        # val: (ndof, NE)
        val = np.einsum('jm, jm, j, j->mj',
            h3, CM[edge2cell[:, 0], 0:ndof, 0], eh/a2[edge2cell[:, 0]], n[:, 1])
        # y[0]: (ndof, 1)  start: (NE, ) -->  y[0] and start : (ndof, NE)
        np.add.at(R00, (y[0][:, None], start), val)

        val = np.einsum('jm, jm, j, j->mj',
            h3, CM[edge2cell[:, 0], 0:ndof, 0], eh/a2[edge2cell[:, 0]], n[:, 0])
        np.subtract.at(R01, (y[0][:, None], start), val)

        val = np.einsum('jm, jm, j, j->mj',
            h2, CM[edge2cell[:, 0], 0:ndof, 0], eh/a2[edge2cell[:, 0]], n[:, 1])
        np.subtract.at(R10, (x[0][:, None], start), val)

        val = np.einsum('jm, jm, j, j->mj',
            h2, CM[edge2cell[:, 0], 0:ndof, 0], eh/a2[edge2cell[:, 0]], n[:, 0])
        np.add.at(R11, (x[0][:, None], start), val)

        if np.any(isInEdge):
            phi1 = self.smspace.basis(ps, index=edge2cell[:, 1], p=p-1)
            F1 = np.einsum('i, ijm, ijn, j, j->jmn', ws, phi1, phi, eh, eh)
            F1 = F1@self.H1
            idx0 = cell2dofLocation[edge2cell[:, [1]]] + edge2cell[:, [3]]*p + np.arange(p)
            h2 = x[1][None, :]/ch[edge2cell[:, [1]]]
            h3 = y[1][None, :]/ch[edge2cell[:, [1]]]

            val = np.einsum('jm, jmn, j-> mjn', h2, F1, n[:, 0])
            np.subtract.at(R00, (x[0][:, None, None], idx0[isInEdge]), val[:, isInEdge])
            val = np.einsum('jm, jmn, j-> mjn', h3, F1, 0.5*n[:, 1])
            np.subtract.at(R00, (y[0][:, None, None], idx0[isInEdge]), val[:, isInEdge])

            val = np.einsum('jm, jmn, j-> mjn', h2, F1, 0.5*n[:, 0])
            np.subtract.at(R11, (x[0][:, None, None], idx0[isInEdge]), val[:, isInEdge])
            val = np.einsum('jm, jmn, j-> mjn', h3, F1, n[:, 1])
            np.subtract.at(R11, (y[0][:, None, None], idx0[isInEdge]), val[:, isInEdge])

            val = np.einsum('jm, jmn, j-> mjn', h3, F1, 0.5*n[:, 0])
            np.subtract.at(R01, (y[0][:, None, None], idx0[isInEdge]), val[:, isInEdge])
            val = np.einsum('jm, jmn, j-> mjn', h2, F1, 0.5*n[:, 1])
            np.subtract.at(R10, (x[0][:, None, None], idx0[isInEdge]), val[:, isInEdge])

            start = cell2dofLocation[edge2cell[:, 1]] + edge2cell[:, 3]*p
            val = np.einsum('jm, jm, j, j->mj',
                h3, CM[edge2cell[:, 1], 0:ndof, 0], eh/a2[edge2cell[:, 1]], n[:, 1])
            np.subtract.at(R00, (y[0][:, None], start[isInEdge]), val[:, isInEdge])

            val = np.einsum('jm, jm, j, j->mj',
                h3, CM[edge2cell[:, 1], 0:ndof, 0], eh/a2[edge2cell[:, 1]], n[:, 0])
            np.add.at(R01, (y[0][:, None], start[isInEdge]), val[:, isInEdge])

            val = np.einsum('jm, jm, j, j->mj',
                h2, CM[edge2cell[:, 1], 0:ndof, 0], eh/a2[edge2cell[:, 1]], n[:, 1])
            np.add.at(R10, (x[0][:, None], start[isInEdge]), val[:, isInEdge])

            val = np.einsum('jm, jm, j, j->mj',
                h2, CM[edge2cell[:, 1], 0:ndof, 0], eh/a2[edge2cell[:, 1]], n[:, 0])
            np.subtract.at(R11, (x[0][:, None], start[isInEdge]), val[:, isInEdge])

        T = self.matrix_T()
        R00 += T[0][0]
        R01 += T[0][1]
        R10 += T[1][0]
        R11 += T[1][1]

        R = [[R00, R01, R02], [R10, R11, R12]]


        J0 = np.zeros((ndof, len(cell2dof)), dtype=self.ftype)
        J1 = np.zeros((ndof, len(cell2dof)), dtype=self.ftype)

        Q0 = self.CM[:, 0, 0:ndof]/area[:, None] # (NC, ndof)
        start = cell2dofLocation[edge2cell[:, 0]] + edge2cell[:, 2]*p

        val = np.einsum('jm, j, j->mj', Q0[edge2cell[:, 0]], eh, n[:, 0])
        np.add.at(J0, (np.s_[:], start), val)

        val = np.einsum('jm, j, j->mj', Q0[edge2cell[:, 0]], eh, n[:, 1])
        np.add.at(J1, (np.s_[:], start), val)

        if np.any(isInEdge):
            start = cell2dofLocation[edge2cell[:, 1]] + edge2cell[:, 3]*p

            val = np.einsum('jm, j, j->mj', Q0[edge2cell[:, 1]], eh, n[:, 0])
            np.subtract.at(J0, (np.s_[:], start[isInEdge]), val[:, isInEdge])

            val = np.einsum('jm, j, j->mj', Q0[edge2cell[:, 1]], eh, n[:, 1])
            np.subtract.at(J1, (np.s_[:], start[isInEdge]), val[:, isInEdge])
        J = [J0, J1]

        return R, J

    def matrix_D(self):
        p = self.p

        mesh = self.mesh
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()
        NV = mesh.number_of_vertices_of_cells()
        edge = mesh.entity('edge')
        eh = mesh.entity_measure('edge')
        node = mesh.entity('node')
        edge2cell = mesh.ds.edge_to_cell()

        area = self.smspace.cellmeasure # 单元面积
        ch = self.smspace.cellsize # 单元尺寸 
        CM = self.CM

        smldof = self.smspace.number_of_local_dofs(p=p) # 标量 p 次单元缩放空间的维数
        ndof = (p-1)*p//2
        idof = (p-2)*(p-1)//2

        cell2dof = self.dof.cell2dof 
        cell2dofLocation = self.dof.cell2dofLocation # 标量的自由度信息

        D0 = np.zeros((len(cell2dof), smldof), dtype=self.ftype)
        D1 = np.zeros((NC, idof, smldof), dtype=self.ftype)
        D2 = np.zeros((NC, idof, smldof), dtype=self.ftype)

        qf = GaussLegendreQuadrature(p)
        bcs, ws = qf.quadpts, qf.weights
        ps = np.einsum('im, jmn->ijn', bcs, node[edge])
        phi0 = self.smspace.basis(ps, index=edge2cell[:, 0], p=p)
        phi = self.smspace.edge_basis(ps, p=p-1)
        F0 = np.einsum('i, ijm, ijn->jmn', ws, phi, phi0)
        idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
        np.add.at(D0, (idx, np.s_[:]), F0)

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        if np.any(isInEdge):
            phi1 = self.smspace.basis(ps, index=edge2cell[:, 1], p=p)
            F1 = np.einsum('i, ijm, ijn->jmn', ws, phi, phi1)
            idx = cell2dofLocation[edge2cell[:, [1]]] + edge2cell[:, [3]]*p + np.arange(p)
            np.add.at(D0, (idx[isInEdge], np.s_[:]), F1[isInEdge])

        if p > 2:
            idx = self.smspace.diff_index_1(p=p-2) # 一次求导后的非零基函数编号及求导系数
            x = idx['x']
            y = idx['y']
            D1 =  CM[:, y[0], :]/area[:, None, None] # check here
            D2 = -CM[:, x[0], :]/area[:, None, None] # check here

        return D0, D1, D2

    def matrix_U(self):
        """
        [[U00 U01 U02]
         [U10 U11 U12]
         [U20 U21 U22]]
        """
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()

        smldof = self.smspace.number_of_local_dofs(p=p-1) # 标量 p-1 次单元缩放空间的维数
        idof = (p-2)*(p-1)//2
        
        cell2dof = self.dof.cell2dof 
        cell2dofLocation = self.dof.cell2dofLocation # 标量的自由度信息

        U00 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        U01 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        U02 = np.zeros((NC, smldof, idof), dtype=self.ftype)

        U10 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        U11 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        U12 = np.zeros((NC, smldof, idof), dtype=self.ftype)

        U20 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        U21 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        U22 = np.zeros((NC, smldof, idof), dtype=self.ftype)

        idx = self.smspace.diff_index_1(p=p-1) # 一次求导后的非零基函数编号及求导系数
        x = idx['x']
        y = idx['y']
        ch = self.smspace.cellsize # 单元尺寸 
        ldofs = self.dof.number_of_local_dofs() # 这里只包含边上的自由度
        h = np.repeat(ch, ldofs)

        U00[x[0]] -= x[1][:, None]*self.E[0][0]/h
        U01[x[0]] -= x[1][:, None]*self.E[0][1]/h

        U10[y[0]] -= y[1][:, None]*self.E[0][0]/h
        U10[x[0]] -= x[1][:, None]*self.E[1][0]/h
        U11[y[0]] -= y[1][:, None]*self.E[0][1]/h
        U11[x[0]] -= x[1][:, None]*self.E[1][1]/h
    
        U20[y[0]] -= y[1][:, None]*self.E[1][0]/h
        U21[y[0]] -= y[1][:, None]*self.E[1][1]/h

        if p > 2:
            U02[:, x[0], :] -= x[1][None, :, None]*self.E[0][2]/ch[:, None, None]

            U12[:, y[0], :] -= y[1][None, :, None]*self.E[0][2]/ch[:, None, None]
            U12[:, x[0], :] -= x[1][None, :, None]*self.E[1][2]/ch[:, None, None]

            U22[:, y[0], :] -= y[1][None, :, None]*self.E[1][2]/ch[:, None, None]

        n = mesh.edge_unit_normal()
        eh = mesh.entity_measure('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights # bcs: (NQ, 2)
        ps = mesh.edge_bc_to_point(bcs)
        phi0 = self.smspace.basis(ps, index=edge2cell[:, 0], p=p-1) 
        phi = self.smspace.edge_basis(ps, p=p-1)
        F0 = np.einsum('i, ijm, ijn, j, j->jmn', ws, phi0, phi, eh, eh)
        F0 = F0@self.H1

        idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
        val = np.einsum('jmn, j-> mjn', F0, n[:, 0])
        np.add.at(U00, (np.s_[:], idx), val)
        np.add.at(U11, (np.s_[:], idx), val)
        val = np.einsum('jmn, j-> mjn', F0, n[:, 1])
        np.add.at(U10, (np.s_[:], idx), val)
        np.add.at(U21, (np.s_[:], idx), val)
        if np.any(isInEdge):
            phi1 = self.smspace.basis(ps, index=edge2cell[:, 1], p=p-1) 
            F1 = np.einsum('i, ijm, ijn, j, j->jmn', ws, phi1, phi, eh, eh)
            F1 = F1@self.H1
            idx = cell2dofLocation[edge2cell[:, [1]]] + edge2cell[:, [3]]*p + np.arange(p)
            val = np.einsum('jmn, j-> mjn', F1, n[:, 0])
            np.subtract.at(U00, (np.s_[:], idx[isInEdge]), val[:, isInEdge, :])
            np.subtract.at(U11, (np.s_[:], idx[isInEdge]), val[:, isInEdge, :])
            val = np.einsum('jmn, j-> mjn', F1, n[:, 1])
            np.subtract.at(U10, (np.s_[:], idx[isInEdge]), val[:, isInEdge, :])
            np.subtract.at(U21, (np.s_[:], idx[isInEdge]), val[:, isInEdge, :])

        return [[U00, U01, U02], [U10, U11, U12], [U20, U21, U22]]

    def matrix_A(self):
        return self.stiff_matrix()

    def stiff_matrix(self):
        p = self.p # 空间次数
        cell2dof = self.dof.cell2dof
        cell2dofLocation = self.dof.cell2dofLocation
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = self.mesh.number_of_cells()
        cell, cellLocation = mesh.entity('cell')
        cell2edge = mesh.ds.cell_to_edge()

        smldof = self.smspace.number_of_local_dofs(p=p)

        eh = mesh.entity_measure('edge')
        ch = self.smspace.cellsize
        area = self.smspace.cellmeasure

        def f1(i):
            s0 = slice(cell2dofLocation[i], cell2dofLocation[i+1])
            PI0 = self.PI0[i]
            D = np.eye(PI0.shape[1])
            D0 = self.D[0][s0, :]
            D1 = self.D[1][i]
            D2 = self.D[2][i]
            D -= block([
                [D0,  0],
                [ 0, D0],
                [D1, D2]])@PI0[:2*smldof]

            s1 = slice(cellLocation[i], cellLocation[i+1]) 
            S = list(self.H1[cell2edge[s1]]*eh[cell2edge[s1]][:, None, None])
            S += S
            if p > 2:
                S.append(inv(self.Q[i])*area[i])
            S = D.T@block_diag(S)@D
            U = block([
                [self.U[0][0][:, s0], self.U[0][1][:, s0], self.U[0][2][i]],
                [self.U[1][0][:, s0], self.U[1][1][:, s0], self.U[1][2][i]],
                [self.U[2][0][:, s0], self.U[2][1][:, s0], self.U[2][2][i]]])
            H0 = block([
                [self.H0[i],              0,          0],
                [         0, 0.5*self.H0[i],          0],
                [         0,              0, self.H0[i]]])
            return S + U.T@H0@U 

        A = list(map(f1, range(NC)))
        idof = (p-2)*(p-1)//2
        def f2(i):
            s = slice(cell2dofLocation[i], cell2dofLocation[i+1])
            cd = np.r_[cell2dof[s], NE*p + cell2dof[s], 2*NE*p + np.arange(i*idof, (i+1)*idof)]
            return np.meshgrid(cd, cd)
        
        idx = list(map(f2, range(NC)))
        I = np.concatenate(list(map(lambda x: x[1].flat, idx)))
        J = np.concatenate(list(map(lambda x: x[0].flat, idx)))

        gdof = self.number_of_global_dofs() 

        A = np.concatenate(list(map(lambda x: x.flat, A)))
        A = csr_matrix((A, (I, J)), shape=(gdof, gdof), dtype=self.ftype)
        return  A

    def matrix_P(self):
        return self.div_matrix()

    def div_matrix(self):
        """
        Notes
        -----

        (div v_h, q_0)
        """
        p = self.p
        idof = (p-2)*(p-1)//2
        cell2dof = self.dof.cell2dof
        cell2dofLocation =  self.dof.cell2dofLocation
        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells()
        cd = self.smspace.cell_to_dof(p=0)
        def f0(i):
            s = slice(cell2dofLocation[i], cell2dofLocation[i+1])
            return np.meshgrid(cell2dof[s], cd[i])
        idx = list(map(f0, range(NC)))
        I = np.concatenate(list(map(lambda x: x[1].flat, idx)))
        J = np.concatenate(list(map(lambda x: x[0].flat, idx)))

        gdof0 = self.smspace.number_of_global_dofs(p=0)

        if False:
            def f1(i, k):
                J = self.J[k][0, cell2dofLocation[i]:cell2dofLocation[i+1]]
                return J
            P0 = np.concatenate(list(map(lambda i: f1(i, 0), range(NC))))
            P1 = np.concatenate(list(map(lambda i: f1(i, 1), range(NC))))

        P0 = csr_matrix((self.J[0][0], (I, J)),
                shape=(gdof0, NE*p), dtype=self.ftype)
        P1 = csr_matrix((self.J[1][0], (I, J)),
                shape=(gdof0, NE*p), dtype=self.ftype)
        P2 = csr_matrix((gdof0, NC*idof), dtype=self.ftype)

        return bmat([[P0, P1, P2]], format='csr')
            

    def source_vector(self, f):
        p = self.p
        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells()
        if p == 2:
            ndof = self.smspace.number_of_local_dofs(p=p)
            def u0(x, index):
                return np.einsum('ijm, ijn->ijmn', 
                        self.smspace.basis(x, index=index, p=p), f(x))
            bb = self.integralalg.integral(u0, celltype=True) # (NC, ndof, 2)
            cell2dof = self.dof.cell2dof
            cell2dofLocation = self.dof.cell2dofLocation
            eb = np.zeros((2, len(cell2dof)), dtype=self.ftype)
            def u1(i):
                PI0 = self.PI0[i]
                n = PI0.shape[1]//2
                s = slice(cell2dofLocation[i], cell2dofLocation[i+1])
                eb[0, s] = bb[i, :, 0]@PI0[:ndof, :n] + bb[i, :, 1]@PI0[ndof:2*ndof, :n]
                eb[1, s] = bb[i, :, 0]@PI0[:ndof, n:] + bb[i, :, 1]@PI0[ndof:2*ndof, n:]
            list(map(u1, range(NC)))
            gdof = self.number_of_global_dofs()
            b = np.zeros((gdof, ), dtype=self.ftype)
            np.add.at(b, cell2dof, eb[0])
            np.add.at(b[NE*p:], cell2dof, eb[1])
            return b
        else:
            ndof = self.smspace.number_of_local_dofs(p=p-2)
            Q = inv(self.CM[:, :ndof, :ndof])
            phi = lambda x: self.smspace.basis(x, p=p-2)
            def u0(x, index):
                return np.einsum('ijm, ijn->ijmn', 
                        self.smspace.basis(x, index=index, p=p-2), f(x))
            bb = self.integralalg.integral(u0, celltype=True) # (NC, ndof, 2)
            bb = Q@bb # (NC, ndof, 2)

            NE = self.mesh.number_of_edges()
            NC = self.mesh.number_of_cells()
            cell2dof = self.dof.cell2dof
            cell2dofLocation = self.dof.cell2dofLocation

            eb = np.zeros((2, len(cell2dof)), dtype=self.ftype)
            def u1(i):
                s = slice(cell2dofLocation[i], cell2dofLocation[i+1])
                eb[0, s] = bb[i, :, 0]@self.E[0][0][:, s] + bb[i, :, 1]@self.E[1][0][:, s]
                eb[1, s] = bb[i, :, 0]@self.E[0][1][:, s] + bb[i, :, 1]@self.E[1][1][:, s]
            list(map(u1, range(NC)))
            gdof = self.number_of_global_dofs()
            b = np.zeros((gdof, ), dtype=self.ftype)

            np.add.at(b, cell2dof, eb[0])
            np.add.at(b[NE*p:], cell2dof, eb[1])
            c2d = self.cell_to_dof('cell')
            b[c2d] += np.sum(bb[:, :, [0]]*self.E[0][2], axis=1)
            b[c2d] += np.sum(bb[:, :, [1]]*self.E[1][2], axis=1)
            return b 

    def to_rtspace(self, uh0, uh1, q=None):

        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells()
        p = self.p
        space0 = self
        space1 = uh1.space
        A = self.interpolation_RT(space1, q=q)

        ldof0 = space0.number_of_local_dofs()[0]
        c2d0 = np.zeros((NC, ldof0), dtype=self.itype)
        c2d, cell2dofLocation = space0.cell_to_dof()

        c2d0[:, 0:3*p] = c2d.reshape(-1, 3*p)
        c2d0[:, 3*p:6*p] = c2d0[:, 0:3*p] + NE*p
        c2d0[:, 6*p:] = space0.cell_to_dof('cell') 

        c2d1 = space1.cell_to_dof()
        uh1[c2d1] = np.einsum('cij, cj->ci', A, uh0[c2d0])


    def interpolation_RT(self, space, q=None):
        """

        Notes
        -----
        把 Reduced 虚单元空间的中基函数插值到 RT_{k-1} 空间中.

        这里假设所用的多边形网格是三角形网格, 

        给定一个三角形网格, 转化为多边形网格, 转化时要注意三角形的第 0
        条边, 转化为多边形网格的第 0 条边.

        self 是 Reduced 虚单元的空间
        space 是 RT 空间的 space
     
        """

        p = self.p 
        mesh = space.mesh # 多边形网格
        GD = mesh.geo_dimension() # GD == 2
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        en = mesh.edge_unit_normal() # 每条边的单位法向(NE, 2)
        eh = self.mesh.entity_measure('edge') # 每条边的长度 (NE, )


        # RT_{k-1} 空间在单元 K 上局部自由度的个数, TODO：确认是 k-1 还是 k
        ldof0 = space.number_of_local_dofs(doftype='all') 
        print('ldof', ldof0)

        
        # 每个单元 K 上的插值矩阵
        A = np.zeros((NC, ldof0, ldof0), dtype=self.ftype)

        # 三角形网格边上的积分公式
        qf = space.integralalg.edgeintegrator if q is None else mesh.integrator(q, 'edge')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = mesh.bc_to_point(bcs)

        # 网格边上的缩放单项式空间的基函数 (NQ, NE, p)
        ephi = self.smspace.edge_basis(ps, p=p-1)

        # 计算左边单元基函数在当前边上积分点处的函数值, (NQ, NE, ldof0, 2)
        phi = space.basis(ps, index=edge2cell[:, 0], barycentric=False) 
        idx0 = edge2cell[:, 0][:, None]
        idx1 = edge2cell[:, 2][:, None]*p + np.arange(p)
        # (NE, p, ldof0)
        A[idx0, idx1] = np.einsum('q, qei, em, qejm, e->eij', ws, ephi, en, phi, eh)  

        # 计算右边单元基函数在当前边上积分点处的函数值, (NQ, NE, ldof0, 2)
        phi = space.basis(ps, index=edge2cell[:, 1], barycentric=False) 
        idx0 = edge2cell[:, 1][:, None]
        idx1 = edge2cell[:, 3][:, None]*p + np.arange(p)
        # (NE, p, ldof0)
        A[idx0, idx1] = np.einsum('q, qei, ed, qejd, e->eij', ws, ephi, en, phi, eh)  

        # 三角形网格单元上的积分公式
        cellmeasure = mesh.entity_measure('cell')
        qf = space.integralalg.cellintegrator if q is None else mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = mesh.bc_to_point(bcs)
        phi0 = space.basis(bcs) # (NQ, NC, ldof0, GD)
        phi1 = self.smspace.basis(ps, p=p-2) # (NQ, NC, ldof1)
        val = np.einsum('q, qci, qcjm, c->cmij', ws, phi1, phi0, cellmeasure)
        ldof = self.smspace.number_of_local_dofs(p=p-2, doftype='cell')
        start = 3*p
        stop = start + ldof
        A[:, start:stop, :] = val[:, 0]
        start = stop
        stop += ldof
        A[:, start:stop, :] = val[:, 1]

        # Reduced 虚单元空间在单元  K 上局部自由度的个数
        ldof1 = self.number_of_local_dofs()[0]
        print('ldof1', ldof1)
        # 每个单元 K 上的右端矩阵
        F = np.zeros((NC, ldof0, ldof1), dtype=self.ftype) 
        idx0 = edge2cell[:, [2]]*p + np.arange(p)
        F[edge2cell[:, 0][:, None], idx0, idx0] = (en[:, 0]*eh)[:, None]
        idx1 = 3*p + idx0
        F[edge2cell[:, 0][:, None], idx0, idx1] = (en[:, 1]*eh)[:, None] 

        idx0 = edge2cell[:, [3]]*p + np.arange(p)
        F[edge2cell[:, 1][:, None], idx0, idx0] = (en[:, 0]*eh)[:, None]
        idx1 = 3*p + idx0
        F[edge2cell[:, 1][:, None], idx0, idx1] = (en[:, 1]*eh)[:, None]
        
        E = self.E
        E00 = np.hsplit(E[0][0], NC)
        E01 = np.hsplit(E[0][1], NC)
        E02 = E[0][2]
        E10 = np.hsplit(E[1][0], NC)
        E11 = np.hsplit(E[1][1], NC)
        E12 = E[1][2]

        smldof = self.smspace.number_of_local_dofs(p=p-2) # 标量 p-2 次单元缩放空间的维数
        start = 3*p
        stop = start + smldof
        F[:, start:stop, 0*p:3*p] = E00
        F[:, start:stop, 3*p:6*p] = E01
        F[:, start:stop, 6*p:] = E02

        start = stop
        stop += smldof
        F[:, start:stop, 0*p:3*p] = E10
        F[:, start:stop, 3*p:6*p] = E11
        F[:, start:stop, 6*p:] = E12

        return inv(A)@F 

    def pressure_robust_source_vector(self, f, space, q=None):
        """

        Note
        ----
        假设网格是三角形网格, 把缩减虚单元空间的基函数投影到 RT_{k-1} 空间

        space 是 k-1 次的 RT 空间的 space
        """
        p = self.p
        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells()

        ldof0 = space.number_of_local_dofs(doftype='all')  
        ldof1 = self.number_of_local_dofs()[0] # 因为假设是三角形, 所有单元自由度个数是相同的
        # 插值矩阵 I.shape == (NC, ldof0, ldof1)
        I = self.interpolation_RT(space, q=q)

        # 向量函数 f 在 RT 空间中的离散, (NC, ldof0)
        # celltype = True, 表示只计算每个单元上的右端
        bb = space.source_vector(f, celltype=True, q=q)
        bb = (bb[:, None, :]@I).reshape(NC, -1)

        gdof = self.number_of_global_dofs()
        b = np.zeros((gdof, ), dtype=self.ftype)

        cell2dof = self.dof.cell2dof # 边上的自由度
        np.add.at(b, cell2dof, bb[:, 0:3*p].flat)
        np.add.at(b[NE*p:], cell2dof, bb[:, 3*p:6*p].flat)

        c2d = self.cell_to_dof('cell')
        b[c2d] += bb[:, 6*p:]

        return b

    def set_dirichlet_bc(self, uh, gd, is_dirichlet_edge=None):
        """
        
        """
        p = self.p
        mesh = self.mesh

        NE = mesh.number_of_edges()

        isBdEdge = mesh.ds.boundary_edge_flag()
        edge2dof = self.dof.edge_to_dof()

        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = mesh.edge_bc_to_point(bcs, index=isBdEdge)
        val = gd(ps)

        ephi = self.smspace.edge_basis(ps, index=isBdEdge, p=p-1)
        b = np.einsum('i, ij..., ijk->jk...', ws, val, ephi)
        # T = self.H1[isBdEdge]@b #TODO: 检查这里的问题
        uh[edge2dof[isBdEdge]] = b[:, :, 0] 
        uh[NE*p:][edge2dof[isBdEdge]] = b[:, :, 1] 

    def number_of_global_dofs(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        p = self.p
        gdof = 2*NE*p
        if p > 2:
            gdof += NC*(p-2)*(p-1)//2
        return gdof

    def number_of_local_dofs(self):
        p = self.p
        mesh = self.mesh
        NCE = mesh.number_of_edges_of_cells()
        ldofs = 2*NCE*p
        if p > 2:
            ldofs += (p-2)*(p-1)//2
        return ldofs

    def cell_to_dof(self, doftype='all'):
        if doftype == 'all':
            return self.dof.cell2dof, self.dof.cell2dofLocation
        elif doftype == 'cell':
            p = self.p
            NE = self.mesh.number_of_edges()
            NC = self.mesh.number_of_cells()
            idof = (p-2)*(p-1)//2
            cell2dof = 2*NE*p + np.arange(NC*idof).reshape(NC, idof)
            return cell2dof
        elif doftype == 'edge':
            return self.dof.edge_to_dof()

    def boundary_dof(self):
        return self.dof.boundary_dof()

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim in {None, 1}:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=self.ftype)
