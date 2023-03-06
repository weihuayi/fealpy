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

class DFNCVEMDof2d():
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

class DivFreeNonConformingVirtualElementSpace2d:
    def __init__(self, mesh, p, q=None):
        """
        Parameter
        ---------
        mesh : polygon mesh
        p : the space order, p>=2
        """
        self.p = p
        self.smspace = ScaledMonomialSpace2d(mesh, p, q=q)
        self.mesh = mesh
        self.dof = DFNCVEMDof2d(mesh, p) # 注意这里只是边上的标量的自由度管理对 象
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
        self.R, self.J = self.matrix_R_J()
        self.U = self.matrix_U()
        self.D = self.matrix_D()
        self.PI0 = self.matrix_PI0()

    def project(self, u):
        p = self.p # p >= 2
        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells()
        uh = self.function()
        def u0(x):
            return np.einsum('ij..., ijm->ijm...', u(x),
                    self.smspace.edge_basis(x, p=p-1))
        eh = self.mesh.entity_measure('edge')
        uh0 = self.integralalg.edge_integral(
                u0, edgetype=True)/eh[:, None, None]
        uh[0:2*NE*p] = uh0.reshape(-1, 2).T.flat

        c2d = self.cell_to_dof('cell') # 单元与内部自由度的对应关系

        idof0 = (p+1)*p//2 - 1 # G_{k-2}^\nabla 梯度空间对应的自由度个数
        def u1(x, index):
            return np.einsum('ijn, ijmn->ijm', u(x), 
                    self.smspace.grad_basis(x, p=p-1, index=index)[:, :, 1:, :])
        ch = self.smspace.cellsize
        # h_K\nabla m_1, h_K\nabla m_2, ......, h_K\nabla m_{n_k -1}
        uh[c2d[:, 0:idof0]] = self.integralalg.integral(u1, celltype=True)/ch[:, None]
        if p > 2:
            # G_{k-2}^\perp 空间对应的自由度
            idx = self.smspace.diff_index_1(p=p-2) # 一次求导后的非零基函数编号及求导系数
            x = idx['x']
            y = idx['y']
            def u2(x, index):
                return np.einsum('ij..., ijm->ijm...', u(x),
                        self.smspace.basis(x, p=p-2, index=index))
            area = self.smspace.cellmeasure
            uh2 = self.integralalg.integral(u2, celltype=True)/area[:, None, None]
            uh[c2d[:, idof0:]] += uh2[:, y[0], 0]
            uh[c2d[:, idof0:]] -= uh2[:, x[0], 1]
        return uh

    def project_to_smspace(self, uh):
        p = self.p
        idof = p*(p-1) 
        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells()
        cell2dof = self.dof.cell2dof
        cell2dofLocation = self.dof.cell2dofLocation
        smldof = self.smspace.number_of_local_dofs(p=p)
        sh = self.smspace.function(dim=2)
        c2d = self.smspace.cell_to_dof()
        def f(i):
            PI0 = self.PI0[i]
            s0 = slice(cell2dofLocation[i], cell2dofLocation[i+1])
            D0 = self.D[0][s0, :]
            D1 = self.D[1][i, 0]
            D2 = self.D[1][i, 1]

            D = block([
                [D0,  0],
                [ 0, D0],
                [D1, D2]])


            idx = cell2dof[cell2dofLocation[i]:cell2dofLocation[i+1]]
            x0 = uh[idx]
            x1 = uh[idx+NE*p]
            x2 = np.zeros(idof, dtype=self.ftype)
            start = 2*NE*p + i*idof
            x2[:] = uh[start:start+idof]
            x = np.r_[x0, x1, x2]
            y = (PI0[:2*smldof]@x).flat
            sh[c2d[i], 0] = y[:smldof]
            sh[c2d[i], 1] = y[smldof:]
        list(map(f, range(NC)))
        return sh

    def matrix_PI0(self):
        p = self.p
        NC = self.mesh.number_of_cells()
        cell2dof = self.dof.cell2dof
        cell2dofLocation = self.dof.cell2dofLocation
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
                [   self.J[0][:, s],    self.J[1][:, s],    self.J[2][i]]])
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
        cell2dof = self.dof.cell2dof
        cell2dofLocation = self.dof.cell2dofLocation # 标量的单元边自由度信息

        CM = self.CM # 单元质量矩阵

        # 构造分块矩阵 
        #  R = [
        #       [R00, R01, R02], 
        #       [R10, R11, R12]
        #  ]

        idof = p*(p-1) # 内部自由度的个数
        idof0 = (p+1)*p//2 - 1 # G_{k-2}^\nabla


        R00 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        R01 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        R02 = np.zeros((NC, smldof, idof), dtype=self.ftype)

        R10 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        R11 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        R12 = np.zeros((NC, smldof, idof), dtype=self.ftype)

        idx1 = self.smspace.diff_index_1(p=p-1)
        x = idx1['x']
        y = idx1['y']
        idx2 = self.smspace.diff_index_2(p=p)
        xx = idx2['xx']
        yy = idx2['yy']
        xy = idx2['xy']
        c = 1.0/np.repeat(range(1, p), range(1, p))

        R02[:, xx[0], x[0]-1] -= xx[1]*c
        R02[:, yy[0], x[0]-1] -= 0.5*yy[1]*c
        R02[:, xy[0], y[0]-1] -= 0.5*xy[1]*c

        R12[:, yy[0], y[0]-1] -= yy[1]*c
        R12[:, xx[0], y[0]-1] -= 0.5*xx[1]*c
        R12[:, xy[0], x[0]-1] -= 0.5*xy[1]*c

        if p > 2:
            idx = np.arange(idof0, idof)
            idx0 = self.smspace.diff_index_1(p=p-2)
            x0 = idx0['x']
            y0 = idx0['y']

            R02[:, xx[0][y0[0]], idx] -= xx[1][y0[0]]*y0[1]*c[y0[0]] 
            R02[:, yy[0][y0[0]], idx] -= 0.5*yy[1][y0[0]]*y0[1]*c[y0[0]]
            R02[:, xy[0][x0[0]], idx] += 0.5*xy[1][x0[0]]*x0[1]*c[x0[0]]
            
            R12[:, yy[0][x0[0]], idx] += yy[1][x0[0]]*x0[1]*c[x0[0]] 
            R12[:, xx[0][x0[0]], idx] += 0.5*xx[1][x0[0]]*x0[1]*c[x0[0]]
            R12[:, xy[0][y0[0]], idx] -= 0.5*xy[1][y0[0]]*y0[1]*c[y0[0]]


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

        idx = self.smspace.diff_index_1(p=p) # 一次求导后的非零基函数编号及求导系数
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


        R02[:, :, 0] += CM[:, :, 0]/area[:, None]
        R12[:, :, 1] += CM[:, :, 0]/area[:, None]

        R = [[R00, R01, R02], [R10, R11, R12]]

        idx0 = self.smspace.diff_index_1(p=p-1)
        x0 = idx0['x']
        y0 = idx0['y']
        J0 = np.zeros((ndof, len(cell2dof)), dtype=self.ftype)
        J1 = np.zeros((ndof, len(cell2dof)), dtype=self.ftype)
        J2 = np.zeros((NC, ndof, idof), dtype=self.ftype)

        J2[:, x0[0], x0[0] - 1] -= ch[:, None]*x0[1]*c
        J2[:, y0[0], y0[0] - 1] -= ch[:, None]*y0[1]*c
        if p > 2:
            idx = np.arange(idof0, idof)
            idx1 = self.smspace.diff_index_1(p=p-2)
            x1 = idx1['x']
            y1 = idx1['y']
            J2[:, x0[0][y1[0]], idx] -= ch[:, None]*x0[1][y1[0]]*y1[1]*c[y1[0]]
            J2[:, y0[0][x1[0]], idx] += ch[:, None]*y0[1][x1[0]]*x1[1]*c[x1[0]]

        idx0 = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
        val = np.einsum('ijk, i->jik', F0, n[:, 0])
        np.add.at(J0, (np.s_[:], idx0), val)
        val = np.einsum('ijk, i->jik', F0, n[:, 1])
        np.add.at(J1, (np.s_[:], idx0), val)

        if isInEdge.sum() > 0:
            idx0 = cell2dofLocation[edge2cell[:, [1]]] + edge2cell[:, [3]]*p + np.arange(p)
            val = np.einsum('ijk, i->jik', F1, n[:, 0])
            np.subtract.at(J0, (np.s_[:], idx0[isInEdge]), val[:, isInEdge])

            val = np.einsum('ijk, i->jik', F1, n[:, 1])
            np.subtract.at(J1, (np.s_[:], idx0[isInEdge]), val[:, isInEdge])

        J = [J0, J1, J2]

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

        cell2dof = self.dof.cell2dof 
        cell2dofLocation = self.dof.cell2dofLocation # 标量的自由度信息

        idof = p*(p-1) # 内部自由度的个数
        # G_{k-2}^\nabla 空间自由度的个数, 注意它的基函数 $h_K\nabla \bfm_{k-1}$
        idof0 = (p+1)*p//2 - 1 

        D0 = np.zeros((len(cell2dof), smldof), dtype=self.ftype)

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

        D1 = np.zeros((NC, 2, idof, smldof), dtype=self.ftype)
        def u0(x, index):
            return np.einsum('ijmk, ijn->ijkmn', 
                    self.smspace.grad_basis(x, p=p-1, index=index)[:, :, 1:, :],
                    self.smspace.basis(x, p=p, index=index))
        D1[:, :, :idof0, :] = self.integralalg.integral(u0, celltype=True)/ch[:, None, None, None]

        if p > 2:
            idx = self.smspace.diff_index_1(p=p-2) # 一次求导后的非零基函数编号及求导系数
            x = idx['x']
            y = idx['y']
            D1[:, 0, idof0:, :] =  CM[:, y[0], :]/area[:, None, None] # check here
            D1[:, 1, idof0:, :] = -CM[:, x[0], :]/area[:, None, None] # check here

        return D0, D1

    def matrix_U(self):
        """
        [[U00 U01 U02]
         [U10 U11 U12]
         [U20 U21 U22]]
        """
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()

        ch = self.smspace.cellsize # 单元尺寸 

        smldof = self.smspace.number_of_local_dofs(p=p-1) # 标量 p-1 次单元缩放空间的维数
        idof = p*(p-1)  
        idof0 = (p+1)*p//2-1
        
        cell2dof = self.dof.cell2dof 
        cell2dofLocation = self.dof.cell2dofLocation # 多边形边上的标量自由度信息

        U00 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        U01 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        U02 = np.zeros((NC, smldof, idof), dtype=self.ftype)

        U10 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        U11 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        U12 = np.zeros((NC, smldof, idof), dtype=self.ftype)

        U20 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        U21 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        U22 = np.zeros((NC, smldof, idof), dtype=self.ftype)

        idx0 = self.smspace.diff_index_1(p=p-1) # 一次求导后的非零基函数编号及求导系数
        x0 = idx0['x']
        y0 = idx0['y']
        c = 1.0/np.repeat(range(1, p), range(1, p))

        U02[:, x0[0], x0[0] - 1] -= x0[1]*ch[:, None]*c

        U12[:, y0[0], x0[0] - 1] -= y0[1]*ch[:, None]*c
        U12[:, x0[0], y0[0] - 1] -= x0[1]*ch[:, None]*c 

        U22[:, y0[0], y0[0] - 1] -= y0[1]*ch[:, None]*c

        if p > 2:
            idx = np.arange(idof0, idof)
            idx1 = self.smspace.diff_index_1(p=p-2) # 一次求导后的非零基函数编号及求导系数
            x1 = idx1['x']
            y1 = idx1['y']
            U02[:, x0[0][y1[0]], idx] -= ch[:, None]*c[y1[0]]*x0[1][y1[0]]*y1[1]

            U12[:, y0[0][y1[0]], idx] -= ch[:, None]*c[y1[0]]*y0[1][y1[0]]*y1[1]
            U12[:, x0[0][x1[0]], idx] += ch[:, None]*c[x1[0]]*x0[1][x1[0]]*x1[1]

            U22[:, y0[0][x1[0]], idx] += ch[:, None]*c[x1[0]]*y0[1][x1[0]]*x1[1]

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

    def stiff_matrix(self, celltype=False):
        """
        """
        p = self.p # 空间次数
        idof0 = (p+1)*p//2-1 # k-1 次多项式空间的梯度空间
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
        Z0 = np.zeros((idof0, idof0), dtype=self.ftype)
        def f1(i):
            s0 = slice(cell2dofLocation[i], cell2dofLocation[i+1])
            PI0 = self.PI0[i]
            D = np.eye(PI0.shape[1])
            D0 = self.D[0][s0, :]
            D1 = self.D[1][i, 0]
            D2 = self.D[1][i, 1]
            D -= block([
                [D0,  0],
                [ 0, D0],
                [D1, D2]])@PI0[:2*smldof]

            s1 = slice(cellLocation[i], cellLocation[i+1]) 
            S = list(self.H1[cell2edge[s1]]*eh[cell2edge[s1]][:, None, None])
            S += S
            S.append(Z0)
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

        if celltype:
            return A
        
        idof = p*(p-1) #
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
        idof = p*(p-1)
        gdof = self.smspace.number_of_global_dofs(p=p-1)
        cell2dof, cell2dofLocation = self.cell_to_dof(doftype='all')
        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells()
        cd = self.smspace.cell_to_dof(p=p-1)
        def f0(i):
            s = slice(cell2dofLocation[i], cell2dofLocation[i+1])
            return np.meshgrid(cell2dof[s], cd[i])
        idx = list(map(f0, range(NC)))
        I = np.concatenate(list(map(lambda x: x[1].flat, idx)))
        J = np.concatenate(list(map(lambda x: x[0].flat, idx)))

        def f1(i, k):
            J = self.J[k][cell2dofLocation[i]:cell2dofLocation[i+1]]
            return J.flat
        J0 = np.concatenate(list(map(lambda x: f1(x, 0), range(NC))))
        J1 = np.concatenate(list(map(lambda x: f1(x, 1), range(NC))))
        P0 = csr_matrix((J0, (I, J)),
                shape=(gdof, NE*p), dtype=self.ftype)
        P1 = csr_matrix((J1, (I, J)),
                shape=(gdof, NE*p), dtype=self.ftype)
        cell2dof = np.arange(NC*idof).reshape(NC, idof)
        def f2(i):
            return np.meshgrid(cell2dof[i], cd[i])
        idx = list(map(f2, range(NC)))
        I = np.concatenate(list(map(lambda x: x[1].flat, idx)))
        J = np.concatenate(list(map(lambda x: x[0].flat, idx)))
        P2 = csr_matrix((self.J[2].flat, (I, J)), 
                shape=(gdof, NC*idof), dtype=self.ftype)

        return bmat([[P0, P1, P2]], format='csr')
            

    def cell_grad_source_vector(self, f):
        """

        Notes
        -----
            计算单元载荷向量, 这里只计算梯度空间对应的部分
        """
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        if p == 2:
            ndof = self.smspace.number_of_local_dofs(p=p)
            cell2dof = self.dof.cell2dof 
            cell2dofLocation = self.dof.cell2dofLocation
            idof = p*(p-1)
            idof0 = (p+1)*p//2 - 1
            def u0(x, index):
                return np.einsum('ijm, ijn->ijmn', 
                        self.smspace.basis(x, index=index, p=p), f(x))
            bb = self.integralalg.integral(u0, celltype=True) # (NC, ndof, 2)
            b = np.zeros((NC, idof0), dtype=self.ftype)
            def u1(i):
                PI0 = self.PI0[i]
                n = cell2dofLocation[i+1] - cell2dofLocation[i] 
                b[i, :] = bb[i, :, 0]@PI0[:ndof, 2*n:2*n+idof0] + bb[i, :,
                        1]@PI0[ndof:2*ndof, 2*n:2*n+idof0]
            list(map(u1, range(NC)))
            return b # (NC, idof0)
        else:
            area = self.smspace.cellmeasure
            ndof = self.smspace.number_of_local_dofs(p=p-2)
            idof0 = (p+1)*p//2 - 1

            c = np.repeat(range(1, p), range(1, p))
            idx = self.smspace.diff_index_1(p=p-1)
            x = idx['x']
            y = idx['y']
            idx0 = np.arange(ndof)
            E00 = np.zeros((NC, ndof, idof0), dtype=self.ftype)
            E10 = np.zeros((NC, ndof, idof0), dtype=self.ftype)
            E00[:, idx0, x[0]-1] = area[:, None]/c
            E10[:, idx0, y[0]-1] = area[:, None]/c

            Q = inv(self.CM[:, :ndof, :ndof])
            phi = lambda x: self.smspace.basis(x, p=p-2)
            def u0(x, index):
                return np.einsum('ijm, ijn->ijmn', 
                        self.smspace.basis(x, index=index, p=p-2), f(x))
            bb = self.integralalg.integral(u0, celltype=True) # (NC, ndof, 2)
            bb = Q@bb # (NC, ndof, 2)
            b = bb[:, :, 0][:, None]@E00 + bb[:, :, 1][:, None]@E10 
            return b.reshape(-1, idof0) # (NC, idof0)

    def source_vector(self, f):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        if p == 2:
            ndof = self.smspace.number_of_local_dofs(p=p)
            idof = p*(p-1)
            def u0(x, index):
                return np.einsum('ijm, ijn->ijmn', 
                        self.smspace.basis(x, index=index, p=p), f(x))
            bb = self.integralalg.integral(u0, celltype=True) # (NC, ndof, 2)
            cell2dof = self.dof.cell2dof # 这里只有边上的自由度
            cell2dofLocation = self.dof.cell2dofLocation
            eb = np.zeros((2, len(cell2dof)), dtype=self.ftype)
            cb = np.zeros((NC, idof), dtype=self.ftype)
            def u1(i):
                PI0 = self.PI0[i]
                n = cell2dofLocation[i+1] - cell2dofLocation[i] 
                s = slice(cell2dofLocation[i], cell2dofLocation[i+1])
                eb[0, s] = bb[i, :, 0]@PI0[:ndof, :n] + bb[i, :, 1]@PI0[ndof:2*ndof, :n]
                eb[1, s] = bb[i, :, 0]@PI0[:ndof, n:2*n] + bb[i, :, 1]@PI0[ndof:2*ndof, n:2*n]
                cb[i, :] = bb[i, :, 0]@PI0[:ndof, 2*n:] + bb[i, :, 1]@PI0[ndof:2*ndof, 2*n:]
            list(map(u1, range(NC)))
            gdof = self.number_of_global_dofs()
            b = np.zeros((gdof, ), dtype=self.ftype)
            np.add.at(b, cell2dof, eb[0])
            np.add.at(b[NE*p:], cell2dof, eb[1])
            c2d = self.cell_to_dof(doftype='cell')
            np.add.at(b, c2d, cb)
            return b
        else:
            area = self.smspace.cellmeasure
            ndof = self.smspace.number_of_local_dofs(p=p-2)
            Q = inv(self.CM[:, :ndof, :ndof])
            phi = lambda x: self.smspace.basis(x, p=p-2)
            def u0(x, index):
                return np.einsum('ijm, ijn->ijmn', 
                        self.smspace.basis(x, index=index, p=p-2), f(x))
            bb = self.integralalg.integral(u0, celltype=True) # (NC, ndof, 2)
            bb = Q@bb # (NC, ndof, 2)

            idx0 = self.smspace.diff_index_1(p=p-2)
            x0 = idx0['x']
            y0 = idx0['y']
            idx1 = self.smspace.diff_index_1(p=p-1)
            x1 = idx1['x']
            y1 = idx1['y']
            c = 1.0/np.repeat(range(1, p), range(1, p))
            c2d = self.cell_to_dof('cell')

            # TODO: check here, something is wrong!
            idof0 = (p+1)*p//2 - 1 
            gdof = self.number_of_global_dofs()
            b = np.zeros(gdof, dtype=self.ftype)
            idx = np.arange(idof0)
            b[c2d[:, idx[x1[0]-1]]] += bb[:, :, 0]*area[:, None]*c 
            b[c2d[:, idx[y1[0]-1]]] += bb[:, :, 1]*area[:, None]*c 

            idx = np.arange(idof0, p*(p-1))
            b[c2d[:, idx]] += bb[:, y0[0], 0]*area[:, None]*y0[1]*c[y0[0]]
            b[c2d[:, idx]] -= bb[:, x0[0], 1]*area[:, None]*x0[1]*c[x0[0]]
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
        # T = self.H1[isBdEdge]@b # 这里是一个 bug, 为什么要做一个 L2 投影呢? 直
        # 接算自由度的值即可
        uh[edge2dof[isBdEdge]] = b[:, :, 0] 
        uh[NE*p:][edge2dof[isBdEdge]] = b[:, :, 1] 

    def number_of_global_dofs(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        p = self.p
        gdof = 2*NE*p
        gdof += NC*p*(p-1)
        return gdof

    def number_of_local_dofs(self):
        p = self.p
        mesh = self.mesh
        NCE = mesh.number_of_edges_of_cells()
        ldofs = 2*NCE*p
        if p > 2:
            ldofs += p*(p-1)
        return ldofs

    def cell_to_dof(self, doftype='all'):
        if doftype == 'all':
            return self.dof.cell2dof, self.dof.cell2dofLocation
        elif doftype == 'cell':
            p = self.p
            NE = self.mesh.number_of_edges()
            NC = self.mesh.number_of_cells()
            idof = p*(p-1) 
            cell2dof = 2*NE*p + np.arange(NC*idof).reshape(NC, idof)
            return cell2dof
        elif doftype == 'edge':
            return self.dof.edge_to_dof()

    def boundary_dof(self):
        p = self.p
        NE = self.mesh.number_of_edges()
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        edge2dof = self.dof.edge_to_dof()
        isBdEdge = self.mesh.ds.boundary_edge_flag()
        isBdDof[edge2dof[isBdEdge]] = True
        isBdDof[NE*p+edge2dof[isBdEdge]] = True
        return isBdDof

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
