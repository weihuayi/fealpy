
import numpy as np
from numpy.linalg import inv
from scipy.sparse import bmat, coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from .function import Function
from .ScaledMonomialSpace2d import ScaledMonomialSpace2d
from ..quadrature import GaussLegendreQuadrature
from ..quadrature import PolygonMeshIntegralAlg
from ..common import ranges

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
        self.cell2dof, self.cell2dofLocation = self.cell_to_dof()

    def boundary_dof(self):
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool)
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
        """
        Construct the cell2dof array which are 1D array with a location array
        cell2dofLocation.

        The following code give the dofs of i-th cell.

        cell2dof[cell2dofLocation[i]:cell2dofLocation[i+1]]
        """
        p = self.p
        mesh = self.mesh
        cellLocation = mesh.ds.cellLocation
        cell2edge = mesh.ds.cell_to_edge(return_sparse=False)

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

        NV = mesh.number_of_vertices_of_cells()
        NE = mesh.number_of_edges()
        idof = (p-1)*p//2
        idx = (cell2dofLocation[:-1] + NV*p).reshape(-1, 1) + np.arange(idof)
        cell2dof[idx] = NE*p + np.arange(NC*idof).reshape(NC, idof)
        return cell2dof, cell2dofLocation

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        gdof = NE*p + NC*(p-1)*p//2
        return gdof

    def number_of_local_dofs(self):
        p = self.p
        mesh = self.mesh
        NCE = mesh.number_of_edges_of_cells()
        ldofs = NCE*p + (p-1)*p//2
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
        self.dof = DFNCVEMDof2d(mesh, p) # 注意这里是标量的自由度管理
        self.integralalg = self.smspace.integralalg

        self.CM = self.smspace.cell_mass_matrix()
        self.EM = self.smspace.edge_mass_matrix()

        smldof = self.smspace.number_of_local_dofs()
        ndof = p*(p+1)//2
        self.H0 = inv(self.CM[:, 0:ndof, 0:ndof])
        self.H1 = inv(self.EM[:, 0:p, 0:p])

        self.ftype = self.mesh.ftype
        self.itype = self.mesh.itype

        self.G, self.B = self.matrix_G_B()
        self.R, self.J = self.matrix_R_J()
        self.D = self.matrix_D()
        self.Q, self.L = self.matrix_Qp_L()
        self.PI0 = self.matrix_PI0()


    def project(self, u):
        p = self.p # p >= 2
        cell2dof = self.cell_to_dof('cell') # 获得单元内部自由度
        uh = self.function()

        def u0(x, index):
            return np.einsum('ij..., ijm->ijm...', u(x),
                    self.smspace.basis(x, p=p-2, index=index))
        area = self.smspace.cellmeasure
        uh[cell2dof] = self.integralalg.integral(
                u0, celltype=True)/area[:, None, None]

        def u1(x):
            return np.einsum('ij..., ijm->ijm...', u(x),
                    self.smspace.edge_basis(x, p=p-1))
        edge2dof = self.dof.edge_to_dof() # 获得边界自由度
        h = self.mesh.entity_measure('edge')
        uh[edge2dof] = self.integralalg.edge_integral(
                u1, edgetype=True)/h[:, None, None]
        return uh

    def project_to_smspace(self, uh):
        NC = self.mesh.number_of_cells()
        cell2dof, cell2dofLocation = self.cell_to_dof()
        smldof = self.smspace.number_of_local_dofs()
        sh = self.smspace.function(dim=2)
        c2d = self.smspace.cell_to_dof()
        def f(i):
            PI0 = self.PI0[i]
            x = uh[cell2dof[cell2dofLocation[i]:cell2dofLocation[i+1]]]
            y = PI0@x.T.flat
            sh[c2d[i], 0] = y[:smldof]
            sh[c2d[i], 1] = y[smldof:2*smldof]
        list(map(f, range(NC))) # must add list for map
        return sh

    def matrix_PI0(self):
        p = self.p
        NC = self.mesh.number_of_cells()
        cell2dof, cell2dofLocation = self.cell_to_dof()
        ndof0 = self.smspace.number_of_local_dofs(p=p-1)
        Z = np.zeros((ndof0, ndof0), dtype=self.ftype)
        def f(i):
            G = np.block([
                [self.G[0][i]  , self.G[2][i]  , self.B[0][i]],
                [self.G[2][i].T, self.G[1][i]  , self.B[1][i]],
                [self.B[0][i].T, self.B[1][i].T, Z]]
                )
            s = slice(cell2dofLocation[i], cell2dofLocation[i+1])
            R =  np.block([
                [self.R[0][0][:, s], self.R[0][1][:, s]],
                [self.R[1][0][:, s], self.R[1][1][:, s]],
                [self.J[0][:, s], self.J[1][:, s]]])
            PI = inv(G)@R
            return PI
        PI0 = list(map(f, range(NC)))
        return PI0

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

        # 分块矩阵 G = [[G00, G01], [G10, G11]]
        G00 = np.zeros((NC, smldof, smldof), dtype=self.ftype)
        G01 = np.zeros((NC, smldof, smldof), dtype=self.ftype)
        G11 = np.zeros((NC, smldof, smldof), dtype=self.ftype)

        idx = self.smspace.index1(p=p)
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
        NE = mesh.number_of_edges()
        NV = mesh.number_of_vertices_of_cells()

        area = self.smspace.cellmeasure # 单元面积
        ch = self.smspace.cellsize # 单元尺寸 

        smldof = self.smspace.number_of_local_dofs() # 标量 p 次单元缩放空间的维数
        ndof = smldof - p - 1 # 标量 p - 1 次单元缩放单项项式空间的维数
        cell2dof, cell2dofLocation = self.cell_to_dof() # 标量的自由度信息

        CM = self.CM # 单元质量矩阵

        # 构造分块矩阵 R = [[R00, R01], [R10, R11]]
        idx = self.smspace.index2(p=p) # 两次求导后的非零基函数编号及求导系数
        xx = idx['xx']
        yy = idx['yy']
        xy = idx['xy']
        R00 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        R01 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        R10 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        R11 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)

        idx0 = (cell2dofLocation[0:-1] + NV*p).reshape(-1, 1) + np.arange(ndof-p)
        R00[xx[0], idx0] -= xx[1][None, :]
        R00[yy[0], idx0] -= 0.5*yy[1][None, :]

        R11[xx[0], idx0] -= 0.5*xx[1][None, :]
        R11[yy[0], idx0] -= yy[1][None, :]

        #here is not idx[3], 
        R01[xy[0], idx0] -= 0.5*xy[1][None, :]
        R10[xy[0], idx0] -= 0.5*xy[1][None, :]

        # 这里错过了
        val = CM[:, :, 0].T/area[None, :]
        R00[:, idx0[:, 0]] += val
        R11[:, idx0[:, 0]] += val

        node = mesh.entity('node')
        edge = mesh.entity('edge')
        n = mesh.edge_unit_normal()
        eh = mesh.entity_measure('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        # self.H1 is the inverse of Q_{k-1}^F
        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        phi0 = self.smspace.basis(ps, index=edge2cell[:, 0], p=p-1)
        phi = self.smspace.edge_basis(ps, p=p-1)
        F0 = np.einsum('i, ijm, ijn, j, j->jmn', ws, phi0, phi, eh, eh)@self.H1

        idx = self.smspace.index1(p=p) # 一次求导后的非零基函数编号及求导系数
        x = idx['x']
        y = idx['y']
        idx0 = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
        h2 = x[1].reshape(1, -1)/ch[edge2cell[:, [0]]]
        h3 = y[1].reshape(1, -1)/ch[edge2cell[:, [0]]]

        val = np.einsum('ij, ijk, i->jik', h2, F0, n[:, 0])
        np.add.at(R00, (x[0][:, None, None], idx0), val)

        val = np.einsum('ij, ijk, i->jik', h3, F0, 0.5*n[:, 1])
        np.add.at(R00, (y[0][:, None, None], idx0), val)

        val = np.einsum('ij, ijk, i->jik', h2, F0, 0.5*n[:, 0])
        np.add.at(R11, (x[0][:, None, None], idx0), val)

        val = np.einsum('ij, ijk, i->jik', h3, F0, n[:, 1])
        np.add.at(R11, (y[0][:, None, None], idx0), val)

        val = np.einsum('ij, ijk, i->jik', h3, F0, 0.5*n[:, 0])
        np.add.at(R01, (y[0][:, None, None], idx0), val)

        val = np.einsum('ij, ijk, i->jik', h2, F0, 0.5*n[:, 1])
        np.add.at(R10, (x[0][:, None, None], idx0), val)

        a2 = area**2
        start = cell2dofLocation[edge2cell[:, 0]] + edge2cell[:, 2]*p
        val = np.einsum('ij, ij, i, i->ji',
            h3, CM[edge2cell[:, 0], 0:ndof, 0], eh/a2[edge2cell[:, 0]], n[:, 1])
        np.add.at(R00, (y[0][:, None], start), val)

        val = np.einsum('ij, ij, i, i->ji',
            h2, CM[edge2cell[:, 0], 0:ndof, 0], eh/a2[edge2cell[:, 0]], n[:, 0])
        np.add.at(R11, (x[0][:, None], start), val)

        val = np.einsum('ij, ij, i, i->ji',
            h3, CM[edge2cell[:, 0], 0:ndof, 0], eh/a2[edge2cell[:, 0]], n[:, 0])
        np.subtract.at(R01, (y[0][:, None], start), val)

        val = np.einsum('ij, ij, i, i->ji',
            h2, CM[edge2cell[:, 0], 0:ndof, 0], eh/a2[edge2cell[:, 0]], n[:, 1])
        np.subtract.at(R10, (x[0][:, None], start), val)


        if isInEdge.sum() > 0:
            phi1 = self.smspace.basis(ps, index=edge2cell[:, 1], p=p-1)
            F1 = np.einsum('i, ijm, ijn, j, j->jmn', ws, phi1, phi, eh, eh)@self.H1
            idx0 = cell2dofLocation[edge2cell[:, [1]]] + edge2cell[:, [3]]*p + np.arange(p)
            h2 = x[1].reshape(1, -1)/ch[edge2cell[:, [1]]]
            h3 = y[1].reshape(1, -1)/ch[edge2cell[:, [1]]]

            val = np.einsum('ij, ijk, i->jik', h2, F1[:, 0:ndof], n[:, 0])
            np.subtract.at(R00, (x[0][:, None, None], idx0[isInEdge]), val[:, isInEdge])
            val = np.einsum('ij, ijk, i->jik', h3, F1[:, 0:ndof], 0.5*n[:, 1])
            np.subtract.at(R00, (y[0][:, None, None], idx0[isInEdge]), val[:, isInEdge])

            val = np.einsum('ij, ijk, i->jik', h2, F1[:, 0:ndof], 0.5*n[:, 0])
            np.subtract.at(R11, (x[0][:, None, None], idx0[isInEdge]), val[:, isInEdge])
            val = np.einsum('ij, ijk, i->jik', h3, F1[:, 0:ndof], n[:, 1])
            np.subtract.at(R11, (y[0][:, None, None], idx0[isInEdge]), val[:, isInEdge])

            val = np.einsum('ij, ijk, i->jik', h3, F1[:, 0:ndof], 0.5*n[:, 0])
            np.subtract.at(R01, (y[0][:, None, None], idx0[isInEdge]), val[:, isInEdge])
            val = np.einsum('ij, ijk, i->jik', h2, F1[:, 0:ndof], 0.5*n[:, 1])
            np.subtract.at(R10, (x[0][:, None, None], idx0[isInEdge]), val[:, isInEdge])

            start = cell2dofLocation[edge2cell[:, 1]] + edge2cell[:, 3]*p
            val = np.einsum('ij, ij, i, i->ji',
                h3, CM[edge2cell[:, 1], 0:ndof, 0], eh/a2[edge2cell[:, 1]], n[:, 1])
            np.subtract.at(R00, (y[0][:, None], start[isInEdge]), val[:, isInEdge])

            val = np.einsum('ij, ij, i, i->ji',
                h2, CM[edge2cell[:, 1], 0:ndof, 0], eh/a2[edge2cell[:, 1]], n[:, 0])
            np.subtract.at(R11, (x[0][:, None], start[isInEdge]), val[:, isInEdge])

            val = np.einsum('ij, ij, i, i->ji',
                h3, CM[edge2cell[:, 1], 0:ndof, 0], eh/a2[edge2cell[:, 1]], n[:, 0])
            np.add.at(R01, (y[0][:, None], start[isInEdge]), val[:, isInEdge])

            val = np.einsum('ij, ij, i, i->ji',
                h2, CM[edge2cell[:, 1], 0:ndof, 0], eh/a2[edge2cell[:, 1]], n[:, 1])
            np.add.at(R10, (x[0][:, None], start[isInEdge]), val[:, isInEdge])


        R = [[R00, R01], [R10, R11]]

        # 分块矩阵 J =[J0, J1]
        J0 = np.zeros((ndof, len(cell2dof)), dtype=self.ftype)
        J1 = np.zeros((ndof, len(cell2dof)), dtype=self.ftype)
        idx = self.smspace.index1(p=p-1)
        x = idx['x']
        y = idx['y']
        idx0 = (cell2dofLocation[0:-1] + NV*p).reshape(-1, 1) + np.arange(ndof-p)

        # 这里也错过了，从 1 到 多个的时候
        val = ch[:, None]*x[1]
        J0[x[0], idx0] -= val
        val = ch[:, None]*y[1]
        J1[y[0], idx0] -= val

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

        J = [J0, J1]
        return R, J


    def matrix_Qp_L(self):
        p = self.p
        if p > 2:
            mesh = self.mesh
            NC = mesh.number_of_cells()
            NV = mesh.number_of_vertices_of_cells()
            area = self.smspace.cellmeasure


            smldof = self.smspace.number_of_local_dofs() # 标量 p 次单元缩放空间的维数
            ndof = (p-2)*(p-1)//2 # 标量 p - 1 次单元缩放单项项式空间的维数
            cell2dof, cell2dofLocation = self.cell_to_dof() # 标量的自由度信息
            CM = self.CM

            idx = self.smspace.index1(p=p-2)
            x = idx['x']
            y = idx['y']

            Qp = CM[:, x[0][:, None], x[0]] + CM[:, y[0][:, None], y[0]]

            L0 = np.zeros((ndof, len(cell2dof)), dtype=self.ftype)
            idx0 = (cell2dofLocation[0:-1] + NV*p).reshape(-1, 1) + y[0]
            L0[:, idx0] = area[:, None]

            L1 = np.zeros((ndof, len(cell2dof)), dtype=self.ftype)
            idx0 = (cell2dofLocation[0:-1] + NV*p).reshape(-1, 1) + x[0]
            L1[:, idx0] = -area[:, None]

            return Qp, [L0, L1]
        else:
            return None, None

    def matrix_D(self):
        p = self.p

        mesh = self.mesh
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()
        NV = mesh.number_of_vertices_of_cells()
        edge = mesh.entity('edge')
        node = mesh.entity('node')
        edge2cell = mesh.ds.edge_to_cell()

        area = self.smspace.cellmeasure # 单元面积
        ch = self.smspace.cellsize # 单元尺寸 
        CM = self.CM

        smldof = self.smspace.number_of_local_dofs() # 标量 p 次单元缩放空间的维数
        ndof = (p-1)*p//2
        cell2dof, cell2dofLocation = self.cell_to_dof() # 标量的自由度信息

        D = np.zeros((len(cell2dof), smldof), dtype=self.ftype)

        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        phi0 = self.smspace.basis(ps, index=edge2cell[:, 0])
        phi = self.smspace.edge_basis(ps, p=p-1)
        F0 = np.einsum('i, ijm, ijn->jmn', ws, phi, phi0) 
        idx0 = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
        np.add.at(D, (idx0, np.s_[:]), F0)

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        if isInEdge.sum() > 0:
            phi1 = self.smspace.basis(ps, index=edge2cell[:, 1])
            F1 = np.einsum('i, ijm, ijn->jmn', ws, phi, phi1)
            idx1 = cell2dofLocation[edge2cell[:, [1]]] + edge2cell[:, [3]]*p + np.arange(p)
            np.add.at(D, (idx1[isInEdge], np.s_[:]), F1[isInEdge])

        idx0 = (cell2dofLocation[0:-1] + NV*p).reshape(-1, 1) + np.arange(ndof)
        np.add.at(D, (idx0, np.s_[:]), CM[:, 0:ndof]/area[:, None, None])
        return D

    def matrix_A(self):

        p = self.p
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        NC = self.mesh.number_of_cells()

        mesh = self.mesh
        NE = mesh.number_of_edges()
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        eh = mesh.entity_measure('edge')
        area = self.smspace.cellmeasure
        ch = self.smspace.cellsize

        ldof = self.dof.number_of_local_dofs()
        f0 = lambda ndof: np.zeros((ndof, ndof), dtype=self.ftype)
        S00 = list(map(f0, ldof))
        S11 = list(map(f0, ldof))
        S01 = list(map(f0, ldof))

        def f1(i):
            j = edge2cell[i, 2]
            c = eh[i]**2/ch[edge2cell[i, 0]]
            S00[edge2cell[i, 0]][p*j:p*(j+1), p*j:p*(j+1)] += self.H1[i]*c
            S11[edge2cell[i, 0]][p*j:p*(j+1), p*j:p*(j+1)] += self.H1[i]*c

        def f2(i):
            if isInEdge[i]:
                j = edge2cell[i, 3]
                c = eh[i]**2/ch[edge2cell[i, 1]]
                S00[edge2cell[i, 1]][p*j:p*(j+1), p*j:p*(j+1)] += self.H1[i]*c
                S11[edge2cell[i, 1]][p*j:p*(j+1), p*j:p*(j+1)] += self.H1[i]*c
        list(map(f1, range(NE)))
        list(map(f2, range(NE)))

        if p > 2:
            Q = self.Q
            L = self.L
            def f3(i):
                L0 = L[0][:, cell2dofLocation[i]:cell2dofLocation[i+1]]
                L1 = L[1][:, cell2dofLocation[i]:cell2dofLocation[i+1]]
                H = inv(Q[i])/area[i]
                S00[i] += L0.T@H@L0
                S11[i] += L1.T@H@L1
                S01[i] += L0.T@H@L1
            list(map(f3, range(NC)))

        ndof0 = p*(p+1)//2
        ndof1 = (p+1)*(p+2)//2
        Z = np.zeros((ndof0, ndof0), dtype=self.ftype)
        ldof = self.dof.number_of_local_dofs()
        def f4(i):
            PI0 = self.PI0[i]
            D = np.eye(PI0.shape[1])
            D[:ldof[i], :] -= self.D[cell2dofLocation[i]:cell2dofLocation[i+1]]@PI0[:ndof1, :]
            D[ldof[i]:, :] -= self.D[cell2dofLocation[i]:cell2dofLocation[i+1]]@PI0[ndof1:2*ndof1, :]

            A = D.T@np.block([[S00[i], S01[i]], [S01[i].T, S11[i]]])@D

            J0 = self.J[0][:, cell2dofLocation[i]:cell2dofLocation[i+1]]
            J1 = self.J[1][:, cell2dofLocation[i]:cell2dofLocation[i+1]]
            F00 = J0.T@self.H0[i]@J0
            F11 = J1.T@self.H0[i]@J1
            F01 = J1.T@self.H0[i]@J0
            A00 = A[:ldof[i], :ldof[i]]
            A11 = A[ldof[i]:, ldof[i]:]
            A01 = A[:ldof[i], ldof[i]:]
            return A00+F00+0.5*F11, A11+F11+0.5*F00, A01+0.5*F01

        S = list(map(f4, range(NC)))

        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        idx = list(map(np.meshgrid, cd, cd))
        I = np.concatenate(list(map(lambda x: x[1].flat, idx)))
        J = np.concatenate(list(map(lambda x: x[0].flat, idx)))

        gdof = self.number_of_global_dofs() # 注意这里是标量的自由度管理

        A00 = np.concatenate(list(map(lambda x: x[0].flat, S)))
        A00 = coo_matrix((A00, (I, J)), shape=(gdof, gdof), dtype=self.ftype)

        A11 = np.concatenate(list(map(lambda x: x[1].flat, S)))
        A11 = coo_matrix((A11, (I, J)), shape=(gdof, gdof), dtype=self.ftype)

        A01 = np.concatenate(list(map(lambda x: x[2].flat, S)))
        A01 = coo_matrix((A01, (I, J)), shape=(gdof, gdof), dtype=self.ftype)

        return bmat([[A00, A01], [A01.T, A11]], format='csr')

    def matrix_P(self):
        p = self.p
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        NC = self.mesh.number_of_cells()
        cd0 = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        cd1 = self.smspace.cell_to_dof(p=p-1)
        idx = list(map(np.meshgrid, cd0, cd1))
        I = np.concatenate(list(map(lambda x: x[1].flat, idx)))
        J = np.concatenate(list(map(lambda x: x[0].flat, idx)))
        def f(i, k):
            J = self.J[k][cell2dofLocation[i]:cell2dofLocation[i+1]]
            return J.flatten()
        gdof0 = self.smspace.number_of_global_dofs(p=p-1)
        gdof1 = self.number_of_global_dofs()
        P0 = np.concatenate(list(map(lambda i: f(i, 0), range(NC))))
        P0 = coo_matrix((P0, (I, J)),
                shape=(gdof0, gdof1), dtype=self.ftype)
        P1 = np.concatenate(list(map(lambda i: f(i, 1), range(NC))))
        P1 = coo_matrix((P1, (I, J)),
                shape=(gdof0, gdof1), dtype=self.ftype)

        return bmat([[P0, P1]], format='csr')

    def source_vector(self, f):
        p = self.p
        ndof = self.smspace.number_of_local_dofs(p=p-2)
        Q = inv(self.CM[:, :ndof, :ndof])
        phi = lambda x: self.smspace.basis(x, p=p-2)
        def u(x, index):
            return np.einsum('ij..., ijm->ijm...', f(x),
                    self.smspace.basis(x, index=index, p=p-2))
        bb = self.integralalg.integral(u, celltype=True)
        bb = Q@bb
        bb *= self.smspace.cellmeasure[:, None, None]
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof(doftype='cell')
        b = np.zeros((gdof, 2), dtype=self.ftype)
        b[cell2dof, :] = bb
        return b.T.flat

    def set_dirichlet_bc(self, gh, g, is_dirichlet_edge=None):
        """
        """
        p = self.p
        mesh = self.mesh
        h = mesh.entity_measure('edge')
        isBdEdge = mesh.ds.boundary_edge_flag()
        edge2dof = self.dof.edge_to_dof()

        qf = GaussLegendreQuadrature(self.p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = mesh.edge_bc_to_point(bcs, index=isBdEdge)
        val = g(ps)

        ephi = self.smspace.edge_basis(ps, index=isBdEdge, p=p-1)
        b = np.einsum('i, ij..., ijk, j->jk...', ws, val, ephi, h[isBdEdge])
        gh[edge2dof[isBdEdge]] = self.H1[isBdEdge]@b

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def cell_to_dof(self, doftype='all'):
        if doftype == 'all':
            return self.dof.cell2dof, self.dof.cell2dofLocation
        elif doftype == 'cell':
            p = self.p
            NE = self.mesh.number_of_edges()
            NC = self.mesh.number_of_cells()
            idof = (p-1)*p//2
            cell2dof = NE*p + np.arange(NC*idof).reshape(NC, idof)
            return cell2dof
        elif doftype == 'edge':
            return self.dof.edge_to_dof()

    def boundary_dof(self):
        return self.dof.boundary_dof()

    def function(self, dim=2, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def array(self, dim=2):
        gdof = self.number_of_global_dofs()
        if dim in [None, 1]:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=self.ftype)
