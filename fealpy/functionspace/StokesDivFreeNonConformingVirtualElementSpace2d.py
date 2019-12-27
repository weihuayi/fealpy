
import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from .function import Function
from .ScaledMonomialSpace2d import ScaledMonomialSpace2d
from ..quadrature import GaussLegendreQuadrature
from ..quadrature import PolygonMeshIntegralAlg
from ..common import ranges

class SDFNCVEMDof2d():
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
        cell2edge = mesh.ds.cell_to_edge(sparse=False)

        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        cell2dofLocation = np.zeros(NC+1, dtype=np.int)
        cell2dofLocation[1:] = np.add.accumulate(ldof)
        cell2dof = np.zeros(cell2dofLocation[-1], dtype=np.int)

        edge2dof = self.edge_to_dof()
        edge2cell = mesh.ds.edge_to_cell()
        idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
        cell2dof[idx] = edge2dof

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        idx = (cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p).reshape(-1, 1) + np.arange(p)
        cell2dof[idx] = edge2dof[isInEdge]

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


class StokesDivFreeNonConformingVirtualElementSpace2d:

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
        self.dof = SDFNCVEMDof2d(mesh, p) # 注意这里是标量的自由度管理
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

    def index1(self, p=None):
        if p is None:
            p = self.p

        n = (p+1)*(p+2)//2
        idx1 = np.cumsum(np.arange(p+1))
        idx0 = np.arange(p+1) + idx1

        mask0 = np.ones(n, dtype=np.bool)
        mask1 = np.ones(n, dtype=np.bool)
        mask0[idx0] = False
        mask1[idx1] = False

        idx = np.arange(n)
        idx0 = idx[mask0]
        idx1 = idx[mask1]

        idx = np.repeat(range(2, p+2), range(1, p+1))
        idx4 = ranges(range(p+1), start=1)
        idx3 = idx - idx4
        return idx0, idx1, idx3, idx4

    def index2(self, p=None):
        if p is None:
            p = self.p

        n = (p+1)*(p+2)//2
        mask0 = np.ones(n, dtype=np.bool)
        mask1 = np.ones(n, dtype=np.bool)
        mask2 = np.ones(n, dtype=np.bool)

        idx1 = np.cumsum(np.arange(p+1))
        idx0 = np.arange(p+1) + idx1
        mask0[idx0] = False
        mask1[idx1] = False

        mask2[idx0] = False
        mask2[idx1] = False

        idx0 = np.cumsum([1]+list(range(3, p+2)))
        idx1 = np.cumsum([2]+list(range(2, p+1)))
        mask0[idx0] = False
        mask1[idx1] = False

        idx = np.arange(n)
        idx0 = idx[mask0]
        idx1 = idx[mask1]
        idx2 = idx[mask2]

        idxa = np.repeat(range(2, p+1), range(1, p))
        idxb = np.repeat(range(4, p+3), range(1, p))

        idxc = ranges(range(p), start=1)
        idxd = ranges(range(p), start=2)

        idx3 = (idxa - idxc)*(idxb - idxd)
        idx4 = idxc*idxd
        idx5 = idxc*(idxa - idxc)

        return idx0, idx1, idx2, idx3, idx4, idx5

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

        idx = self.index1()
        L = idx[2][None, ...]/ch[..., None]
        R = idx[3][None, ...]/ch[..., None]
        mxx = np.einsum('ij, ijk, ik->ijk', L, CM[:, 0:ndof, 0:ndof], L)
        myx = np.einsum('ij, ijk, ik->ijk', R, CM[:, 0:ndof, 0:ndof], L)
        myy = np.einsum('ij, ijk, ik->ijk', R, CM[:, 0:ndof, 0:ndof], R)

        G00[:, idx[0][:, None], idx[0]] += mxx
        G00[:, idx[1][:, None], idx[1]] += 0.5*myy

        G01[:, idx[1][:, None], idx[0]] += 0.5*myx

        G11[:, idx[0][:, None], idx[0]] += 0.5*mxx
        G11[:, idx[1][:, None], idx[1]] += myy

        mx = L*CM[:, 0, 0:ndof]/area[:, None]
        my = R*CM[:, 0, 0:ndof]/area[:, None]
        G00[:, idx[1][:, None], idx[1]] += np.einsum('ij, ik->ijk', my, my)
        G01[:, idx[1][:, None], idx[0]] -= np.einsum('ij, ik->ijk', my, mx)
        G11[:, idx[0][:, None], idx[0]] += np.einsum('ij, ik->ijk', mx, mx)

        m = CM[:, 0, :]/area[:, None]
        val = np.einsum('ij, ik->ijk', m, m)
        G00[:, 0:smldof, 0:smldof] += val
        G11[:, 0:smldof, 0:smldof] += val
        G = [G00, G11, G01]

        # 分块矩阵 B = [[B0], [B1]]
        B0 = np.zeros((NC, smldof, ndof), dtype=self.ftype)
        B1 = np.zeros((NC, smldof, ndof), dtype=self.ftype)
        B0[:, idx[0]] = np.einsum('ij, ijk->ijk', L, CM[:, 0:ndof, 0:ndof])
        B1[:, idx[1]] = np.einsum('ij, ijk->ijk', R, CM[:, 0:ndof, 0:ndof])

        B = [B0, [B1]]
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
        idx = self.index2() # 两次求导后的非零基函数编号及求导系数
        R00 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        R01 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        R10 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)
        R11 = np.zeros((smldof, len(cell2dof)), dtype=self.ftype)

        idx0 = (cell2dofLocation[0:-1] + NV*p).reshape(-1, 1) + np.arange(ndof-p)
        R00[idx[0][:, None, None], idx0] -= idx[3][:, None, None]
        R00[idx[1][:, None, None], idx0] -= 0.5*idx[4][:, None, None]
        R01[idx[3][:, None, None], idx0] -= 0.5*idx[5][:, None, None]
        R10[idx[3][:, None, None], idx0] -= 0.5*idx[5][:, None, None]
        R11[idx[0][:, None, None], idx0] -= 0.5*idx[3][:, None, None]
        R11[idx[1][:, None, None], idx0] -= idx[4][:, None, None]

        val = CM[:, :, 0].T/area
        R00[:, idx0] += val[:, None]
        R11[:, idx0] += val[:, None]


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
        phi0 = self.smspace.basis(ps, cellidx=edge2cell[:, 0], p=p-1)
        phi = self.smspace.edge_basis(ps, p=p-1)
        F0 = np.einsum('i, ijm, ijn, j, j->jmn', ws, phi0, phi, eh, eh)@self.H1

        idx = self.index1() # 一次求导后的非零基函数编号及求导系数
        idx0 = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
        h2 = idx[2].reshape(1, -1)/ch[edge2cell[:, [0]]]
        h3 = idx[3].reshape(1, -1)/ch[edge2cell[:, [0]]]

        val = np.einsum('ij, ijk, i->jik', h2, F0, n[:, 0])
        np.add.at(R00, (idx[0][:, None, None], idx0), val)
        val = np.einsum('ij, ijk, i->jik', h3, F0, 0.5*n[:, 1])
        np.add.at(R00, (idx[1][:, None, None], idx0), val)

        val = np.einsum('ij, ijk, i->jik', h3, F0, 0.5*n[:, 0])
        np.add.at(R01, (idx[1][:, None, None], idx0), val)

        val = np.einsum('ij, ijk, i->jik', h2, F0, 0.5*n[:, 1])
        np.add.at(R10, (idx[0][:, None, None], idx0), val)

        val = np.einsum('ij, ijk, i->jik', h2, F0, 0.5*n[:, 0])
        np.add.at(R11, (idx[0][:, None, None], idx0), val)
        val = np.einsum('ij, ijk, i->jik', h3, F0, n[:, 1])
        np.add.at(R11, (idx[1][:, None, None], idx0), val)

        a2 = area**2
        start = cell2dofLocation[edge2cell[:, 0]] + edge2cell[:, 2]*p
        val = np.einsum('ij, ij, i, i->ji',
            h3, CM[edge2cell[:, 0], 0:ndof, 0], eh/a2[edge2cell[:, 0]], n[:, 1])
        np.add.at(R00, (idx[1][:, None], start), val)
        val = np.einsum('ij, ij, i, i->ji',
            h3, CM[edge2cell[:, 0], 0:ndof, 0], eh/a2[edge2cell[:, 0]], n[:, 0])
        np.subtract.at(R01, (idx[1][:, None], start), val)
        val = np.einsum('ij, ij, i, i->ji',
            h2, CM[edge2cell[:, 0], 0:ndof, 0], eh/a2[edge2cell[:, 0]], n[:, 1])
        np.subtract.at(R01, (idx[0][:, None], start), val)
        val = np.einsum('ij, ij, i, i->ji',
            h2, CM[edge2cell[:, 0], 0:ndof, 0], eh/a2[edge2cell[:, 0]], n[:, 0])
        np.add.at(R11, (idx[0][:, None], start), val)


        if isInEdge.sum() > 0:
            phi1 = self.smspace.basis(ps, cellidx=edge2cell[:, 1], p=p-1)
            F1 = np.einsum('i, ijm, ijn, j, j->jmn', ws, phi1, phi, eh, eh)@self.H1
            idx0 = cell2dofLocation[edge2cell[:, [1]]] + edge2cell[:, [3]]*p + np.arange(p)
            h2 = idx[2].reshape(1, -1)/ch[edge2cell[:, [1]]]
            h3 = idx[3].reshape(1, -1)/ch[edge2cell[:, [1]]]
            val = np.einsum('ij, ijk, i->jik', h2, F1[:, 0:ndof], n[:, 0])
            np.add.at(R00, (idx[0][:, None, None], idx0[isInEdge]), val[:, isInEdge])
            val = np.einsum('ij, ijk, i->jik', h3, F1[:, 0:ndof], 0.5*n[:, 1])
            np.add.at(R00, (idx[1][:, None, None], idx0[isInEdge]), val[:, isInEdge])
            val = np.einsum('ij, ijk, i->jik', h3, F1[:, 0:ndof], 0.5*n[:, 0])
            np.add.at(R01, (idx[1][:, None, None], idx0[isInEdge]), val[:, isInEdge])
            val = np.einsum('ij, ijk, i->jik', h2, F1[:, 0:ndof], 0.5*n[:, 1])
            np.add.at(R10, (idx[0][:, None, None], idx0[isInEdge]), val[:, isInEdge])
            val = np.einsum('ij, ijk, i->jik', h2, F1[:, 0:ndof], 0.5*n[:, 0])
            np.add.at(R11, (idx[0][:, None, None], idx0[isInEdge]), val[:, isInEdge])
            val = np.einsum('ij, ijk, i->jik', h3, F1[:, 0:ndof], n[:, 1])
            np.add.at(R11, (idx[1][:, None, None], idx0[isInEdge]), val[:, isInEdge])

            idx1 = cell2dofLocation[edge2cell[:, 1]] + edge2cell[:, 3]*p
            val = np.einsum('ij, ij, i, i->ji',
                h3, CM[edge2cell[:, 0], 0:ndof, 0], eh/a2[edge2cell[:, 1]], -n[:, 1])
            np.add.at(R00, (idx[1][:, None], idx1[isInEdge]), val[:, isInEdge])
            val = np.einsum('ij, ij, i, i->ji',
                h3, CM[edge2cell[:, 0], 0:ndof, 0], eh/a2[edge2cell[:, 1]], -n[:, 0])
            np.subtract.at(R01, (idx[1][:, None], idx1[isInEdge]), val[:, isInEdge])
            val = np.einsum('ij, ij, i, i->ji',
                h2, CM[edge2cell[:, 0], 0:ndof, 0], eh/a2[edge2cell[:, 1]], -n[:, 1])
            np.subtract.at(R01, (idx[0][:, None], idx1[isInEdge]), val[:, isInEdge])
            val = np.einsum('ij, ij, i, i->ji',
                h2, CM[edge2cell[:, 0], 0:ndof, 0], eh/a2[edge2cell[:, 1]], -n[:, 0])
            np.add.at(R11, (idx[0][:, None], idx1[isInEdge]), val[:, isInEdge])

        R = [[R00, R01], [R10, R11]]

        # 分块矩阵 J =[J0, J1]
        J0 = np.zeros((ndof, len(cell2dof)), dtype=self.ftype)
        J1 = np.zeros((ndof, len(cell2dof)), dtype=self.ftype)

        idx = self.index1(p=p-1)
        idx0 = (cell2dofLocation[0:-1] + NV*p).reshape(-1, 1) + np.arange(ndof-p)
        val = ch[:, None]*idx[2]
        J0[idx[0][:, None, None], idx0] -= val[None, ...]
        val = ch[:, None]*idx[3]
        J1[idx[1][:, None, None], idx0] -= val[None, ...]

        idx0 = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)

        val = np.einsum('ijk, i->jik', F0, n[:, 0])
        np.add.at(J0, (np.s_[:], idx0), val)
        val = np.einsum('ijk, i->jik', F0, n[:, 1])
        np.add.at(J1, (np.s_[:], idx0), val)

        if isInEdge.sum() > 0:
            idx1 = cell2dofLocation[edge2cell[:, [1]]] + edge2cell[:, [3]]*p + np.arange(p)
            val = np.einsum('ijk, i->jik', F1, n[:, 0])
            np.subtract(J0, (np.s_[:], idx1[isInEdge]), val[isInEdge])
            val = np.einsum('ijk, i->jik', F1, n[:, 1])
            np.subtract(J1, (np.s_[:], idx1[isInEdge]), val[isInEdge])

        J = [J0, J1] # U = [[J0, None]， [J1, None]， [None， J0]， [None，J1]]

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

            idx = self.index1(p=p-2)
            Qp = CM[:, idx[0][:, None], idx[0]] + CM[:, idx[1][:, None], idx[1]]

            L0 = np.zeros((ndof, len(cell2dof)), dtype=self.ftype)
            idx0 = (cell2dofLocation[0:-1] + NV*p).reshape(-1, 1) + idx[1]
            L0[:, idx0] = area[:, None]

            L1 = np.zeros((ndof, len(cell2dof)), dtype=self.ftype)
            idx0 = (cell2dofLocation[0:-1] + NV*p).reshape(-1, 1) + idx[0]
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
        phi0 = self.smspace.basis(ps, cellidx=edge2cell[:, 0])
        phi = self.smspace.edge_basis(ps, p=p-1)
        F0 = self.H1@np.einsum('i, ijm, ijn->jmn', ws, phi, phi0)
        idx0 = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
        np.add.at(D, (idx0, np.s_[:]), F0)

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        if isInEdge.sum() > 0:
            phi1 = self.smspace.basis(ps, cellidx=edge2cell[:, 1])
            F1 = self.H1@np.einsum('i, ijm, ijn->jmn', ws, phi, phi1)
            idx1 = cell2dofLocation[edge2cell[:, [1]]] + edge2cell[:, [3]]*p + np.arange(p)
            np.add.at(D, (idx1[isInEdge], np.s_[:]), F1[isInEdge])

        idx0 = (cell2dofLocation[0:-1] + NV*p).reshape(-1, 1) + np.arange(ndof)
        np.add.at(D, (idx0, np.s_[:]), CM[:, 0:ndof]/area[:, None, None])
        return D

    def matrix_A(self):
        
        p = self.p
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        NC = self.mesh.number_of_cells()

        def f0(k, i, j):
            Ji = self.J[i][:, cell2dofLocation[i]:cell2dofLocation[i+1]]
            Jj = self.J[j][:, cell2dofLocation[i]:cell2dofLocation[i+1]]
            return (Ji.T@self.H0[i]@Jj).flat
        F00 = np.concatenate(map(lambda k: f0(k, 0, 0), range(NC)))
        F11 = np.concatenate(map(lambda k: f1(k, 1, 1), range(NC)))
        F10 = np.concatenate(map(lambda k: f1(k, 1, 0), range(NC)))

        def f1(i, copy):
            cd = cell2dof[cell2dofLocation[i]:cell2dofLocation[i+1]]
            return copy(cd, cd.shape[0])
        I = np.concatenate(list(map(lambda i: f1(i, np.repeat), range(NC))))
        J = np.concatenate(list(map(lambda i: f1(i, np.tile), range(NC))))

        val00 = F00 + 0.5*F11
        val11 = 0.5*F00 + F11
        val01 = 0.5*F10

        gdof = self.dof.number_of_global_dofs() # 注意这里是标量的自由度管理
        A00 = coo_matrix((val00, (I, J)), shape=(gdof, gdof), dtype=self.ftype)
        A01 = coo_matrix((val01, (I, J)), shape=(gdof, gdof), dtype=self.ftype)
        A11 = coo_matrix((val11, (I, J)), shape=(gdof, gdof), dtype=self.ftype)

        mesh = self.mesh
        NE = mesh.number_of_edges()
        edge2cell = mesh.ds.edge_to_cell()
        eh = mesh.entity_measure('edge')
        area = self.smspace.cellmeasure
        ch = self.smspace.cellsize

        ldof = self.dof.number_of_local_dofs()
        f2 = lambda ndof: np.zeros((ndof, ndof), dtype=self.ftype)
        S00 = np.array(list(map(f2, ldof)))
        S11 = np.array(list(map(f2, ldof)))
        S01 = np.array(list(map(f2, ldof)))

        def f3(i):
            j = edge2cell[i, 2]
            c = h[i]**2/ch[edge2cell[i, 0]]
            S00[edge2cell[i, 0]][p*j:p*(j+1), p*j:p*(j+1)] += self.H1[i]*c
            S11[edge2cell[i, 0]][p*j:p*(j+1), p*j:p*(j+1)] += self.H1[i]*c

        def f4(i):
            if isInEdge[i]:
                j = edge2cell[i, 3]
                c = h[i]**2/ch[edge2cell[i, 1]]
                S00[edge2cell[i, 1]][p*j:p*(j+1), p*j:p*(j+1)] += self.H1[i]*c
                S11[edge2cell[i, 1]][p*j:p*(j+1), p*j:p*(j+1)] += self.H1[i]*c
        list(map(f3, range(NE)))
        list(map(f4, range(NE)))

        if p > 2:
            Q = self.Q
            L = self.L
            def f5(i):
                L0 = L[0][:, cell2dofLocation[i]:cell2dofLocation[i+1]]
                L1 = L[1][:, cell2dofLocation[i]:cell2dofLocation[i+1]]
                H = inv(Q[i])/area[i]
                S00[i] += L0.T@H@L0
                S11[i] += L1.T@H@L1
                S01[i] += L0.T@H@L1
            list(map(f5, range(NC)))

        def f6(i):
            G00 = self.G[0][i]
            G11 = self.G[1][i]
            G01 = self.G[2][i]
            B0 = self.B[0]
            B1 = self.B[1]

    def number_of_global_dofs(self):
        return 2*self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return 2*self.dof.number_of_local_dofs()

    def cell_to_dof(self):
        return self.dof.cell2dof, self.dof.cell2dofLocation

    def boundary_dof(self):
        return self.dof.boundary_dof()

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = (2, gdof)
        elif type(dim) is int:
            shape = (dim, 2, gdof)
        elif type(dim) is tuple:
            shape = dim + (2, gdof)
        return np.zeros(shape, dtype=np.float)
