
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

        self.ftype = self.mesh.ftype
        self.itype = self.mesh.itype

    def index0(self, p=None):
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

    def index1(self, p=None):
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
        p = self.p
        CM = self.CM
        print(CM)
        smldof = self.smspace.number_of_local_dofs()
        NC = self.mesh.number_of_cells()
        h = self.smspace.cellsize
        self.G = np.zeros((NC, 2*smldof, 2*smldof), dtype=self.ftype)

        idx = self.index0()
        print(idx)
        idx0 = np.arange(smldof - p - 1)
        L = idx[2][np.newaxis, ...]/h[..., np.newaxis]
        R = idx[3][np.newaxis, ...]/h[..., np.newaxis]
        mxx = np.einsum('ij, ijk, ik->ijk',
                L, CM[:, idx0.reshape(-1, 1), idx0], L)
        mxy = np.einsum('ij, ijk, ik->ijk',
                L, CM[:, idx0.reshape(-1, 1), idx0], R)
        myy = np.einsum('ij, ijk, ik->ijk',
                R, CM[:, idx0.reshape(-1, 1), idx0], R)

        G = np.zeros((NC, 2*smldof, 2*smldof), dtype=self.ftype)
        G[:, idx[0].reshape(-1, 1), idx[0]] += mxx
        G[:, idx[1].reshape(-1, 1), idx[1]] += 0.5*myy

        G[:, smldof+idx[1].reshape(-1, 1), smldof+idx[1]] += myy
        G[:, smldof+idx[0].reshape(-1, 1), smldof+idx[0]] += 0.5*mxx

        G[:, smldof+idx[0].reshape(-1, 1), idx[1]] += 0.5*mxy
        G[:, idx[1].reshape(-1, 1), smldof+idx[0]] += 0.5*mxy.swapaxes(-1, -2)

        B = np.zeros((NC, 2*smldof, smldof-p-1), dtype=self.ftype)
        B[:, idx[0]] = np.einsum('ij, ijk->ijk', L, CM[:, idx0])
        B[:, smdof+idx[1]] = np.einsum('ij, ijk->ijk', R, CM[:, idx0])

        area = self.smspace.cellmeasure
        mx = L*CM[:, 0, idx0]/area[:, np.newaxis]
        my = R*CM[:, 0, idx0]/area[:, np.newaxis]
        G[:, idx[1].reshape(-1, 1), idx[1]] += np.einsum('ij, ik->ijk', my, my)
        G[:, smldof+idx[0].reshape(-1, 1), smldof+idx[0]] += np.einsum('ij, ik->ijk', mx, mx)
        G[:, smldof+idx[0].reshape(-1, 1), idx[1]] -= np.einsum('ij, ik->ijk', mx, my)
        G[:, idx[1].reshape(-1, 1), idx[0]] = G[:, smldof+idx[0].reshape(-1, 1), idx[1]].swapaxes(-1, -2)


        m = CM[:, 0, :]/area
        idx0 = np.arange(smldof)
        G[:, idx0.reshape(-1, 1), idx0] += np.einsum('ij, ik->ijk', m, m)
        G[:, smldof+idx0.reshape(-1, 1), smldof+idx0] += G[:, idx0.reshape(-1, 1), idx0]
        return G

    def matrix_R_J(self):
        p = self.p
        smldof = self.smspace.number_of_local_dofs()
        area = self.smspace.cellmeasure
        NC = self.mesh.number_of_cells()
        NE = self.mesh.number_of_edges()

        n = (p-1)*p//2  # P_{k-2}(K)

        idx = self.index1()
        R0 = np.zeros((NC, n0, n0), dtype=self.ftype)
        R1 = np.zeros((2, NE, p, p), dtype=self.ftype)
        R0[:, range(n0), range(n0)] = area[:, np.newaxis]

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
