import numpy as np
from numpy.linalg import inv

from .Function import Function
from .ScaledMonomialSpace2d import ScaledMonomialSpace2d

class CVVEDof2d:
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.itype = self.mesh.itype
        self.cell2dof = self.cell_to_dof()

    def edge_to_dof(self, index=np.s_[:]):
        e2p = self.mesh.edge_to_ipoint(self.p, index=index)
        NE, ldof = e2p.shape
        edge2dof = np.zeros(NE, 2*ldof, dtype=e2p.dtype)
        edge2dof[:, 0::2] = 2*e2p
        edge2dof[:, 1::2] = edge2dof[:, 0::2] + 1
        return edge2dof

    face_fo_dof = edge_to_dof

    def cell_to_dof(self, index=np.s_[:]):
        """
        """
        cell = self.mesh.ds._cell

        if p == 1:
            cellLocation = self.mesh.ds.cellLocation
            cell2dof = np.zeros(2*len(cell), dtype=cell.dtype)
            cell2dof[0::2] = 2*cell
            cell2dof[1::2] = cell2dof[0::2] + 1
            return np.hsplit(cell2dof, 2*cellLocation[1:-1])[index]

        NC = self.mesh.number_of_cells()
        ldof = self.number_of_local_dofs(doftype='all')

        location = np.zeros(NC+1, dtype=self.itype)
        location[1:] = np.add.accumulate(ldof)

        cell2dof = np.zeros(location[-1], dtype=self.itype)

        edge2dof = self.edge_to_dof()
        edge2cell = self.ds.edge_to_cell()

        edof = self.number_of_local_dofs(doftype='edge')
        idx = location[edge2cell[:, [0]]] + edge2cell[:, [2]]*(edof-2)+ np.arange(edof-2)
        cell2dof[idx] = edge2dof[:, 0:edof-2]

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        idx = (location[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*(edof-2)).reshape(-1, 1) + np.arange(edof-2) 
        cell2dof[idx] = edge2dof[isInEdge, edof-1:1:-1]

        NN = self.number_of_nodes()
        NV = self.mesh.ds.number_of_vertices_of_cells()
        NE = self.number_of_edges()
        cdof = self.number_of_local_dofs(p, iptype='cell') 
        idx = (location[:-1] + NV*2*p).reshape(-1, 1) + np.arange(cdof)
        cell2dof[idx] = 2*NN + NE*2*(p-1) + np.arange(NC*cdof).reshape(NC, cdof)

        return np.hsplit(cell2dof, location[1:-1])[index]

    def number_of_global_dofs(self):
        return 2*self.mesh.number_of_global_ipoints(self.p)

    def number_of_local_dofs(self, doftype='all'):
        return 2*self.mesh.number_of_local_ipoints(self.p, iptype=doftype)

    def interpolation_points(self, index=np.s_[:]):
        return self.mesh.interpolation_points(self.p, scale=0.3)


class ConformingVectorVESpace2d:
    def __init__(self, mesh, p=1):
        self.mesh = mesh
        self.p = p
        self.itype = mesh.itype
        self.ftype = mesh.ftype
        self.dof = CVVEDof2d(mesh, p)





