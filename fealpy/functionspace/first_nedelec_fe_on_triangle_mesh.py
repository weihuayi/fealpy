import numpy as np

from ..mesh import TriangleMesh
from ..decorator import barycentric
from .Function import Function

class FirstNedelecFEOnTriangleMesh:

    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.dof = FNDof2d(mesh, p)

    @barycentric
    def basis(self, bcs):
        p = self.p
        if p == 1:
            pass

    def interpolation_points(self):
        return self.dof.interpolation_points()

class FNDof2d:
    def __init__(self, mesh, p):
        """
        """
        self.p = p
        self.mesh = mesh
        self.itype = mesh.itype
        self.cell2dof = self.cell_to_dof()  # 默认的自由度数组

    def is_boundary_dof(self, threshold=None):
        """
        @brief 
        """
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_edge_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('edge', index=index)
                flag = threshold(bc)
                index = index[flag]
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        edge2dof = self.edge_to_dof()
        isBdDof[edge2dof[index]] = True
        return isBdDof

    def edge_to_dof(self, index=np.s_[:]):
        """
        @brief  
        """
        if isinstance(index, slice) and index == slice(None):
            NE = self.mesh.number_of_edges()
            index = np.arange(NE)
        elif isinstance(index, np.ndarray) and (index.dtype == np.bool_):
            index, = np.nonzero(index)
            NE = len(index)
        elif isinstance(index, list) and (type(index[0]) is np.bool_):
            index, = np.nonzero(index)
            NE = len(index)
        else:
            NE = len(index)
        p = self.p
        edge2dof =  p*index[:, None] + np.arange(p) 
        return edge2dof

    face_to_dof = edge_to_dof

    def cell_to_dof(self, index=np.s_[:]):
        """
        @brief 
        """
        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells()
        ldof = self.number_of_local_dofs(doftype='all')
        cdof = self.number_of_local_dofs(doftype='cell')
        edof = self.number_of_local_dofs(doftype='edge')
        cell2dof = np.zeros((NC, ldof), dtype=self.itype) 

        edge2cell = self.mesh.ds.edge_to_cell()
        edge2dof = self.edge_to_dof()

        edof = self.number_of_local_dofs(doftype='edge')
        flag = edge2cell[:, 2] == 0
        idx = np.arange(0*edof, 1*edof)
        cell2dof[edge2cell[flag, 0][:, None], idx] = edge2dof[flag]

        flag = edge2cell[:, 2] == 1
        idx = np.arange(1*edof, 2*edof)
        cell2dof[edge2cell[flag, 0][:, None], idx] = edge2dof[flag]

        flag = edge2cell[:, 2] == 2
        idx = np.arange(2*edof, 3*edof)
        cell2dof[edge2cell[flag, 0][:, None], idx] = edge2dof[flag]

        iflag = edge2cell[:, 0] != edge2cell[:, 1]

        flag = iflag & (edge2cell[:, 3] == 0)
        idx = np.arange(1*edof-1, 0*edof-1, -1)
        cell2dof[edge2cell[flag, 1][:, None], idx] = edge2dof[flag]

        flag = iflag & (edge2cell[:, 3] == 1)
        idx = np.arange(2*edof-1, 1*edof-1, -1)
        cell2dof[edge2cell[flag, 1][:, None], idx] = edge2dof[flag]

        flag = iflag & (edge2cell[:, 3] == 2)
        idx = np.arange(3*edof-1, 2*edof-1, -1)
        cell2dof[edge2cell[flag, 1][:, None], idx] = edge2dof[flag]

        cdof = self.number_of_local_dofs(doftype='cell')
        cell2dof[:, 3*edof:] = NE*edof + np.arange(NC*cdof).reshape(NC, cdof)
        return cell2dof[index]

    def number_of_local_dofs(self, doftype='all'):
        p = self.p 
        if doftype == 'all':  # number of all dofs on a cell
            return 3*p + p*(p-1)  
        elif doftype in {'cell', 2}:  # number of dofs inside the cell
            return p*(p-1) 
        elif doftype in {'face', 'edge', 1}:  # number of dofs on a edge
            return p 
        elif doftype in {'node', 0}:  # number of dofs on a node
            return 0

    def number_of_global_dofs(self):
        NE = self.mesh.number_of_edges()
        NC = self.mesh.number_of_cells()
        edof = self.number_of_local_dofs(doftype='edge')
        cdof = self.number_of_local_dofs(doftype='cell')
        gdof = NE*edof + NC*cdof 
        return gdof

    def interpolation_points(self, index=np.s_[:], scale=0.8):
        """
        @brief 为每个插值点配备一个唯一的插值点
        """
        p = self.p
        GD = self.mesh.geo_dimension()
        gdof = self.number_of_global_dofs()
        ips = np.zeros((gdof, GD), dtype=self.mesh.ftype) 

        NE = self.mesh.number_of_edges()
        edof = self.number_of_local_dofs(doftype='edge')
        qf = self.mesh.integrator(p, etype='edge')
        bcs, ws = qf.get_quadrature_points_and_weights()
        node = self.mesh.entity('node')
        edge = self.mesh.entity('edge')
        ips[:NE*edof] = np.einsum('...j, ijk->i...k', bcs, node[edge]).reshape(-1, GD)

        if p == 2: 
            ps =  self.mesh.entity_barycenter('cell')
            ips[NE*edof::2]  = ps 
            ips[NE*edof+1::] = ps 

        if p > 2:
            mi = self.mesh.multi_index_matrix(p-2)
            bcs = mi/(p-2)
            bcs += scale*(1/3 - bcs)
            bcs[:, 0] = 1 - bcs[:, 1] - bcs[:, 2]
            cell = self.mesh.entity('cell')
            ps = np.einsum('...j, ijk->i...k', bcs, node[cell]).reshape(-1, GD)
            ips[NE*edof::2] = ps 
            ips[NE*edof+1::2] = ps

        return ips[index]

