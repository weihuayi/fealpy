import numpy as np
from functools import reduce
import operator as op
from typing import Optional, Union, Callable

class LinearMeshCFEDof():
    def __init__(self, mesh, p):
        TD = mesh.top_dimension()
        self.mesh = mesh
        self.p = p
        self.multiIndex = mesh.multi_index_matrix(p, TD) 
        self.cell2dof = self.cell_to_dof()

    def is_boundary_dof(self, threshold=None):
        TD = self.mesh.top_dimension()
        gdof = self.number_of_global_dofs()
        if type(threshold) is np.ndarray:
            index = threshold
            if (index.dtype == np.bool_) and (len(index) == gdof):
                return index
        else:
            index = self.mesh.ds.boundary_face_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter(TD-1, index=index)
                flag = threshold(bc)
                index = index[flag]

        face2dof = self.face_to_dof(index=index) # 只获取指定的面的自由度信息
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[face2dof] = True
        return isBdDof

    def face_to_dof(self, index=np.s_[:]):
        return self.mesh.face_to_ipoint(self.p, index=index)

    def edge_to_dof(self, index=np.s_[:]):
        return self.mesh.edge_to_ipoint(self.p, index=index)

    def cell_to_dof(self, index=np.s_[:]):
        return self.mesh.cell_to_ipoint(self.p, index=index)

    def interpolation_points(self, index=np.s_[:]):
        return self.mesh.interpolation_points(self.p, index=index)

    def number_of_global_dofs(self):
        return self.mesh.number_of_global_ipoints(self.p)

    def number_of_local_dofs(self, doftype='cell'):
        return self.mesh.number_of_local_ipoints(self.p, iptype=doftype)

class IntervalMeshCFEDof(LinearMeshCFEDof):
    def __init__(self, mesh, p: int):
        super(IntervalMeshCFEDof, self).__init__(mesh, p)

    def entity_to_dof(self, 
            etype: Union[str, int]='cell', 
            index: Union[np.ndarray, slice]=np.s_[:]
            ) -> np.ndarray:
        if etype in {'cell', 'edge', 1}:
            return self.cell_to_dof()[index] #TODO: cell_to_dof 应该接收一个 index 参数
        elif etype in {'node', 'face', 0}:
            NN = self.mesh.number_of_nodes()
            return np.arange(NN)[index]
        else:
            raise ValueError(f"Unsupported etype: {etype}. Supported types are: 'cell', 'edge', 1, 'node', 'face', and 0.")

class TriangleMeshCFEDof(LinearMeshCFEDof):
    def __init__(self, mesh, p):
        super(TriangleMeshCFEDof, self).__init__(mesh, p)

    def entity_to_dof(self, 
            etype: Union[str, int]='cell', 
            index: Union[np.ndarray, slice]=np.s_[:]
            ) -> np.ndarray:
        if etype in {'cell', 2}:
            return self.cell_to_dof()[index] #TODO
        elif etype in {'face', 'edge', 1}:
            return self.edge_to_dof()[index] #TODO
        elif etype in {'node', 0}:
            NN = self.mesh.number_of_nodes()
            return np.arange(NN)[index]
        else:
            raise ValueError(f"Unsupported etype: {etype}. Supported types are: 'cell', 2, 'face', 'edge', 1, 'node', and 0.")

    def is_on_node_local_dof(self):
        p = self.p
        isNodeDof = np.sum(self.multiIndex == p, axis=-1) > 0
        return isNodeDof

    def is_on_edge_local_dof(self):
        return self.multiIndex == 0

class TetrahedronMeshCFEDof(LinearMeshCFEDof):
    def __init__(self, mesh, p):
        super(TetrahedronMeshCFEDof, self).__init__(mesh, p)

    def entity_to_dof(self, 
            etype: Union[str, int]='cell', 
            index: Union[np.ndarray, slice]=np.s_[:]
            ) -> np.ndarray:
        if etype in {'cell', 3}:
            return self.cell_to_dof()[index] #TODO:
        elif etype in {'face', 2}:
            return self.face_to_dof()[index] #TODO:
        elif etype in {'edge', 1}:
            return self.edge_to_dof()[index] #TODO:
        elif etype in {'node', 0}:
            NN = self.mesh.number_of_nodes() #TODO:
            return np.arange(NN)[index]
        else:
            raise ValueError(f"Unsupported etype: {etype}. Supported types are: 'cell', 3, 'face', 2, 'edge', 1, 'node', and 0.")

    def is_on_node_local_dof(self):
        p = self.p
        isNodeDof = np.sum(self.multiIndex == p, axis=-1) == 1
        return isNodeDof

    def is_on_edge_local_dof(self):
        p =self.p
        ldof = self.number_of_local_dofs()
        localEdge = self.mesh.ds.localEdge
        isEdgeDof = np.zeros((ldof, 6), dtype=np.bool_)
        for i in range(6):
            isEdgeDof[:, i] = (self.multiIndex[:, localEdge[-(i+1), 0]] == 0) & (self.multiIndex[:, localEdge[-(i+1), 1]] == 0 )
        return isEdgeDof

    def is_on_face_local_dof(self):
        p = self.p
        ldof = self.number_of_local_dofs()
        isFaceDof = (self.multiIndex == 0)
        return isFaceDof

class QuadrangleMeshCFEDof(LinearMeshCFEDof):
    def __init__(self, mesh, p):
        super(QuadrangleMeshCFEDof, self).__init__(mesh, p)

class HexahedronMeshCFEDof(LinearMeshCFEDof):
    def __init__(self, mesh, p):
        super(HexahedronMeshCFEDof, self).__init__(mesh, p)

class LinearMeshDFEDof():
    """
    间断单元自由度管理基类.
    """
    def __init__(self, mesh, p):
        TD = mesh.top_dimension()
        self.mesh = mesh
        self.p = p
        if p > 0:
            self.multiIndex = mesh.multi_index_matrix(self.p, TD)
        else:
            TD = mesh.top_dimension()
            self.multiIndex = np.array((TD+1)*(0,), dtype=mesh.itype)
        self.cell2dof = self.cell_to_dof()


    def cell_to_dof(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs()
        cell2dof = np.arange(NC*ldof).reshape(NC, ldof)
        return cell2dof

    def number_of_global_dofs(self):
        NC = self.mesh.number_of_cells()
        ldof = self.number_of_local_dofs()
        gdof = ldof*NC
        return gdof

    def number_of_local_dofs(self):
        p = self.p
        TD = self.mesh.top_dimension()
        numer = reduce(op.mul, range(p + TD, p, -1))
        denom = reduce(op.mul, range(1, TD + 1))
        return numer//denom

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')
        GD = mesh.geo_dimension()

        if p == 0:
            return mesh.entity_barycenter('cell')

        if p == 1:
            return node[cell].reshape(-1, GD)

        w = self.multiIndex/p
        ipoint = np.einsum('ij, kj...->ki...', w, node[cell]).reshape(-1, GD)
        return ipoint


class IntervalMeshDFEDof(LinearMeshDFEDof):
    """
    区间间断单元自由度管理类.
    """
    def __init__(self, mesh, p):
        super(IntervalMeshDFEDof, self).__init__(mesh, p)

    def entity_to_dof(self, 
            etype: Union[str, int]='cell', 
            index: Union[np.ndarray, slice]=np.s_[:]
            ) -> np.ndarray:
        if etype in {'cell', 'edge', 1}:
            return self.cell_to_dof()[index]
        else:
            raise ValueError(f"Unsupported etype: {etype}. Supported types are: 'cell', 'edge', and 1.")

class TriangleMeshDFEDof(LinearMeshDFEDof):
    """
    三角形间断单元自由度管理类.
    """
    def __init__(self, mesh, p):
        super(TriangleMeshDFEDof, self).__init__(mesh, p)

    def entity_to_dof(self, 
            etype: Union[str, int]='cell', 
            index: Union[np.ndarray, slice]=np.s_[:]
            ) -> np.ndarray:
        if etype in {'cell',  2}:
            return self.cell_to_dof()[index]
        else:
            raise ValueError(f"Unsupported etype: {etype}. Supported types are: 'cell' and 2.")


class TetrahedronMeshDFEDof(LinearMeshDFEDof):
    """
    四面体间断单元自由度管理类.
    """
    def __init__(self, mesh, p):
        super(TetrahedronMeshDFEDof, self).__init__(mesh, p)

    def entity_to_dof(self, 
            etype: Union[str, int]='cell', 
            index: Union[np.ndarray, slice]=np.s_[:]
            ) -> np.ndarray:
        if etype in {'cell', 3}:
            return self.cell_to_dof()[index]
        else:
            raise ValueError(f"Unsupported etype: {etype}. Supported types are: 'cell' and 3.")


class QuadrangleMeshDFEDof(LinearMeshDFEDof):
    def __init__(self, mesh, p):
        super(QuadrangleMeshDFEDof, self).__init__(mesh, p)

class HexahedronMeshDFEDof(LinearMeshCFEDof):
    def __init__(self, mesh, p):
        super(HexahedronMeshDFEDof, self).__init__(mesh, p)

class EdgeMeshCFEDof():
    """
    @brief EdgeMesh 上的分片 p 次连续元的自由度管理类
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = mesh.multi_index_matrix(p, etype=mesh.top_dimension()) 

    def is_boundary_dof(self, threshold=None):
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_node_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('node', index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[index] = True
        return isBdDof

    def entity_to_dof(self, 
            etype: Union[str, int]='cell', 
            index: Union[np.ndarray, slice]=np.s_[:]
            ) -> np.ndarray:
        if etype in {'cell', 'edge', 1}:
            return self.cell_to_dof()[index]
        elif etype in {'node', 'face', 0}:
            NN = self.mesh.number_of_nodes()
            return np.arange(NN)[index]
        else:
            raise ValueError(f"Unsupported etype: {etype}. Supported types are: 'cell', 'edge', 1, 'node', 'face', and 0.")

    def cell_to_dof(self, index=np.s_[:]):
        return self.mesh.cell_to_ipoint(self.p)[index]

    def number_of_local_dofs(self, doftype='cell'):
        return self.mesh.number_of_local_ipoints(self.p, iptype=doftype)

    def number_of_global_dofs(self):
        return self.mesh.number_of_global_ipoints(self.p)

    def interpolation_points(self):
        return self.mesh.interpolation_points(self.p)

class EdgeMeshDFEDof():
    """
    @brief EdgeMesh 上的分片 p 次间断元的自由度管理类
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = mesh.multi_index_matrix(p, etype=mesh.top_dimension()) 

    def is_boundary_dof(self, threshold=None):
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_node_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('node', index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[index] = True
        return isBdDof

    def entity_to_dof(self, 
            etype: Union[str, int]='cell', 
            index: Union[np.ndarray, slice]=np.s_[:]
            ) -> np.ndarray:
        if etype in {'cell', 'edge', 1}:
            return self.cell_to_dof()[index] # TODO:
        else:
            raise ValueError(f"Unsupported etype: {etype}. Supported types are: 'cell', 'edge', and 1.")

    def cell_to_dof(self):
        NC = self.mesh.number_of_cells()
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        ldof = self.number_of_local_dofs()
        cell2dof = np.arange(NC*(p+1)).reshape(NC, p+1)
        return cell2dof

    def number_of_local_dofs(self, doftype='cell'):
        return self.p + 1

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        gdof = NC*(p+1)
        return gdof

    def interpolation_points(self, index=np.s_[:]):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')
        GD = mesh.geo_dimension()

        if p == 0:
            return mesh.entity_barycenter('cell')

        if p == 1:
            return node[cell].reshape(-1, GD)

        w = self.multiIndex/p
        ipoint = np.einsum('ij, kj...->ki...', w, node[cell]).reshape(-1, GD)
        return ipoint