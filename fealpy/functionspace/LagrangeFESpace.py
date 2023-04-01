import numpy as np
from typing import Optional, Union, Callable
from .Function import Function
from ..decorator import barycentric

class SimplexMeshCLFEDof():
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = mesh.multi_index_matrix(p) 
        self.cell2dof = self.cell_to_dof()

    def is_boundary_dof(self, threshold=None):
        TD = self.mesh.top_dimension()
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_node_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter(TD-1, index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[index] = True
        return isBdDof

    def face_to_dof(self, index=np.s_[:]):
        return self.mesh.face_to_ipoint(self.p)

    def edge_to_dof(self, index=np.s_[:]):
        return self.mesh.edge_to_ipoint(self.p)

    def cell_to_dof(self, index=np.s_[:]):
        return self.mesh.cell_to_ipoint(self.p)

    def interpolation_points(self):
        return self.mesh.interpolation_points(self.p)

    def number_of_global_dofs(self):
        return self.mesh.number_of_global_ipoints(self.p)

    def number_of_local_dofs(self, doftype='cell'):
        return self.mesh.number_of_local_ipoints(self.p, iptype=doftype)

class IntervalMeshCLFEDof(SimplexMeshCLFEDof):
    def __init__(self, mesh, p: int):
        super(IntervalMeshCLFEDof, self).__init__(mesh, p)

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

class TriangleMeshCLFEDof(SimplexMeshCLFEDof):
    def __init__(self, mesh, p):
        super(TriangleMeshCLFEDof, self).__init__(mesh, p)

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

class TetrahedronMeshCLFEDof(SimplexMeshCLFEDof):
    def __init__(self, mesh, p):
        super(TetrahedronMeshCLFEDof, self).__init__(mesh, p)

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

class SimplexMeshDLFEDof():
    """
    间断单元自由度管理基类.
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        if p > 0:
            self.multiIndex = mesh.multi_index_matrix(self.p)
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


class IntervalMeshDLFEDof(SimplexMeshDLFEDof):
    """
    区间间断单元自由度管理类.
    """
    def __init__(self, mesh, p):
        super(IntervalMeshDLFEDof, self).__init__(mesh, p)

    def entity_to_dof(self, 
            etype: Union[str, int]='cell', 
            index: Union[np.ndarray, slice]=np.s_[:]
            ) -> np.ndarray:
        if etype in {'cell', 'edge', 1}:
            return self.cell_to_dof()[index]
        else:
            raise ValueError(f"Unsupported etype: {etype}. Supported types are: 'cell', 'edge', and 1.")

class TriangleMeshDLFEDof(SimplexMeshDLFEDof):
    """
    三角形间断单元自由度管理类.
    """
    def __init__(self, mesh, p):
        super(TriangleMeshDLFEDof, self).__init__(mesh, p)

    def entity_to_dof(self, 
            etype: Union[str, int]='cell', 
            index: Union[np.ndarray, slice]=np.s_[:]
            ) -> np.ndarray:
        if etype in {'cell',  2}:
            return self.cell_to_dof()[index]
        else:
            raise ValueError(f"Unsupported etype: {etype}. Supported types are: 'cell' and 2.")


class TetrahedronMeshDLFEDof(SimplexMeshDLFEDof):
    """
    四面体间断单元自由度管理类.
    """
    def __init__(self, mesh, p):
        super(TetrahedronMeshDLFEDof, self).__init__(mesh, p)

    def entity_to_dof(self, 
            etype: Union[str, int]='cell', 
            index: Union[np.ndarray, slice]=np.s_[:]
            ) -> np.ndarray:
        if etype in {'cell', 3}:
            return self.cell_to_dof()[index]
        else:
            raise ValueError(f"Unsupported etype: {etype}. Supported types are: 'cell' and 3.")


class EdgeMeshCLFEDof():
    """
    @brief EdgeMesh 上的分片 p 次连续元的自由度管理类
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = mesh.multi_index_matrix(p) 

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
        return self.mesh.cell_to_ipoint(self.p)

    def number_of_local_dofs(self, doftype='cell'):
        return self.mesh.number_of_local_ipoints(iptype=doftype)

    def number_of_global_dofs(self):
        return self.mesh.number_of_global_ipoints()

    def interpolation_points(self):
        return self.mesh.interpolation_points(self.p)

class EdgeMeshDLFEDof():
    """
    @brief EdgeMesh 上的分片 p 次间断元的自由度管理类
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = mesh.multi_index_matrix() 

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
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        ldof = self.number_of_local_dofs()
        cell2dof = np.arange(NC*(p+1)).reshape(NC, p+1)
        return cell2dof

    def number_of_local_dofs(self, _):
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

class LagrangeFESpace():
    DOF = { 'C': {
                "IntervalMesh": IntervalMeshCLFEDof,
                "TriangleMesh": TriangleMeshCLFEDof,
                "TetrahedronMesh": TetrahedronMeshCLFEDof,
                "EdgeMesh": EdgeMeshCLFEDof,
                }, 
            'D':{
                "IntervalMesh": IntervalMeshDLFEDof,
                "TriangleMesh": TriangleMeshDLFEDof,
                "TetrahedronMesh": TetrahedronMeshDLFEDof,
                "EdgeMesh": EdgeMeshDLFEDof, 
                }
        } 
        
    def __init__(self, 
            mesh, 
            p: int=1, 
            spacetype: str='C', 
            doforder: str='nodes'):
        """
        @brief Initialize the Lagrange finite element space.

        @param mesh The mesh object.
        @param p The order of interpolation polynomial, default is 1.
        @param spacetype The space type, either 'C' or 'D'.
        @param doforder The convention for ordering degrees of freedom in vector space, either 'nodes' (default) or 'vdims'.

        @note 'nodes': x_0, x_1, ..., y_0, y_1, ..., z_0, z_1, ...
              'vdims': x_0, y_0, z_0, x_1, y_1, z_1, ...
        """
        self.mesh = mesh
        self.p = p
        assert spacetype in {'C', 'D'} 
        self.spacetype = spacetype
        self.doforder = doforder

        mname = type(mesh).__name__
        self.dof = self.DOF[spacetype][mname](mesh, p)
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

        self.cellmeasure = mesh.entity_measure('cell')
        self.itype = mesh.itype
        self.ftype = mesh.ftype

    def __str__(self):
        return "Lagrange finite element space on linear mesh!"

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='cell'):
        if self.spacetype == 'C':
            return self.dof.number_of_local_dofs(doftype=doftype)
        elif self.spacetype == 'D':
            return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def cell_to_dof(self, index=np.s_[:]):
        return self.dof.cell2dof[index]

    def face_to_dof(self, index=np.s_[:]):
        return self.dof.face_to_dof() #TODO: index

    def edge_to_dof(self, index=np.s_[:]):
        return self.dof.edge_to_dof() #TODO：index

    def is_boundary_dof(self, threshold=None):
        if self.spacetype == 'C':
            return self.dof.is_boundary_dof(threshold=threshold)
        else:
            raise ValueError('This space is a discontinuous space!')

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    @barycentric
    def basis(self, bc, index=np.s_[:]):
        p = self.p
        phi = self.mesh.shape_function(bc, p=p)
        return phi[..., None, :]

    @barycentric
    def grad_basis(self, bc, index=np.s_[:]):
        return self.mesh.grad_shape_function(bc, p=self.p, index=index)

    @barycentric
    def value(self, 
            uh: np.ndarray, 
            bc: np.ndarray, 
            index: Union[np.ndarray, slice]=np.s_[:]
            ) -> np.ndarray:
        """
        @brief Computes the value of a finite element function `uh` at a set of
        barycentric coordinates `bc` for each mesh cell.

        @param uh: numpy.ndarray, the dof coefficients of the basis functions.
        @param bc: numpy.ndarray, the barycentric coordinates with shape (NQ, TD+1).
        @param index: Union[numpy.ndarray, slice], index of the entities (default: np.s_[:]).
        @return numpy.ndarray, the computed function values.

        This function takes the dof coefficients of the finite element function `uh` and a set of barycentric
        coordinates `bc` for each mesh cell. It computes the function values at these coordinates
        and returns the results as a numpy.ndarray.
        """
        gdof = self.number_of_global_dofs()
        phi = self.basis(bc, index=index) # (NQ, NC, ldof)
        cell2dof = self.dof.cell_to_dof(index=index)

        dim = len(uh.shape) - 1
        s0 = 'abdefg'
        if self.doforder == 'nodes':
            # phi.shape == (NQ, NC, ldof)
            # uh.shape == (..., gdof)
            # uh[..., cell2dof].shape == (..., NC, ldof)
            # val.shape == (NQ, ..., NC)
            s1 = f"...ci, {s0[:dim]}ci->...{s0[:dim]}c"
            val = np.einsum(s1, phi, uh[..., cell2dof])
        elif self.doforder == 'vdims':
            # phi.shape == (NQ, NC, ldof)
            # uh.shape == (gdof, ...)
            # uh[cell2dof, ...].shape == (NC, ldof, ...)
            # val.shape == (NQ, NC, ...)
            s1 = f"...ci, ci{s0[:dim]}->...c{s0[:dim]}"
            val = np.einsum(s1, phi, uh[cell2dof, ...])
        else:
            raise ValueError(f"Unsupported doforder: {self.doforder}. Supported types are: 'nodes' and 'vdims'.")
        return val


    @barycentric
    def grad_value(self, 
            uh: np.ndarray, 
            bc: np.ndarray, 
            index: Union[np.ndarray, slice]=np.s_[:]
            ) -> np.ndarray:
        """
        """
        gdof = self.number_of_global_dofs()
        gphi = self.grad_basis(bc, index=index) # (NQ, NC, ldof, GD)
        cell2dof = self.dof.cell_to_dof(index=index)
        dim = len(uh.shape) - 1
        s0 = 'abdefg'
        if self.doforder == 'nodes':
            # gphi.shape == (NQ, NC, ldof, GD)
            # uh.shape == (..., gdof)
            # uh[..., cell2dof].shape == (..., NC, ldof)
            # val.shape == (NQ, ..., GD, NC)
            s1 = '...cim, {}ci->...{}mc'.format(s0[:dim], s0[:dim])
            val = np.einsum(s1, gphi, uh[..., cell2dof])
        elif self.doforder == 'vdims':
            # gphi.shape == (NQ, NC, ldof, GD)
            # uh.shape == (gdof, ...)
            # uh[cell2dof, ...].shape == (NC, ldof, ...)
            # val.shape == (NQ, NC, ..., GD)
            s1 = '...cim, ci{}->...c{}m'.format(s0[:dim], s0[:dim])
            val = np.einsum(s1, gphi, uh[cell2dof[index], ...])
        else:
            raise ValueError(f"Unsupported doforder: {self.doforder}. Supported types are: 'nodes' and 'vdims'.")

        return val

    def set_dirichlet_bc(self, 
            gD: Union[Callable, int, float, np.ndarray], 
            uh: np.ndarray, 
            threshold: Union[Callable, np.ndarray, None]=None) -> np.ndarray:
        """
        @brief Set the first type (Dirichlet) boundary conditions.

        @param gD: boundary condition function or value (can be a callable, int, float, or numpy.ndarray).
        @param uh: numpy.ndarray, FE function uh .
        @param threshold: optional, threshold for determining boundary degrees of freedom (default: None).

        @return numpy.ndarray, a boolean array indicating the boundary degrees of freedom.

        This function sets the Dirichlet boundary conditions for the FE function `uh`. It supports
        different types for the boundary condition `gD`, such as a function, a scalar, or a numpy array.
        """
        ipoints = self.interpolation_points() # TODO: 直接获取过滤后的插值点
        isDDof = self.is_boundary_dof(threshold=threshold)
        if callable(gD):
            uh[isDDof] = gD(ipoints[isDDof]) #TODO: 考虑更多的情况，如gD 是数组 
        elif isinstance(gD, (int, float, np.ndarray)):
            uh[isDDof] = gD 
        else:
            raise ValueError("Unsupported type for gD. Must be a callable, int, float, or numpy.ndarray.")
        return isDDof

    def function(self, dim=None, array=None, dtype=np.float64):
        return Function(self, dim=dim, array=array, 
                coordtype='barycentric', dtype=dtype)

    def array(self, dim=None, dtype=np.float64):
        gdof = self.number_of_global_dofs()
        if dim is None:
            dim = tuple() 
        if type(dim) is int:
            dim = (dim, )

        if self.doforder == 'nodes':
            shape = dim + (gdof, )
        elif self.doforder == 'vdims':
            shape = (gdof, ) + dim

        return np.zeros(shape, dtype=dtype)


