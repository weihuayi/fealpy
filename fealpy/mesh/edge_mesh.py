from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .mesh_base import SimplexMesh
from .plot import Plotable


class EdgeMesh(SimplexMesh, Plotable):
    """
    A class for 1D edge element meshes in finite element analysis.

    EdgeMesh represents and manipulates simple element meshes composed of nodes and edges, primarily used for 1D structures like trusses and beams. 
    It provides geometric information, interpolation point generation, edge length and tangent calculations, and supports visualization and basic mesh operations. 
    This class inherits from SimplexMesh and Plotable, enabling advanced mesh functionalities and visualization capabilities.

    Parameters
    node : TensorLike
        Array of node coordinates, shape (num_nodes, geo_dimension).
    cell : TensorLike
        Array of element connectivity, shape (num_cells, 2), each row represents the indices of the two endpoints of an edge.

    Attributes
    node : TensorLike
        Node coordinate array.
    cell : TensorLike
        Element connectivity array.
    nodedata : dict
        Dictionary for node-related data.
    facedata : dict
        Dictionary for face-related data (shared with nodedata).
    celldata : dict
        Dictionary for cell-related data.
    edgedata : dict
        Dictionary for edge-related data (shared with celldata).
    meshdata : dict
        Dictionary for mesh-level data.
    cell_length : callable
        Method to compute the length of elements (edges).
    cell_tangent : callable
        Method to compute the tangent vector of elements (edges).

    Methods
    cell_to_ipoint(p, index)
        Return interpolation point coordinates on the element.
    face2cell
        Return mapping from nodes to elements.
    ref_cell_measure()
        Return the measure (length) of the reference element.
    ref_face_measure()
        Return the measure of the reference face.
    quadrature_formula(q, etype)
        Return the Gaussian quadrature formula of specified order.
    edge_tangent(index)
        Compute the tangent vector of edges.
    edge_length(index)
        Compute the length of edges.
    entity_measure(etype, index, node)
        Return the measure of the specified entity type.
    grad_lambda(index)
        Compute the derivatives of barycentric coordinate functions.
    number_of_local_ipoints(p, iptype)
        Return the number of local interpolation points.
    number_of_global_ipoints(p)
        Return the number of global interpolation points.
    interpolation_points(p, index)
        Return the coordinates of interpolation points.
    face_unit_normal(index, node)
        Compute the unit normal vector of faces (nodes) (not implemented).
    cell_normal(index, node)
        Compute the normal vector of elements (in 2D).
    from_triangle_mesh(mesh)
        Generate edge mesh from triangle mesh (not implemented).
    from_tetrahedron_mesh(mesh)
        Generate edge mesh from tetrahedron mesh (not implemented).
    from_tower()
        Generate edge mesh for tower structure.
    from_four_bar_mesh()
        Generate edge mesh for four-bar mechanism.
    generate_balcony_truss_mesh()
        Generate edge mesh for balcony truss structure.
    from_simple_3d_truss()
        Generate edge mesh for simple 3D truss.
    generate_cantilevered_mesh()
        Generate edge mesh for cantilever beam structure.
    generate_tri_beam_frame_mesh()
        Generate edge mesh for tri-beam frame structure.
    plane_frame()
        Generate edge mesh for plane frame structure.

    Notes    
    This class is suitable for finite element modeling of 1D structures and supports rapid generation of various typical structures.
    Some methods depend on external libraries (such as bm), and some advanced features need to be implemented in subclasses or externally.

    Examples
    >>> node = bm.tensor([[0, 0], [1, 0], [2, 0]], dtype=bm.float64)
    >>> cell = bm.tensor([[0, 1], [1, 2]], dtype=bm.int32)
    >>> mesh = EdgeMesh(node, cell)
    >>> print(mesh.cell_length())
    [1.0, 1.0]
    """
    def __init__(self, node, cell):
        super().__init__(TD=1, itype=cell.dtype, ftype=node.dtype)
        self.node = node
        self.cell = cell

        self.nodedata = {}
        self.facedata = self.nodedata
        self.celldata = {}
        self.edgedata = self.celldata 
        self.meshdata = {}

        self.cell_length = self.edge_length
        self.cell_tangent = self.edge_tangent
        
        
        
        
    
    def cell_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        """
        Get the interpolation point indices for the specified cell.
        Calls the edge_to_ipoint method to return the interpolation point indices for the given cell order p and index.
        Parameters
            p : int
            Cell order, specifies the interpolation point order for the cell.
            index : Index, optional, default=_S
            Indices of the cells, default is _S (all).
        Returns
            ipoint : TensorLike
            Indices of the interpolation points, type TensorLike, representing the set of interpolation points associated with the cell(s).
        Notes
            This method is typically used to query the mapping between cells and interpolation points in finite element meshes, facilitating subsequent interpolation or numerical computations.
        """
        return self.edge_to_ipoint(p, index)
    

    @property
    def face2cell(self):
        """
        Return mapping from faces to cells.
        This property provides a mapping from the nodes of the mesh to the cells (edges) they belong to.
        It is equivalent to the node_to_cell method, which returns the mapping of nodes to cells.
        Returns
            TensorLike
            A tensor representing the mapping from nodes to cells (edges).
        Notes
            This mapping is useful for understanding the connectivity of the mesh and how nodes are associated with edges.
        """
        return self.node_to_cell()


    def ref_cell_measure(self):
        """
        Return the measure (length) of the reference cell.
        This function returns the measure of the reference cell used in the mesh, 
        which is typically 1.0 for a standardized reference edge.
        Returns
        measure : float
            The measure (length) of the reference cell. For a reference edge, this is 1.0.
        Notes
        This value is used for scaling and integration over the reference cell in finite element computations.
        """
        return 1.0
    
    def ref_face_measure(self):
        """
        Return the measure of the reference face.
        This function returns the measure of the reference face used in the mesh,
        which is typically 1.0 for a standardized reference edge.
        Returns
        measure : float
            The measure of the reference face. For a reference edge, this is 1.0.
        Notes
        This value is used for scaling and integration over the reference face in finite element computations.
        """
        return 0.0
    
    def quadrature_formula(self, q: int, etype: Union[str, int]='cell'):
        """
        Return the Gaussian quadrature formula for the specified order and entity type.
        Parameters
        q : int
            The order of the quadrature formula.
            etype : Union[str, int], optional
            The type of entity for which the quadrature formula is defined. 
            Default is 'cell', which refers to the cell (edge) entity.
        Returns
        quadrature : GaussLegendreQuadrature
            An instance of GaussLegendreQuadrature representing the quadrature formula for the specified order and entity type.
        Notes
        This method is used to obtain the quadrature points and weights for numerical integration over the specified entity type.
        If etype is 'cell', it returns the quadrature formula for the edge elements.
        """
        from ..quadrature import GaussLegendreQuadrature
        return GaussLegendreQuadrature(q, dtype=self.ftype, device=self.device)

    def edge_tangent(self,index = None):
        """
        Compute the tangent vector of edges.

        This method calculates the tangent vectors for the specified edges in the mesh.
        The tangent vector is defined as the normalized direction vector from the first node
        to the second node of each edge.

        Parameters
        index : optional
            Indices of the edges for which to compute the tangent vectors. If None, computes for all edges.

        Returns
        tangent : TensorLike
            An array of tangent vectors for the specified edges. Shape: (num_edges, geo_dimension).

        Notes
        The tangent vector is useful for geometric computations and finite element formulations
        involving edge orientation.
        """
        edge = self.entity('edge', index=index)
        node = self.entity('node')
        return bm.edge_tangent(edge, node)

    def edge_length(self, index=None):
        """
        Compute the length of edges.
        This method calculates the lengths of the specified edges in the mesh.
        The length is computed as the Euclidean distance between the two nodes of each edge.
        Parameters
        index : optional
            Indices of the edges for which to compute the lengths. If None, computes for all edges.
        Returns
        length : TensorLike
            An array of lengths for the specified edges. Shape: (num_edges,).
        Notes
        The edge length is used in finite element analysis for scaling and integration purposes.
        """
        edge = self.entity('edge', index=index)
        node = self.entity('node')
        return bm.edge_length(edge, node)
    
    def entity_measure(self, etype: Union[int, str]='cell', index=None, node=None):
        """
        Compute the measure (length or placeholder) of a specified mesh entity.
        This function returns the measure of a given entity type in the mesh, such as cell (edge) or node (face). For cells/edges, it returns their length. For nodes/faces, it returns a tensor with value 0.0 as a placeholder.
        Parameters
        etype : int or str, optional, default='cell'
            The type of entity whose measure is to be computed. Supported values are 1, 'cell', 'edge' for edges/cells, and 0, 'face', 'node' for nodes/faces.
        index : array-like or None, optional, default=None
            Indices of the entities for which the measure is computed. If None, measures for all entities of the specified type are computed.
        node : array-like or None, optional, default=None
            Node information, if required by the implementation. Default is None.
        Returns
        measure : tensor
            The measure of the specified entity type. For cells/edges, returns their length as a tensor. For nodes/faces, returns a tensor with value 0.0.
        Raises
        ValueError
            If the provided entity type `etype` is not recognized.
        Notes
        The function distinguishes between edge/cell and node/face types. For unsupported types, an exception is raised.
        Examples
        >>> mesh.entity_measure('cell')
        tensor([...])  # Lengths of all cells/edges
        >>> mesh.entity_measure('node')
        tensor([0.0])
        """
        if etype in {1, 'cell', 'edge'}:
            return self.cell_length(index=index)
        elif etype in {0, 'face', 'node'}:
            return bm.tensor([0.0], dtype=self.ftype)
        else:
            raise ValueError(f"entity type: {etype} is wrong!")
        
    def grad_lambda(self, index=None):
        """
        Compute the derivatives of barycentric coordinate functions for edges.
        This method calculates the derivatives of the barycentric coordinate functions for the edges of the mesh.
        The barycentric coordinates are used to interpolate values along the edges of the mesh.
        Parameters
        index : optional
            Indices of the edges for which to compute the derivatives. If None, computes for all edges.
        Returns
        Dlambda : TensorLike
            An array of shape (num_edges, 2, geo_dimension) containing the derivatives of the barycentric coordinate functions for each edge.
            The first dimension corresponds to the two nodes of the edge, and the second dimension corresponds to the spatial dimensions.
        Notes
        The barycentric coordinates are defined such that they are linear functions of the position along the edge.
        The derivatives are computed as the difference between the coordinates of the two nodes of each edge, normalized by the length of the edge.
        """
        node = self.entity('node')
        cell = self.entity('cell', index=index)
        v = node[cell[:, 1]] - node[cell[:, 0]]
        NC = len(cell) 
        GD = self.geo_dimension()
        h2 = bm.sum(v**2, axis=-1)
        v /=h2.reshape(-1, 1)
        v = v[:,None,:]
        Dlambda = bm.concatenate([-v,v],axis=1)
        return Dlambda
    
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell') -> int:
        """
        Return the number of local interpolation points for the specified order and entity type.
        Parameters
        p : int
            The order of the interpolation points.
        iptype : int or str, optional, default='cell'

            The type of entity for which the interpolation points are defined.
            Supported values are 1, 'cell', 'edge' for edges/cells, and 0, 'face', 'node' for nodes/faces.
        Returns
        n_ipoints : int
            The number of local interpolation points for the specified order and entity type.
        Raises
        ValueError
            If the provided entity type `iptype` is not recognized.
        Notes
        The function distinguishes between edge/cell and node/face types. For edges/cells, it returns p+1, indicating the number of interpolation points along the edge.
        For nodes/faces, it returns 1, indicating a single interpolation point at the node.
        Examples
        >>> mesh.number_of_local_ipoints(1, 'cell')
        2
        """
        return p + 1
    
    def number_of_global_ipoints(self, p: int) -> int:
        """
        Return the number of global interpolation points for the specified order.
        Parameters
        p : int
            The order of the interpolation points.
        Returns
        n_ipoints : int
            The total number of global interpolation points for the specified order.
        Notes
        The total number of global interpolation points is calculated as the sum of the number of nodes and the additional interpolation points introduced by the order p.
        For a 1D edge mesh, this is given by the formula:
        n_ipoints = number_of_nodes + (p - 1) * number_of_cells
        where `number_of_nodes` is the number of nodes in the mesh and `number_of_cells` is the number of edges (cells).
        """
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        return NN + (p-1)*NC
    
    def interpolation_points(self, p: int, index=None):
        """
        Return the coordinates of interpolation points for the specified order and index.
        Parameters
        p : int
            The order of the interpolation points.
        index : optional
            Indices of the cells for which to compute the interpolation points. If None, computes for all cells.
        Returns
        ipoint : TensorLike
            An array of shape (n_ipoints, geo_dimension) containing the coordinates of the interpolation points.
        Notes
        For a 1D edge mesh, if p = 1, it returns the coordinates of the nodes.
        If p > 1, it computes additional interpolation points along the edges based on the specified order.
        The interpolation points are computed as a linear combination of the node coordinates, where the coefficients are determined by the order p.
        The interpolation points are generated by dividing the edge into p equal segments and computing the coordinates accordingly.
        """
        GD = self.geo_dimension()
        node = self.entity('node')

        if p == 1:
            return node
        else:
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            gdof = NN + NC*(p-1)
            cell = self.entity('cell')
            a = bm.arange(p-1,0,-1,dtype=bm.float64)/p
            a = a.reshape(p-1,-1)
            b = bm.arange(1,p,1,dtype=bm.float64)/p
            b = b.reshape(p-1,-1)
            w = bm.concatenate([a,b],axis=1)
            GD = self.geo_dimension()
            cip = bm.einsum('ij, kj...->ki...', w,node[cell]).reshape(-1, GD)
            ipoint = bm.concatenate([node,cip],axis=0)
            return ipoint
    
    def face_unit_normal(self, index=None, node=None):
        """
        Compute the unit normal vector of faces (nodes).
        This method is not implemented for 1D edge meshes, as edges do not have a well-defined normal vector in the same way that faces in higher dimensions do.
        Parameters
        index : optional
            Indices of the faces for which to compute the normal vectors. If None, computes for all faces.
        node : optional
            Node information, if required by the implementation. Default is None.
        Returns
        Raises
        NotImplementedError
        This method raises a NotImplementedError because edges do not have a normal vector in 1D.
        """
        raise NotImplementedError

    def cell_normal(self, index=None, node=None):
        """
        Compute the normal vector of elements (in 2D).
        This method computes the normal vector for edges in a 2D context by taking the tangent vector of the edge and rotating it by 90 degrees.
        Parameters
        index : optional
            Indices of the edges for which to compute the normal vectors. If None, computes for all edges.
        node : optional
            Node information, if required by the implementation. Default is None.
        Returns
        normal : TensorLike
            An array of shape (num_edges, geo_dimension) containing the normal vectors for the specified edges.
            The normal vectors are computed by rotating the tangent vectors by 90 degrees in the 2D plane.
        Notes
        This method assumes that the mesh is 2D and that the edges are represented as line segments in the plane.
        The normal vector is computed by taking the tangent vector of the edge and applying a rotation matrix to obtain the perpendicular direction.
        The rotation matrix used is:
        [[0, -1],
         [1, 0]]
        """
        assert self.geo_dimension() == 2
        v = self.cell_tangent(index=index)
        w = bm.tensor([(0, -1),(1, 0)],dtype=self.ftype)
        return v@w
    
    def vtk_cell_type(self):
        VTK_LINE = 3
        return VTK_LINE
    
    def to_vtk(self, fname=None, etype='edge', index:Index=_S):
        """
        Convert the edge mesh to VTK format and optionally write to a file.
        This method prepares the mesh data for visualization in VTK format, including node coordinates and cell connectivity.
        Parameters
        fname : str, optional
            The name of the output file to write the VTK data. If None, returns the data without writing to a file.
        etype : str, optional, default='edge'
            The type of entity to convert to VTK format. Default is 'edge', which refers to the edges of the mesh.
        index : Index, optional, default=_S
            Indices of the entities to include in the VTK output. If None, includes all entities of the specified type.
        Returns
        node : TensorLike
            An array of node coordinates, shape (num_nodes, geo_dimension).
        cell : TensorLike
            An array of cell connectivity, shape (num_cells, NV).
        cellType : int  
            The VTK cell type for the edges, which is typically 3 for line segments.
        NC : int    
            The number of cells (edges) in the mesh.
        Notes
        If `fname` is provided, the method writes the mesh data to a VTK file using the `write_to_vtu` function.
        If `fname` is None, it returns the node coordinates, cell connectivity, cell type, and number of cells without writing to a file.
        This method is useful for exporting the mesh for visualization in VTK-compatible software, such as ParaView or VisIt.
        Examples
        >>> mesh.to_vtk('mesh.vtu')
        This will write the mesh data to a file named 'mesh.vtu'.
        """
        from fealpy.mesh.vtk_extent import  write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD < 3:
            node = bm.concatenate((node, bm.zeros((node.shape[0], 3-GD), dtype=bm.float64)), axis=1)

        cell = self.entity(etype)[index]
        NV = cell.shape[-1]
        NC = len(cell)

        cell = bm.concatenate((bm.zeros((len(cell), 1), dtype=cell.dtype), cell), axis=1)
        cell[:, 0] = NV

        cellType = self.vtk_cell_type()  # segment
        if fname is None:
            return node, cell.flatten(), cellType, NC
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)
    
    ## @ingroup MeshGenerators
    @classmethod
    def from_triangle_mesh(cls, mesh):
        pass

    ## @ingroup MeshGenerators
    @classmethod
    def from_tetrahedron_mesh(cls, mesh):
        pass

    ## @ingroup MeshGenerators
    @classmethod
    def from_tower(cls):
        node = bm.tensor([
            [-950, 0, 5080], [950, 0, 5080], [-950, 950, 2540], 
            [950, 950, 2540], [950, -950, 2540], [-950, -950, 2540],
            [-2540, 2540, 0], [2540, 2540, 0], [2540, -2540, 0], 
            [-2540, -2540, 0]], dtype=bm.float64)
        cell = bm.tensor([
            [0, 1], [3, 0], [1, 2], [1, 5], [0, 4], 
            [1, 3], [1, 4], [0, 2], [0, 5], [2, 5],
            [4, 3], [2, 3], [4, 5], [2, 9], [6, 5], 
            [8, 3], [7, 4], [6, 3], [2, 7], [9, 4],
            [8, 5], [9, 5], [2, 6], [7, 3], [8, 4]], dtype=bm.int32)
        mesh = cls(node, cell)

        mesh.meshdata['disp_bc'] = (bm.tensor([6, 7, 8, 9], dtype=bm.int32), bm.zeros(3))
        mesh.meshdata['force_bc'] = (bm.tensor([0, 1], dtype=bm.int32), bm.tensor([0, 900, 0]))

        return mesh 
    
    ## @ingroup MeshGenerators
    @classmethod
    def from_four_bar_mesh(cls):
        # 单位为 mm
        node = bm.tensor([
            [0, 0], [400, 0], 
            [400, 300], [0, 300]], dtype=bm.float64)
        cell = bm.tensor([
            [0, 1], [2, 1], 
            [0, 2], [3, 2]], dtype=bm.int32)
        mesh = cls(node, cell)

        # 按分量处理自由度索引
        mesh.meshdata['disp_bc'] = (bm.tensor([0, 1, 3, 6, 7], dtype=bm.int32), bm.zeros(1))
        mesh.meshdata['force_bc'] = (bm.tensor([1, 2], dtype=bm.int32), 
                                     bm.tensor([[2e4, 0], [0, -2.5e4]], dtype=bm.float64))

        return mesh 
    
    ## @ingroup MeshGenerators
    @classmethod
    def generate_balcony_truss_mesh(cls):
        # 单位为英寸 in
        node = bm.tensor([
            [0, 0], [36, 0], 
            [0, 36], [36, 36], [72, 36]], dtype=bm.float64)
        cell = bm.tensor([
            [0, 1], [1, 2], [2, 3],
            [1, 3], [1, 4], [3, 4]], dtype=bm.int32)
        mesh = cls(node, cell)

        mesh.meshdata['disp_bc'] = (bm.tensor([0, 2], dtype=bm.int32), bm.zeros(2))
        mesh.meshdata['force_bc'] = (bm.tensor([3, 4], dtype=bm.int32), bm.tensor([[0, -500], [0, -500]]))

        return mesh 
    
    ## @ingroup MeshGenerators
    @classmethod
    def from_simple_3d_truss(cls):
        # 单位为英寸 in
        node = bm.tensor([
            [0, 0, 36], [72, 0, 0], 
            [0, 0, -36], [0, 72, 0]], dtype=bm.float64)
        cell = bm.tensor([
            [0, 1], [0, 2], [0, 3],
            [1, 2], [1, 3], [2, 3]], dtype=bm.int32)
        mesh = cls(node, cell)

        mesh.meshdata['disp_bc'] = (bm.tensor([6, 7, 8, 9], dtype=bm.int32), bm.zeros(3))
        mesh.meshdata['force_bc'] = (bm.tensor([0, 1], dtype=bm.int32), bm.tensor([0, 900, 0]))

        return mesh 
    
    ## @ingroup MeshGenerators
    @classmethod
    def generate_cantilevered_mesh(cls):
        # Unit m
        node = bm.tensor([
            [0], [5], [7.5]], dtype=bm.float64)
        cell = bm.tensor([
            [0, 1], [1, 2]], dtype=bm.int32)
        mesh = cls(node, cell)

        mesh.meshdata['disp_bc'] = (bm.tensor([0, 1], dtype = bm.int32), bm.zeros(2))
        mesh.meshdata['force_bc'] = (bm.tensor([0, 1, 2], dtype = bm.int32), 
                                     bm.tensor([[-62500, -52083], [-93750, 39062], [-31250, 13021]], dtype = bm.int32))

        return mesh 
    
    ## @ingroup MeshGenerators
    @classmethod
    def generate_tri_beam_frame_mesh(cls):
        # Unit: m
        node = bm.tensor([
            [0, 0.96], [1.44, 0.96], 
            [0, 0], [1.44, 0]], dtype=bm.float64)
        cell = bm.tensor([
            [0, 1], [2, 0], [3, 1]], dtype=bm.int32)
        mesh = cls(node, cell)

        mesh.meshdata['disp_bc'] = (bm.tensor([2, 3], dtype=bm.int32), bm.zeros(3))
        mesh.meshdata['force_bc'] = (bm.tensor([0, 1], dtype=bm.int32), 
                                     bm.tensor([[3000, -3000, -720], 
                                               [0, -3000, 720]], dtype=bm.float64))

        return mesh 
    
    ## @ingroup MeshGenerators
    @classmethod
    def plane_frame(cls):
        # 单位为 m
        node = bm.tensor([[0, 6], [5, 6], [5, 3], [0, 3], [0, 0], [5, 9],
                         [5, 0], [0, 9], [1, 6], [2, 6], [3, 6], [4, 6],
                         [5, 4], [5, 5], [1, 3], [2, 3], [3, 3], [4, 3],
                         [0, 1], [0, 2], [0, 4], [0, 5], [5, 7], [5, 8],
                         [5, 1], [5, 2], [0, 7], [0, 8], [1, 9], [2, 9],
                         [3, 9], [4, 9]])

EdgeMesh.set_ploter('1d')
