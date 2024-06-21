
##################################################
### Mesh Data Structure Base
##################################################

class MeshDataStructure():
    _STORAGE_ATTR = ['cell', 'face', 'edge', 'cell_location','face_location']
    def __init__(self, NN: int, TD: int) -> None:
        self._entity_storage: Dict[int, Array] = {}
        self.NN = NN
        self.TD = TD

    @overload
    def __getattr__(self, name: EntityName) -> Array: ...
    def __getattr__(self, name: str):
        if name not in self._STORAGE_ATTR:
            return self.__dict__[name]
        etype_dim = entity_str2dim(self, name)
        return entity_dim2array(self, etype_dim)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._STORAGE_ATTR:
            if not hasattr(self, '_entity_storage'):
                raise RuntimeError('please call super().__init__() before setting attributes.')
            etype_dim = entity_str2dim(self, name)
            self._entity_storage[etype_dim] = value
        else:
            super().__setattr__(name, value)

    ### cuda
    def to(self, device: Optional[_device]=None):
        for entity_tensor in self._entity_storage.values():
            jax.device_put(entity_tensor, device)
        return self

    ### properties
    def top_dimension(self) -> int: return self.TD
    @property
    def itype(self) -> _dtype: return self.cell.dtype

    ### counters
    def count(self, etype: Union[int, str]) -> int:
        """_summary_

        Args:
            etype (Union[int, str]): _description_

        Returns:
            int: _description_
        """
        if etype in ('node', 0):
            return self.NN
        if isinstance(etype, str):
            edim = entity_str2dim(self, etype)
        if -edim in self._entity_storage: # for polygon mesh
            return self._entity_storage[-edim].shape[0] - 1
        return entity_dim2array(self, edim).shape[0] # for homogeneous mesh

    def number_of_nodes(self): return self.NN
    def number_of_edges(self): return self.count('edge')
    def number_of_faces(self): return self.count('face')
    def number_of_cells(self): return self.count('cell')

    ### constructors
    def construct(self) -> None:
        raise NotImplementedError

    @overload
    def entity(self, etype: Union[int, str], index: Optional[Index]=None) -> Array: ...
    @overload
    def entity(self, etype: Union[int, str], index: Optional[Index]=None, *, default: _T) -> Union[Array, _T]: ...
    def entity(self, etype: Union[int, str], index: Optional[Index]=None, *, default=_default):
        """Get entities in mesh structure.

        Args:
            etype (int or str): The topology dimension of the entity, or name\
            'cell' | 'face' | 'edge'. Note that 'node' is not in mesh structure.\
            For polygon meshes, the names 'cell_location' | 'face_location' may also be\
            available, and the `index` argument is applied on the flattened entity array.
            index (int, slice or Array): The index of the entity.

        Returns:
            Array or Sequence[Array].
        """
        if isinstance(etype, str):
            etype = entity_str2dim(self, etype)
        return entity_dim2array(self, etype, index, default=default)

    def total_face(self) -> Array:
        raise NotImplementedError

    def total_edge(self) -> Array:
        raise NotImplementedError

    ### topology
    # TODO: add more methods here

    ### boundary
    def boundary_face_flag(self): return self.face2cell[:, 0] == self.face2cell[:, 1]
    def boundary_face_index(self): return jnp.nonzero(self.boundary_face_flag())[0]


class HomoMeshDataStructure(MeshDataStructure):
    ccw: Array
    localEdge: Array
    localFace: Array

    def __init__(self, NN: int, TD: int, cell: Array) -> None:
        super().__init__(NN, TD)
        self.cell = cell

    number_of_vertices_of_cells: _int_func = lambda self: self.cell.shape[-1]
    number_of_nodes_of_cells = number_of_vertices_of_cells
    number_of_edges_of_cells: _int_func = lambda self: self.localEdge.shape[0]
    number_of_faces_of_cells: _int_func = lambda self: self.localFace.shape[0]
    number_of_vertices_of_faces: _int_func = lambda self: self.localFace.shape[-1]
    number_of_vertices_of_edges: _int_func = lambda self: self.localEdge.shape[-1]

    def total_face(self) -> Array:
        NVF = self.number_of_faces_of_cells()
        cell = self.entity(self.TD)
        local_face = self.localFace
        total_face = cell[..., local_face].reshape(-1, NVF)
        return total_face

    def total_edge(self) -> Array:
        NVE = self.number_of_vertices_of_edges()
        cell = self.entity(self.TD)
        local_edge = self.localEdge
        total_edge = cell[..., local_edge].reshape(-1, NVE)
        return total_edge

    def construct(self) -> None:
        raise NotImplementedError


##################################################
### Mesh Base
##################################################

class Mesh():
    """The base class for mesh."""
    ds: MeshDataStructure
    node: Array

    def geo_dimension(self) -> int: return self.node.shape[-1]
    def top_dimension(self) -> int: return self.ds.top_dimension()
    GD = property(geo_dimension)
    TD = property(top_dimension)

    def count(self, etype: Union[int, str]) -> int: return self.ds.count(etype)
    def number_of_cells(self) -> int: return self.ds.number_of_cells()
    def number_of_faces(self) -> int: return self.ds.number_of_faces()
    def number_of_edges(self) -> int: return self.ds.number_of_edges()
    def number_of_nodes(self) -> int: return self.ds.number_of_nodes()

    @staticmethod
    def multi_index_matrix(p: int, etype: int):
        """
        @brief 获取 p 次的多重指标矩阵

        @param[in] p 正整数

        @return multiIndex  ndarray with shape (ldof, TD+1)
        """
        if etype == 3:
            ldof = (p+1)*(p+2)*(p+3)//6
            idx = np.arange(1, ldof)
            idx0 = (3*idx + np.sqrt(81*idx*idx - 1/3)/3)**(1/3)
            idx0 = np.floor(idx0 + 1/idx0/3 - 1 + 1e-4) # a+b+c
            idx1 = idx - idx0*(idx0 + 1)*(idx0 + 2)/6
            idx2 = np.floor((-1 + np.sqrt(1 + 8*idx1))/2) # b+c
            multiIndex = np.zeros((ldof, 4), dtype=np.int_)
            multiIndex[1:, 3] = idx1 - idx2*(idx2 + 1)/2
            multiIndex[1:, 2] = idx2 - multiIndex[1:, 3]
            multiIndex[1:, 1] = idx0 - idx2
            multiIndex[:, 0] = p - np.sum(multiIndex[:, 1:], axis=1)
            return jnp.array(multiIndex)
        elif etype == 2:
            ldof = (p+1)*(p+2)//2
            idx = np.arange(0, ldof)
            idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
            multiIndex = np.zeros((ldof, 3), dtype=np.int_)
            multiIndex[:,2] = idx - idx0*(idx0 + 1)/2
            multiIndex[:,1] = idx0 - multiIndex[:,2]
            multiIndex[:,0] = p - multiIndex[:, 1] - multiIndex[:, 2]
            return jnp.array(multiIndex)
        elif etype == 1:
            ldof = p+1
            multiIndex = np.zeros((ldof, 2), dtype=np.int_)
            multiIndex[:, 0] = np.arange(p, -1, -1)
            multiIndex[:, 1] = p - multiIndex[:, 0]
            return jnp.array(multiIndex)

    def entity(self, etype: Union[int, str], index: Optional[Index]=None) -> Array:
        if etype in ('node', 0):
            return self.node if index is None else self.node[index]
        else:
            return self.ds.entity(etype, index)

    def integrator(self, q: int, etype: Union[int, str]='cell', qtype: str='legendre') -> Quadrature:
        """Get the quadrature points and weights.

        Args:
            q (int): index of the quadrature formula.
            etype (int | str): The type of the entity.
            qtype (str): quadrature type.

        Returns:
            Quadrature.
        """
        raise NotImplementedError

    def edge_unit_tangent(self, index=jnp.s_[:], node: Optional[NDArray]=None):
        """Calculate the tangent vector with unit length of each edge.\
        See `Mesh.edge_tangent`.
        """
        node = self.entity('node') if node is None else node
        edge = self.entity('edge', index=index)
        v = node[edge[:, 1], :] - node[edge[:, 0], :]
        length = jnp.sqrt(jnp.square(v).sum(axis=1))
        return v/length.reshape(-1, 1)

    def shape_function(self, bc: Array, p: int=1, *, index: Array,
                       variable: str='u', mi: Optional[Array]=None) -> Array:
        raise NotImplementedError(f"shape function is not supported by {self.__class__.__name__}")

    def grad_shape_function(self, bc: Array, p: int=1, *, index: Array,
                            variable: str='u', mi: Optional[Array]=None) -> Array:
        raise NotImplementedError(f"grad shape function is not supported by {self.__class__.__name__}")

    def hess_shape_function(self, bc: Array, p: int=1, *, index: Array,
                            variable: str='u', mi: Optional[Array]=None) -> Array:
        raise NotImplementedError(f"hess shape function is not supported by {self.__class__.__name__}")


class HomoMesh(Mesh):
    def entity_barycenter(self, etype: Union[int, str], index: Optional[Index]=None) -> Array:
        node = self.entity('node')
        if etype in ('node', 0):
            return node if index is None else node[index]
        entity = self.ds.entity(etype, index)
        return F.homo_entity_barycenter(entity, node)

    def bc_to_point(self, bcs: Array, etype='cell', index=jnp.s_[:]) -> Array:
        """Convert barycenter coordinate points to cartesian coordinate points\
        on mesh entities.

        Args:
            bc (Array): Barycenter coordinate points array, with shape (NQ, NVC), where\
                NVC is the number of nodes in each entity.
            etype (str | int): Specify the type of entities on which the coordinates\
                be converted.
            index (Array | int | slice): Index to slice entities.

        Note:
            To get the correct result, the order of bc must match the order of nodes\
        in the entity.

        Returns:
            Cartesian coordinate points array, with shape (NQ, GD).
        """
        node = self.entity('node')
        entity = self.ds.entity(etype, index=index)
        p = jnp.einsum('...j, ijk -> ...ik', bcs, node[entity])
        return p

    ### ipoints
    def interpolation_points(self, p: int, index: Index=_S) -> Array:
        raise NotImplementedError

    def edge_to_ipoint(self, p: int, index=_S) -> Array:
        """Fetch the relationship between edges and interpolation points.

        Args:
            p (int): The interpolation order.
            index (Array | int | slice): The index of edges.

        Return:
            Array: An indices array.
        """
        if isinstance(index, slice) and index == slice(None):
            NE = self.number_of_edges()
            index = np.arange(NE)
        elif isinstance(index, np.ndarray) and (index.dtype == np.bool_):
            index, = np.nonzero(index)
            NE = len(index)
        elif isinstance(index, list) and (type(index[0]) is np.bool_):
            index, = np.nonzero(index)
            NE = len(index)
        else:
            NE = len(index)

        edges = self.entity('edge')[index]
        return K.edge_to_ipoint(edges, index, p)

    def face_to_ipoint(self, p: int, index: Index=_S) -> Array:
        raise NotImplementedError

    def cell_to_ipoint(self, p: int, index: Index=_S) -> Array:
        raise NotImplementedError
