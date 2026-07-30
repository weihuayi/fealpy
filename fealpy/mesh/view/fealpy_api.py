from collections.abc import Iterable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from ...backend import bm
from ...backend import Tensor, Index
from ..schema import registry as _Reg
from .mesh_view import MeshView

__all__ = ["Mesh"]


@dataclass
class Mesh(MeshView):
    """FEALPy-compatible view of a mesh block.

    ``FEALPyMesh`` exposes the newer mesh storage and entity-view layer through
    method names used by FEALPy's classical mesh API.  The class is a thin view:
    it keeps the mesh connectivity, geometry, backend tensors, and device
    placement owned by :class:`Mesh` unchanged, while forwarding numerical
    operations to the corresponding entity sectors.

    Attributes:
        ftype: Floating-point dtype of the node coordinate tensor.
        itype: Integer dtype used for connectivity arrays.
        device: Backend device of the node coordinate tensor, if available.
    """
    def __post_init__(self):
        """Initialize FEALPy dtype and device attributes from the mesh block."""
        self.ftype = self.block.positions.dtype

        node_block = self.block.sectors.get("node")
        if node_block is not None:
            self.itype = node_block.indices.dtype
        else:
            self.itype = bm.int32

        self.device = getattr(self.block.positions, "device", None)

    @classmethod
    def read(cls, filename: str | Path, file_format: str | None = None) -> Self:
        """Read a mesh from a file and return a new Mesh instance."""
        from ..mesh_io import read
        block = read(filename, file_format=file_format)
        return cls(block)

    def write(
        self,
        filename: str | Path,
        entity: str | list[str] | None = None,
        file_format: str | None = None,
        **kwargs: Any
    ) -> None:
        """Write selected entity sectors to a mesh file.

        Parameters:
            filename (str | Path): Output file path.
            entity (str | list[str] | None): Name or names of entity to write.
                If ``None``, all top-dimensional entities (cells) are written.
                Default is ``None``.
            file_format (str | None): Format of the output file.
            **kwargs: Additional keyword arguments passed to the meshio writer.
        """
        from ..mesh_io import write
        if entity is None:
            entity_names = _Reg.schema_name_multi_parser(
                "cell", self.top_dimension(), self.block.sectors.keys()
            )
        elif not isinstance(entity, (list, tuple)):
            entity = [entity]
            entity_names: list[str] = []
            for e in entity:
                entity_names.extend(_Reg.schema_name_multi_parser(
                    e, self.top_dimension(), self.block.sectors.keys()
                ))
        return write(filename, self.block, entity_names, file_format=file_format, **kwargs)

    @property
    def localEdge(self) -> Tensor:
        """Local edge-to-vertex table of the cell reference entity.

        Returns:
            Tensor: Integer tensor of shape ``(NEC, NVE)``, where ``NEC`` is
            the number of local edges per cell and ``NVE`` is the number of
            vertices per edge.
        """
        cell_sec = self.Entities(-1)[0]
        edge_sec = self.Entities(1)[0]
        data = cell_sec.schema.local_entity(edge_sec.schema.name)
        return bm.asarray(data, dtype=self.itype, device=self.device)

    @property
    def localFace(self) -> Tensor:
        """Local face-to-vertex table of the cell reference entity.

        Returns:
            Tensor: Integer tensor of shape ``(NFC, NVF)``, where ``NFC`` is
            the number of local faces per cell and ``NVF`` is the number of
            vertices per face.
        """
        cell_sec = self.Entities(-1)[0]
        face_sec = self.Entities(-2)[0]
        data = cell_sec.schema.local_entity(face_sec.schema.name)
        return bm.asarray(data, dtype=self.itype, device=self.device)

    # Meta

    def entity(self, name_or_topdim: str | int, /) -> Tensor:
        """Return entity data in the classical FEALPy layout.

        Node entities return the coordinate tensor.  All other entities return
        their integer connectivity array.

        Parameters:
            name_or_topdim (str | int): Entity selector.  It may be an entity
                type such as ``"cell"``, ``"face"``, ``"edge"`` or
                ``"node"``, a concrete schema name, or a topological dimension.

        Returns:
            Tensor: Node coordinates with shape ``(NN, GD)`` for nodes, or a
            connectivity tensor with shape ``(NE, NVE)`` for other entities.

        See Also:
            - :meth:`Mesh.Entity`
        """
        if name_or_topdim in {"node", "Node", "NODE", 0}:
            return self.block.positions
        return self.Entity(name_or_topdim).indices

    @property
    def cell(self) -> Tensor:
        """Cell connectivity array."""
        return self.entity("cell")

    @property
    def face(self) -> Tensor:
        """Face connectivity array."""
        return self.entity("face")

    @property
    def edge(self) -> Tensor:
        """Edge connectivity array."""
        return self.entity("edge")

    @property
    def node(self) -> Tensor:
        """Node coordinate array."""
        return self.entity("node")

    # [Shape functions]

    def shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...] = 1,
        *,
        index: Index | None = None,
        variables: str = "u",
        mi=None
    ) -> Tensor:
        """Evaluate cell shape functions at barycentric points.

        Parameters:
            bcs (Tensor | tuple[Tensor, ...]): Barycentric coordinates of
                quadrature or interpolation points.  A tuple is used for tensor
                product entities.
            p (int | tuple[int, ...], optional): Polynomial degree.  Tensor
                product entities may use one degree per factor.  Default is 1.
            index (Index, optional): Cell subset on which the values are used.
                The backend indexing convention is followed.
            variables (str, optional): Coordinate convention requested by the
                FEALPy API.  ``"u"`` returns values in reference coordinates;
                ``"x"`` follows the entity-view broadcast convention.
            mi: Optional precomputed multi-index matrix.  Accepted for API
                compatibility; the current implementation delegates to the
                entity view.

        Returns:
            Tensor: Shape-function values at ``bcs``.  The trailing dimension
            enumerates local basis functions.
        """
        return self.Entities(-1)[0].shape_function(
            bcs, p=p, index=index, variables=variables, mi=mi
        )

    cell_shape_function = shape_function

    def face_shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...] = 1,
        *,
        index: Index | None = None,
        variables: str = "u",
        mi=None
    ) -> Tensor:
        """Evaluate face shape functions at barycentric points.

        Parameters:
            bcs (Tensor | tuple[Tensor, ...]): Barycentric coordinates on the
                reference face entity.
            p (int | tuple[int, ...], optional): Polynomial degree.  Default is
                1.
            index (Index, optional): Face subset.
            variables (str, optional): Coordinate convention passed to
                :meth:`EntityView.shape_function`.
            mi: Optional precomputed multi-index matrix, kept for FEALPy API
                compatibility.

        Returns:
            Tensor: Shape-function values.  The trailing dimension enumerates
            local face basis functions.
        """
        return self.Entities(-2)[0].shape_function(
            bcs, p=p, index=index, variables=variables, mi=mi
        )

    def edge_shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...] = 1,
        *,
        index: Index | None = None,
        variables: str = "u",
        mi=None
    ) -> Tensor:
        """Evaluate edge shape functions at barycentric points.

        Parameters:
            bcs (Tensor | tuple[Tensor, ...]): Barycentric coordinates on the
                reference edge entity.
            p (int | tuple[int, ...], optional): Polynomial degree.  Default is
                1.
            index (Index, optional): Edge subset.
            variables (str, optional): Coordinate convention passed to
                :meth:`EntityView.shape_function`.
            mi: Optional precomputed multi-index matrix, kept for FEALPy API
                compatibility.

        Returns:
            Tensor: Shape-function values.  The trailing dimension enumerates
            local edge basis functions.
        """
        return self.Entities(1)[0].shape_function(
            bcs, p=p, index=index, variables=variables, mi=mi
        )

    def grad_shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...] = 1,
        *,
        index: Index | None = None,
        variables: Literal['b', 'u', 'x'] = "u",
        mi=None
    ) -> Tensor:
        """Evaluate gradients of cell shape functions.

        Parameters:
            bcs (Tensor | tuple[Tensor, ...]): Barycentric coordinates where
                gradients are evaluated.
            p (int | tuple[int, ...], optional): Polynomial degree.  Default is
                1.
            index (Index, optional): Cell subset.
            variables ({"b", "u", "x"}, optional): Coordinate system of the
                gradients: barycentric, reference, or Cartesian coordinates.
                Default is ``"u"``.
            mi: Optional precomputed multi-index matrix, accepted for FEALPy API
                compatibility.

        Returns:
            Tensor: Gradient values.  The last axis stores geometric or
            reference coordinate components, depending on ``variables``.
        """
        return self.Entities(-1)[0].grad_shape_function(
            bcs, p=p, index=index, variables=variables, mi=mi
        )

    def number_of_cells(self) -> int:
        """Return the total number of top-dimensional entities."""
        return sum(sec.size() for sec in self.Entities(-1))

    def number_of_faces(self) -> int:
        """Return the total number of codimension-one entities."""
        return sum(sec.size() for sec in self.Entities(-2))

    def number_of_edges(self) -> int:
        """Return the total number of edge entities."""
        return sum(sec.size() for sec in self.Entities(1))

    def number_of_nodes(self) -> int:
        """Return the total number of node entities."""
        return sum(sec.size() for sec in self.Entities(0))

    def number_of_global_ipoints(self, p: int | tuple[int, ...]) -> int:
        """Return the number of global interpolation points of order ``p``.

        Parameters:
            p (int | tuple[int, ...]): Interpolation order.

        Returns:
            int: Number of globally numbered interpolation points, accumulated
            over all entity sectors and their internal interpolation points.
        """
        total = 0
        for name in self.block.sectors:
            sector_view = self.Entity(name)
            total += sector_view.num_multi_index(p, internal=True) * sector_view.size()
        return total

    def number_of_local_ipoints(self, p: int | tuple[int, ...], iptype: str | int = "cell") -> int:
        """Return the number of local interpolation points on one entity.

        Parameters:
            p (int | tuple[int, ...]): Interpolation order.
            iptype (str | int, optional): Entity selector on which local points
                are counted.  Default is ``"cell"``.

        Returns:
            int: Number of local interpolation points for the selected entity
            type.
        """
        return self.Entity(iptype).num_multi_index(p)

    def multi_index_matrix(self, p: int | tuple[int, ...], etype: int | str = "cell") -> Tensor:
        """Return the local multi-index matrix for an entity type.

        Parameters:
            p (int | tuple[int, ...]): Polynomial degree or tensor-product
                degrees.
            etype (int | str, optional): Entity selector.  Default is
                ``"cell"``.

        Returns:
            Tensor: Integer tensor whose rows are multi-indices of local basis
            or interpolation points.
        """
        sec = self.Entity(etype)
        return sec.multi_index_matrix(p)

    def quadrature_formula(self, q: int, name_or_topdim: str | int = "cell", qtype: str = "legendre"):
        """Return a quadrature formula for the selected entity type.

        Parameters:
            q (int): Quadrature order.
            name_or_topdim (str | int, optional): Entity selector.  Default is
                ``"cell"``.
            qtype (str, optional): Quadrature rule family.  Default is
                ``"legendre"``.

        Returns:
            object: FEALPy quadrature object supplied by the entity schema.
        """
        return self.Entity(name_or_topdim).quadrature_formula(q, qtype)

    # Topology

    def cell_to_face(self, src_id: int = 0, dst_id: int = 0) -> Tensor:
        """Return the cell-to-face connectivity map.

        Parameters:
            src_id (int, optional): Cell sector index for mixed meshes.
                Default is 0.
            dst_id (int, optional): Face sector index for mixed meshes.
                Default is 0.

        Returns:
            Tensor: Integer tensor of shape ``(NC, NFC)`` mapping each local
            face of each cell to a global face index.
        """
        return self.Entity("cell", src_id).to("face", dst_id).as_array()

    def cell_to_edge(self, src_id: int = 0, dst_id: int = 0) -> Tensor:
        """Return the cell-to-edge connectivity map.

        Parameters:
            src_id (int, optional): Cell sector index for mixed meshes.
                Default is 0.
            dst_id (int, optional): Edge sector index for mixed meshes.
                Default is 0.

        Returns:
            Tensor: Integer tensor of shape ``(NC, NEC)`` mapping each local
            edge of each cell to a global edge index.
        """
        return self.Entity("cell", src_id).to("edge", dst_id).as_array()

    def face_to_edge(self, src_id: int = 0, dst_id: int = 0) -> Tensor:
        """Return the face-to-edge connectivity map.

        Parameters:
            src_id (int, optional): Face sector index for mixed meshes.
                Default is 0.
            dst_id (int, optional): Edge sector index for mixed meshes.
                Default is 0.

        Returns:
            Tensor: Integer tensor of shape ``(NF, NEF)`` mapping each local
            edge of each face to a global edge index.
        """
        return self.Entity("face", src_id).to("edge", dst_id).as_array()

    def face_to_cell(self, src_id: int = 0, dst_id: int = 0):
        """Return sparse face-to-cell adjacency.

        Parameters:
            src_id (int, optional): Face sector index for mixed meshes.
                Default is 0.
            dst_id (int, optional): Cell sector index for mixed meshes.
                Default is 0.

        Returns:
            Tensor: a 4-column integer tensor whose rows are:
                - The left/front cell index of the face;
                - The right/back cell index of the face;
                - The local index of the face in the left/front cell;
                - The local index of the face in the right/back cell.
        """
        rel = self.Entity("face", src_id).to("cell", dst_id)
        data = rel.unique
        lidx = rel.local_index
        return bm.stack([data.first, data.last, lidx.floc, lidx.lloc], axis=1)

    def boundary_cell_flag(self) -> Tensor:
        """Return a boolean mask marking boundary cells."""
        return self.Entity(-1).boundary().mask

    def boundary_face_flag(self) -> Tensor:
        """Return a boolean mask marking boundary faces."""
        return self.Entity(-2).boundary().mask

    def boundary_edge_flag(self) -> Tensor:
        """Return a boolean mask marking boundary edges."""
        return self.Entity(1).boundary().mask

    def boundary_node_flag(self) -> Tensor:
        """Return a boolean mask marking boundary nodes."""
        return self.Entity(0).boundary().mask

    def boundary_cell_index(self) -> Tensor:
        """Return global indices of boundary cells."""
        return self.Entity(-1).boundary().index

    def boundary_face_index(self) -> Tensor:
        """Return global indices of boundary faces."""
        return self.Entity(-2).boundary().index

    def boundary_edge_index(self) -> Tensor:
        """Return global indices of boundary edges."""
        return self.Entity(1).boundary().index

    def boundary_node_index(self) -> Tensor:
        """Return global indices of boundary nodes."""
        return self.Entity(0).boundary().index

    def cell_to_edge_sign(self) -> Tensor: # TODO: remove implementation here
        """Return orientation signs of local cell edges.

        Returns:
            Tensor: Boolean tensor of shape ``(NC, NEC)``.  An entry is ``True``
            when the local cell-edge vertex ordering agrees with the global
            edge ordering, and ``False`` otherwise.
        """
        cell_sec = self.Entities(-1)[0]
        edge_sec = self.Entities(1)[0]
        c2e = self.cell_to_edge()

        local_pair = cell_sec.indices[:, self.localEdge]
        global_pair = edge_sec.indices[c2e]

        return bm.all(local_pair == global_pair, axis=-1)

    def face_to_edge_sign(self) -> Tensor: # TODO: remove implementation here
        """Return orientation signs of local face edges.

        Returns:
            Tensor: Boolean tensor of shape ``(NF, NEF)``.  An entry is ``True``
            when the local face-edge orientation agrees with the global edge
            orientation, and ``False`` otherwise.
        """
        face_sec = self.Entities(-2)[0]
        edge_sec = self.Entities(1)[0]
        f2e = self.face_to_edge()
        sign = bm.zeros((face_sec.indices.shape[0], 3), dtype=bm.bool)
        local_f2e = face_sec.schema.local_entity("segment")
        n = [item[0] for item in local_f2e]

        for i in range(len(n)):
            sign[:, i] = face_sec.indices[:, n[i]] == edge_sec.indices[f2e[:, i], 0]

        return sign

    def cell_to_ipoint(self, p: int, *, index: Index | None = None) -> Tensor:
        """Map cells to global interpolation-point indices.

        Parameters:
            p (int): Interpolation order.
            index (Index, optional): Cell subset.

        Returns:
            Tensor: Integer tensor of shape ``(NC, NIP)`` or the indexed subset,
            where ``NIP`` is the number of local interpolation points per cell.
        """
        from ..ipoints import to_ipoint
        view = self.Entities(-1)[0]
        result = to_ipoint(self, view.schema.name, p)
        return result if index is None else result[index]

    def face_to_ipoint(self, p: int, *, index: Index | None = None) -> Tensor:
        """Map faces to global interpolation-point indices.

        Parameters:
            p (int): Interpolation order.
            index (Index, optional): Face subset.

        Returns:
            Tensor: Integer tensor of shape ``(NF, NIP)`` or the indexed subset,
            where ``NIP`` is the number of local interpolation points per face.
        """
        from ..ipoints import to_ipoint
        view = self.Entities(-2)[0]
        result = to_ipoint(self, view.schema.name, p)
        return result if index is None else result[index]

    def interpolation_points(
        self,
        p: int,
        entity: str | int | Iterable[str] | Iterable[int] | None = None,
        index: Index | None = None
    ) -> Tensor:
        """Return coordinates of global interpolation points.

        Parameters:
            p (int): Interpolation order.  Must be positive.
            entity (str | int | Iterable[str] | Iterable[int] | None, optional):
                Entity selector or selectors that contribute interpolation
                points.  If ``None``, all topological dimensions from nodes to
                cells are included.
            index (Index, optional): Interpolation-point subset applied to the
                first axis of the result.

        Returns:
            Tensor: Coordinate tensor of shape ``(NGIP, GD)`` or the indexed
            subset, where ``NGIP`` is the number of selected global
            interpolation points and ``GD`` is the geometric dimension.
        """
        from ..ipoints import ipoints
        names: list[str] = []
        if entity is None:
            entity = range(self.top_dimension() + 1)

        if isinstance(entity, Iterable) and not isinstance(entity, str):
            for e in entity:
                names.extend(_Reg.schema_name_multi_parser(e, self.top_dimension(), self.block.sectors.keys()))
        else:
            names.append(_Reg.schema_name_single_parser(entity, 0, self.top_dimension(), self.block.sectors.keys()))

        ips = ipoints(self, p, names)

        if index is None:
            return ips
        return ips[index, :]

    # Geometry

    def bc_to_point(self, bc: Tensor | tuple[Tensor, ...], *, index: Index | None = None) -> Tensor:
        """Convert barycentric coordinates to physical points.

        The target entity dimension is inferred from the barycentric coordinate
        widths.  For example, a barycentric array with last dimension ``TD + 1``
        selects entities of topological dimension ``TD``.

        Parameters:
            bc (Tensor | tuple[Tensor, ...]): Barycentric coordinates.  A tuple
                represents tensor-product coordinates.
            index (Index, optional): Entity subset.

        Returns:
            Tensor: Physical coordinates in geometric dimension ``GD``.
        """
        if not isinstance(bc, tuple):
            bc = (bc,)

        top = sum(b.shape[1] - 1 for b in bc)
        sec = self.Entities(top)[0]
        return sec.bc_to_point(bc, index=index)

    def entity_barycenter(self, name_or_topdim: str | int, /, *, index: Index | None = None) -> Tensor:
        """Return barycenters of selected entities.

        Parameters:
            name_or_topdim (str | int): Entity selector.
            index (Index, optional): Entity subset.

        Returns:
            Tensor: Barycenter coordinates with shape ``(NE, GD)`` or the
            indexed subset.
        """
        return self.Entity(name_or_topdim).barycenter(index=index)

    def entity_measure(self, name_or_topdim: str | int, /, *, index: Index | None = None) -> Tensor:
        """Return measures of selected entities.

        Parameters:
            name_or_topdim (str | int): Entity selector.
            index (Index, optional): Entity subset.

        Returns:
            Tensor: Measure tensor with one value per selected entity.  The
            measure is length for edges, area for faces, and volume for cells.
        """
        return self.Entity(name_or_topdim).measure(index=index)

    def edge_tangent(self, *, index: Index | None = None) -> Tensor:
        """Return edge tangent vectors.

        Parameters:
            index (Index, optional): Edge subset.

        Returns:
            Tensor: Tangent vectors with shape ``(NE, GD)`` or the indexed
            subset.
        """
        block = self.Entities(1)[0]
        return block.tangent(index=index)[:, 0, :]

    def edge_unit_tangent(self, *, index: Index | None = None) -> Tensor:
        """Return unit tangent vectors of edges.

        Parameters:
            index (Index, optional): Edge subset.

        Returns:
            Tensor: Unit tangent vectors with shape ``(NE, GD)`` or the indexed
            subset.
        """
        block = self.Entities(1)[0]
        tangent = block.tangent(index=index)[:, 0, :]
        norm = bm.linalg.vector_norm(tangent, axis=1, keepdims=True) # type: ignore
        return tangent / norm

    def error(
        self,
        f1: Callable[..., Tensor],
        f2: Callable[..., Tensor],
        /,
        power: float = 2.0,
        q: int = 3,
        *,
        cell_axis: bool = False,
        index: Index | None = None
    ) -> Tensor:
        """Compute the mesh norm of the difference between two functions.

        The error is evaluated on top-dimensional entities using quadrature.
        Cartesian functions are converted to barycentric-coordinate functions by
        the underlying entity view when necessary.

        Parameters:
            f1 (Callable[..., Tensor]): First function.
            f2 (Callable[..., Tensor]): Second function.
            power (float, optional): Power of the integral norm.  Default is
                2.0, corresponding to an L2-type error.
            q (int, optional): Quadrature order.  Default is 3.
            cell_axis (bool, optional): If ``True``, return one error value per
                selected cell.  If ``False``, return the global error.  Default
                is ``False``.
            index (Index, optional): Cell subset.

        Returns:
            Tensor: Scalar global error or a tensor of cell-wise errors.
        """
        cell_sec = self.Entities(-1)[0]
        return cell_sec.error(f1, f2, power=power, q=q, cell_axis=cell_axis, index=index)

    def face_normal(self, *, index: Index | None = None) -> Tensor:
        """Return face normal vectors.

        Parameters:
            index (Index, optional): Face subset.

        Returns:
            Tensor: Normal vectors with shape ``(NF, GD)`` or the indexed
            subset.
        """
        block = self.Entities(self.top_dimension() - 1)[0]
        return block.normal(index=index)[:, 0, :]

    def face_unit_normal(self, *, index: Index | None = None) -> Tensor:
        """Return unit normal vectors of faces.

        Parameters:
            index (Index, optional): Face subset.

        Returns:
            Tensor: Unit normal vectors with shape ``(NF, GD)`` or the indexed
            subset.
        """
        block = self.Entities(self.top_dimension() - 1)[0]
        normal = block.normal(index=index)[:, 0, :]
        norm = bm.linalg.vector_norm(normal, axis=1, keepdims=True) # type: ignore
        return normal / norm

    def grad_lambda(
        self,
        index: Index | None = None,
        TD: int | None = None,
    ) -> Tensor:
        """Return gradients of barycentric coordinate functions.

        Parameters:
            index (Index, optional): Entity subset.
            TD (int | None, optional): Topological dimension of the entity on
                which gradients are requested.  If ``None``, the mesh top
                dimension is used.

        Returns:
            Tensor: Gradients of barycentric coordinates for the selected
            entity dimension.

        Raises:
            ValueError: If ``TD`` is neither the mesh top dimension,
                codimension-one dimension, nor 1.
        """
        if TD is None:
            TD = self.top_dimension()

        if TD == self.top_dimension():
            block = self.Entities(self.top_dimension())[0]
        elif TD == self.top_dimension() - 1:
            block = self.Entities(self.top_dimension() - 1)[0]
        elif TD == 1:
            block = self.Entities(1)[0]
        else:
            raise ValueError(f"Unsupported top dimension: {TD}")

        return block.grad_lambda(index=index)

    def grad_face_lambda(self, index: Index | None = None):
        """Return gradients of face barycentric coordinate functions.

        Parameters:
            index (Index, optional): Face subset.

        Returns:
            Tensor: Gradients of barycentric coordinates on codimension-one
            entities.
        """
        return self.grad_lambda(index=index, TD=self.top_dimension() - 1)

    # Plot

    @property
    def add_plot(self):
        """Provides a plotting interface for the mesh."""
        from ..plotting.classic import MeshPloter
        return MeshPloter(self)

    def find_node(
        self,
        ax,
        color = '#990000',
        showindex: bool = False,
        multiindex = None,
        fontcolor: str = 'k',
        fontsize: int = 24
    ):
        from ..plotting.classic import EntityFinder
        return EntityFinder(self)(
            ax,
            etype='node',
            color=color,
            showindex=showindex,
            multiindex=multiindex,
            fontcolor=fontcolor,
            fontsize=fontsize
        )

    def find_edge(
        self,
        ax,
        color = '#009900',
        showindex: bool = False,
        multiindex = None,
        fontcolor: str = 'k',
        fontsize: int = 24
    ):
        from ..plotting.classic import EntityFinder
        return EntityFinder(self)(
            ax,
            etype='edge',
            color=color,
            showindex=showindex,
            multiindex=multiindex,
            fontcolor=fontcolor,
            fontsize=fontsize
        )

    def find_face(
        self,
        ax,
        color = '#009999',
        showindex: bool = False,
        multiindex = None,
        fontcolor: str = 'k',
        fontsize: int = 24
    ):
        from ..plotting.classic import EntityFinder
        return EntityFinder(self)(
            ax,
            etype='face',
            color=color,
            showindex=showindex,
            multiindex=multiindex,
            fontcolor=fontcolor,
            fontsize=fontsize
        )

    def find_cell(
        self,
        ax,
        color = '#0000CC',
        showindex: bool = False,
        multiindex = None,
        fontcolor: str = 'k',
        fontsize: int = 24
    ):
        from ..plotting.classic import EntityFinder
        return EntityFinder(self)(
            ax,
            etype='cell',
            color=color,
            showindex=showindex,
            multiindex=multiindex,
            fontcolor=fontcolor,
            fontsize=fontsize
        )
