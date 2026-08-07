from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Concatenate, final, Literal, ParamSpec, TYPE_CHECKING

from ...backend import bm, Tensor, Index
from ..schema import registry as _Reg
from ..schema.entity_schema import EntityContext, EntitySchema

if TYPE_CHECKING:
    from ..storage import MeshBlock, EntitySector, Relation
    from ..topology.boundary import BoundaryInfo

__all__ = ["EntityView"]

P = ParamSpec("P")


@final
@dataclass(slots=True)
class EntityView:
    """View of one homogeneous mesh entity sector.

    ``EntityView`` binds a mesh block to one entity sector, such as points,
    segments, triangles, tetrahedra, quadrilaterals, or hexahedra.  It provides
    a compact user API for topology, geometry, interpolation, quadrature, and
    entity-to-entity relations while delegating the actual formulas to the
    sector's :class:`EntitySchema` implementation.

    Attributes:
        block: Mesh storage block containing coordinates, sectors, and
            relations.
        sector: Homogeneous entity sector represented by this view.
        schema: Entity schema class associated with ``sector``.
    """
    block: MeshBlock
    sector: EntitySector
    schema: type[EntitySchema] = field(init=False)

    def __post_init__(self) -> None:
        """Bind the entity schema after dataclass initialization."""
        self.schema = self.sector.schema

    def __len__(self) -> int:
        """Return the number of entities in this sector."""
        return self.schema.size(self.context())

    def context(self) -> EntityContext:
        """Return the schema context for this entity view.

        Returns:
            EntityContext: Lightweight container holding the mesh block and the
            current entity sector.
        """
        return EntityContext(self.block, self.sector)

    # User APIs

    def barycentric[**P, R](self, func: Callable[Concatenate[Tensor, P], R], /, *, index: Index | None = None):
        """Wrap a Cartesian-coordinate function as a barycentric function.

        Parameters:
            func (Callable): Function whose first positional argument is a
                physical coordinate tensor.  The expected coordinate shape is
                determined by the entity schema.
            index (Index, optional): Entity subset used when converting
                barycentric coordinates to physical points.

        Returns:
            Callable: Function with the same remaining arguments as ``func``
            whose first positional argument is a barycentric coordinate tensor,
            or a tuple of barycentric coordinate tensors for tensor-product
            entities.
        """
        return self.schema.barycentric(self.context(), func, index)

    def barycenter(self, *, index: Index | None = None) -> Tensor:
        """Return barycenters of selected entities.

        Parameters:
            index (Index, optional): Entity subset.  If ``None``, all entities
                in this sector are used.

        Returns:
            Tensor: Barycenter coordinates with shape ``(NE, GD)`` or the
            indexed subset shape, where ``GD`` is the geometric dimension.
        """
        return self.schema.barycenter(self.context(), index)

    def bc_to_point(self, bc: Tensor | tuple[Tensor, ...], *, index: Index | None = None) -> Tensor:
        """Map barycentric coordinates to physical coordinates.

        Parameters:
            bc (Tensor | tuple[Tensor, ...]): Barycentric coordinates.  Simplex
                entities use a single tensor, while tensor-product entities may
                use one tensor per factor.
            index (Index, optional): Entity subset on which the mapping is
                evaluated.

        Returns:
            Tensor: Physical points.  For common simplex entities the shape is
            ``(NE, NQ, GD)`` after selecting entities and quadrature points.
        """
        if not isinstance(bc, tuple):
            bc = (bc,)
        return self.schema.bc_to_point(self.context(), bc, index)

    def boundary(self) -> "BoundaryInfo":
        """Infer boundary information for this entity sector.

        Returns:
            BoundaryInfo:
            - mask: Boolean tensor indicating which entities are on the boundary.
            - index: Integer tensor of boundary entity indices.
        """
        return self.schema.boundary(self.context())

    def del_attribute(self, name: str) -> None:
        """Delete a user attribute from the entity sector.

        Parameters:
            name (str): Attribute name.

        Raises:
            KeyError: If ``name`` is not present in ``sector.attributes``.
        """
        if name in self.sector.attributes:
            del self.sector.attributes[name]
        else:
            raise KeyError(f"Attribute '{name}' not found in sector attributes.")

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
        """Compute an integral error norm between two functions on the entity.

        Functions not marked as barycentric are wrapped with
        :meth:`barycentric` before integration.  The computed value is
        ``(integral(abs(f1 - f2)**power))**(1/power)``.

        Parameters:
            f1 (Callable[..., Tensor]): First function.
            f2 (Callable[..., Tensor]): Second function.
            power (float, optional): Norm power.  Default is 2.0.
            q (int, optional): Quadrature order.  Default is 3.
            cell_axis (bool, optional): If ``True``, return one error value per
                selected entity.  If ``False``, return the global error over
                all selected entities.
            index (Index, optional): Entity subset.

        Returns:
            Tensor: Scalar global error, or a tensor of entity-wise errors when
            ``cell_axis`` is ``True``.
        """
        from ...decorator import barycentric
        if not getattr(f1, "coordtype", None) == "barycentric":
            f1 = self.barycentric(f1, index=index)
        if not getattr(f2, "coordtype", None) == "barycentric":
            f2 = self.barycentric(f2, index=index)
        @barycentric
        def integrand(bcs: Tensor | tuple[Tensor, ...]) -> Tensor:
            v1 = f1(bcs)
            v2 = f2(bcs)
            return bm.abs(v1 - v2) ** power

        if cell_axis:
            return self.integral(integrand, q=q, index=index) ** (1.0 / power)
        return bm.sum(self.integral(integrand, q=q, index=index)) ** (1.0 / power)

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the embedding space."""
        return self.schema.geo_dimension(self.context())

    def get_attribute(self, name: str) -> Any:
        """Return a user attribute stored on the entity sector.

        Parameters:
            name (str): Attribute name.

        Returns:
            Any: Stored attribute value, or ``None`` if the attribute does not
            exist.
        """
        return self.sector.attributes.get(name)

    def global_permutations(self, name_or_topdim: str | int, idx: int = 0, indexing: Literal["o", "s"] = "o") -> Tensor:
        """Return local-to-global orientation permutations for sub-entities.

        Parameters:
            name_or_topdim (str | int): Target sub-entity schema name, such as
                ``"point"``, ``"segment"``, or ``"tri"``, or a topological
                dimension.  A negative dimension is interpreted as a dimension
                relative to the current entity type.
            idx (int, optional): Target sector index when ``name_or_topdim``
                selects a dimension or entity type that has multiple sectors.  Default is 0.

        Returns:
            Tensor: Integer tensor whose leading axes enumerate source entities
            and their local target entities.  The last axis stores the vertex
            permutation induced by global orientation.
        """
        tgt = _Reg.schema_name_single_parser(
            name_or_topdim, idx, self.schema.top_dim, self.schema.OFace.keys()
        )
        return self.schema.global_permutations(self.context(), tgt, indexing=indexing)

    def grad_lambda(
        self,
        bcs: tuple[Tensor, ...] | None = None,
        index: Index | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        """Return gradients of barycentric coordinate functions.

        Parameters:
            bcs (tuple[Tensor, ...] | None, optional): Evaluation points in
                barycentric coordinates.  If provided, gradients may be
                broadcast along the quadrature-point axis.
            index (Index, optional): Entity subset.
            ref (bool, optional): If ``True``, return gradients on the reference
                entity.  If ``False``, return gradients in physical coordinates.
                Default is ``False``.

        Returns:
            Tensor: Gradients of barycentric coordinates.  Without ``bcs``, a
            typical shape is ``(NE, NV, GD)`` for physical gradients or
            ``(NE, NV, NR)`` for reference gradients.
        """
        return self.schema.grad_lambda(self.context(), index, bcs=bcs, ref=ref) # type: ignore

    def grad_shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...] = 1,
        *,
        index: Index | None = None,
        variables: Literal["b", "u", "x"] = "u",
        mi = None
    ) -> Tensor:
        """Evaluate gradients of local shape functions.

        Parameters:
            bcs (Tensor | tuple[Tensor, ...]): Barycentric evaluation points.
            p (int | tuple[int, ...], optional): Polynomial degree, or one
                degree per tensor-product factor.  Default is 1.
            index (Index, optional): Entity subset, required only for Cartesian
                gradients.
            variables ({"b", "u", "x"}, optional): Coordinate system of the
                derivative: barycentric coordinates, reference coordinates, or
                physical Cartesian coordinates.  Default is ``"u"``.
            mi: Optional precomputed multi-index matrix.  Accepted for API
                compatibility with schema implementations.

        Returns:
            Tensor: Shape-function gradients.  The trailing axis stores the
            derivative components in the coordinate system selected by
            ``variables``.

        Raises:
            ValueError: If ``variables`` is not one of ``"b"``, ``"u"``, or
                ``"x"``.
        """
        if isinstance(bcs, Tensor):
            bcs = (bcs,)
        if isinstance(p, int):
            p = (p,)
        if variables == "b":
            return self.schema.grad_shape_function_barycentric(bcs=bcs, p=p)
        elif variables == "u":
            return self.schema.grad_shape_function_reference(bcs=bcs, p=p)
        elif variables == "x":
            return self.schema.grad_shape_function_cartesian(
                self.context(), bcs=bcs, p=p, index=index
            )
        else:
            raise ValueError(f"Unsupported variable type: {variables}")

    @property
    def indices(self) -> Tensor:
        """Connectivity array of this entity sector."""
        return getattr(self.sector, "indices")

    def integral(
        self,
        func: Callable[..., Tensor],
        /,
        q: int = 3,
        *,
        index: Index | None = None
    ) -> Tensor:
        """Integrate a barycentric function over selected entities.

        Parameters:
            func (Callable[..., Tensor]): Function evaluated at barycentric
                quadrature points.
            q (int, optional): Quadrature order.  Default is 3.
            index (Index, optional): Entity subset.

        Returns:
            Tensor: Integral values.  For scalar integrands this is typically
            one value per selected entity before any caller-side reduction.
        """
        return self.schema.integral(self.context(), func, q, index)

    def jacobi_matrix(self, bcs: Tensor | tuple[Tensor, ...], *, index: Index | None = None) -> Tensor:
        """Return Jacobian matrices of the reference-to-physical map.

        Parameters:
            bcs (Tensor | tuple[Tensor, ...]): Barycentric evaluation points.
            index (Index, optional): Entity subset.

        Returns:
            Tensor: Jacobian tensor, commonly with shape ``(NE, NQ, GD, ref_dim)``,
            where ``ref_dim`` is the reference dimension.
        """
        if isinstance(bcs, Tensor):
            bcs = (bcs,)
        return self.schema.jacobi_matrix(self.context(), bcs, index)

    def measure(self, *, index: Index | None = None) -> Tensor:
        """Return measures of selected entities.

        Parameters:
            index (Index, optional): Entity subset.

        Returns:
            Tensor: One measure per selected entity.  The measure is length for
            edges, area for surface entities, and volume for volume entities.
        """
        return self.schema.measure(self.context(), index)

    def multi_index_matrix(self, order: int | tuple[int, ...], *, internal: bool = False, tensorprod: bool = True):
        """Return interpolation multi-indices for this entity type.

        Parameters:
            order (int | tuple[int, ...]): Polynomial degree, or tensor-product
                degrees.
            internal (bool, optional): If ``True``, return only interior
                interpolation-point multi-indices.  Default is ``False``.
            tensorprod (bool, optional): If ``True``, use the tensor-product
                ordering expected by interpolation utilities.  Default is
                ``True``.

        Returns:
            Tensor: Integer tensor with one row per local interpolation point
            and one column per local vertex or tensor-product coordinate.
        """
        if isinstance(order, int):
            order = (order,)
        return self.schema.multi_index(order, internal=internal, tensorprod=tensorprod)

    def normal(self, *, index: Index | None = None) -> Tensor:
        """Return normal vectors associated with selected entities.

        Parameters:
            index (Index, optional): Entity subset.

        Returns:
            Tensor: Normal vectors.  Schemas commonly return shape
            ``(NE, NN, GD)``, where ``NN`` is the number of normal directions.
        """
        return self.schema.normal(self.context(), index)

    def num_multi_index(self, order: int | tuple[int, ...], *, internal: bool = False) -> int:
        """Return the number of local interpolation multi-indices.

        Parameters:
            order (int | tuple[int, ...]): Polynomial degree, or tensor-product
                degrees.
            internal (bool, optional): If ``True``, count only interior
                interpolation points.  Default is ``False``.

        Returns:
            int: Number of local multi-indices for the requested order.
        """
        if isinstance(order, int):
            order = (order,)
        return self.schema.num_multi_index(order, internal=internal)

    def quadrature_formula(self, q: int = 3, qtype: str = "legendre"):
        """Return a quadrature formula on the reference entity.

        Parameters:
            q (int, optional): Quadrature order.  Default is 3.
            qtype (str, optional): Quadrature family.  Default is
                ``"legendre"``.

        Returns:
            Quadrature: Quadrature object supplied by the entity schema.
        """
        return self.schema.quadrature_formula(q, qtype)

    def set_attribute(self, name: str, value: Any) -> None:
        """Set a user attribute on the entity sector.

        Parameters:
            name (str): Attribute name.
            value (Any): Attribute value.
        """
        self.sector.attributes[name] = value

    def shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...] = 1,
        *,
        index: Index | None = None,
        variables: str = "u",
        mi = None
    ) -> Tensor:
        """Evaluate local scalar shape functions.

        Parameters:
            bcs (Tensor | tuple[Tensor, ...]): Barycentric evaluation points.
            p (int | tuple[int, ...], optional): Polynomial degree, or one
                degree per tensor-product factor.  Default is 1.
            index (Index, optional): Entity subset.  Accepted for API symmetry;
                the current value computation is reference-entity based.
            variables (str, optional): Output convention.  ``"u"`` returns
                reference values; ``"x"`` adds a leading broadcast axis for
                physical-coordinate FEALPy compatibility.  Default is ``"u"``.
            mi: Optional precomputed multi-index matrix.  Accepted for API
                compatibility.

        Returns:
            Tensor: Shape-function values.  The trailing dimension enumerates
            local basis functions.

        Raises:
            ValueError: If ``variables`` is unsupported.
        """
        if isinstance(bcs, Tensor):
            bcs = (bcs,)
        if isinstance(p, int):
            p = (p,)
        val = self.schema.shape_function(bcs, p)
        if variables == "u":
            return val
        elif variables == "x":
            return val[None, ...] # type: ignore[return-value]
        else:
            raise ValueError(f"Unsupported variable type: {variables}")

    def size(self) -> int:
        """Return the number of entities in this sector."""
        return self.schema.size(self.context())

    def tangent(self, *, index: Index | None = None) -> Tensor:
        """Return tangent vectors associated with selected entities.

        Parameters:
            index (Index, optional): Entity subset.

        Returns:
            Tensor: Tangent vectors, commonly with shape ``(NE, TD, GD)``.
        """
        return self.schema.tangent(self.context(), index)

    def to(self, target: int | str | EntityView, idx: int = 0, /) -> Relation:
        """Return the relation from this entity sector to a target sector.

        Parameters:
            target (int | str | EntityView): Target entity selector.  It may be
                a topological dimension, an entity type/name, or another
                ``EntityView``.
            idx (int, optional): Target sector index when ``target`` selects a
                dimension or entity type that has multiple sectors.  Default is
                0.

        Returns:
            Relation: Relation object containing source and target indices and
            conversion helpers such as array or COO representations.
        """
        if isinstance(target, EntityView):
            tgt = target.schema.name
        else:
            mesh_top_dim = max(self.block.sectors[name].schema.top_dim for name in self.block.sectors.keys())
            tgt = _Reg.schema_name_single_parser(target, idx, mesh_top_dim, self.block.sectors.keys())
        return self.schema.relation(self.context(), tgt)

    def to_ipoint(self, order: int, index: Index | None = None) -> Tensor:
        """Map entities to global interpolation-point indices.

        Parameters:
            order (int): Interpolation order.
            index (Index, optional): Entity subset.

        Returns:
            Tensor: Integer tensor of shape ``(NE, NIP)`` or the indexed subset,
            where ``NIP`` is the number of local interpolation points on this
            entity type.
        """
        from ..ipoints import to_ipoint
        from .mesh_view import MeshView
        mapping = to_ipoint(MeshView(self.block), self.schema.name, order)
        return mapping if index is None else mapping[index]

    def top_dimension(self) -> int:
        """Return the topological dimension of this entity type."""
        return self.schema.top_dim
