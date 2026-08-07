
from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Concatenate, Literal, ParamSpec, TYPE_CHECKING

from ...backend import Tensor, Index
from ..storage import EntityContext, Relation
from ..topology.boundary import BoundaryInfo

if TYPE_CHECKING:
    from ...quadrature import Quadrature

__all__ = ["EntitySchema"]

P = ParamSpec("P")


class EntitySchema:
    name: ClassVar[str]
    top_dim: ClassVar[int]
    OFace: ClassVar[dict[str, list[list[int]]]] = {}
    SFace: ClassVar[dict[str, list[list[int]]]] = {}
    orientation: ClassVar[list[tuple[int, ...]]] = []
    ccw: ClassVar[list[int] | None] = None
    ref_measure: ClassVar[float | None] = None

    ### [Entity Topology] ###

    @classmethod
    def boundary(cls, ctx: EntityContext) -> BoundaryInfo:
        """Boundary information of the entity.

        Returns:
            NamedTuple:
            - index: Tensor of shape (num_boundary,) containing the indices
                of boundary entities.
            - mask: Tensor of shape (num_entity,) containing a boolean mask
                indicating whether each entity is a boundary entity.
        """
        raise NotImplementedError()

    @classmethod
    def local_entity(cls, tgt_name: str, /, indexing: Literal["o", "s"] = "o") -> list[list[int]]:
        """Local entity indices of the target entity.

        Parameters:
            tgt_name (str): The name of the target entity.
            indexing (Literal["o", "s"], optional): The indexing method. Defaults to "o".

        Returns:
            list[list[int]]: The local entity indices.
        """
        raise NotImplementedError()

    @classmethod
    def relation(cls, ctx: EntityContext, tgt_name: str) -> Relation:
        """Compute the relation between two entities."""
        raise NotImplementedError()

    @classmethod
    def size(cls, ctx: EntityContext) -> int:
        """Number of entities in the sector."""
        raise NotImplementedError()

    ### [Multi-Indices] ###

    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        """Multi-index of the entity, with one column per vertex."""
        raise NotImplementedError()

    @classmethod
    def num_multi_index(cls, order: tuple[int, ...], *, internal: bool = False) -> int:
        """Number of multi-indices."""
        return int(cls.multi_index(order, internal=internal).shape[0])

    @classmethod
    def global_permutations(cls, ctx: EntityContext, tgt_name: str, indexing: Literal["o", "s"] = "o") -> Tensor:
        """Permutation indices from local to global."""
        raise NotImplementedError()

    @classmethod
    def vo_to_do(cls, order: tuple[int, ...]) -> dict[tuple[int, ...], Tensor]:
        """Return the mapping from vertex orientation to DoF ordering."""
        raise NotImplementedError()

    ### [Geometric Computations] ###

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute the barycenter of the entity."""
        raise NotImplementedError()

    @classmethod
    def barycentric[**P, R](
        cls,
        ctx: EntityContext,
        func: Callable[Concatenate[Tensor, P], R],
        index: Index | None
    ) -> Callable[Concatenate[Tensor | tuple[Tensor, ...], P], R]:
        """Transform functions from cartesian to barycentric coordinates."""
        raise NotImplementedError()

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        """Convert barycentric coordinates to physical points."""
        raise NotImplementedError()

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        """Geometric dimension of the cell."""
        raise NotImplementedError()

    @classmethod
    def grad_shape_function_barycentric(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        """Gradient of shape functions to barycentric coordinates.

        Parameters:
            bcs (tuple[Tensor, ...]): Barycentric coordinates of evaluation points, with shape (NQ, num_bc).
            p (tuple[int, ...]): Polynomial degree(s) of the shape functions.

        Returns:
            Tensor: The gradient of shape functions with shape (NQ, num_shape, num_bc), where
                NQ is the number of points, num_shape is the number of shape functions,
                and num_bc is the number of barycentric coordinates.
        """
        raise NotImplementedError()

    @classmethod
    def grad_shape_function_cartesian(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...],
        *,
        index: Index | None = None
    ) -> Tensor:
        """Gradient of shape functions to cartesian coordinates."""
        from ..transform import piola_transform_covariant
        grad_ref = cls.grad_shape_function_reference(bcs, p)[None, ...]
        J = cls.jacobi_matrix(ctx, bcs, index=index)[..., None, :, :]
        # [NC, NQ, num_shape, ref_dim], [NC, NQ, num_shape, GD, ref_dim]
        return piola_transform_covariant(grad_ref, J)

    @classmethod
    def grad_shape_function_reference(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        """Gradient of shape functions to reference coordinates.

        Parameters:
            bcs (tuple[Tensor, ...]): Barycentric coordinates of evaluation points, with shape (NQ, num_bc).
            p (tuple[int, ...]): Polynomial degree(s) of the shape functions.

        Returns:
            Tensor: The gradient of shape functions with shape (NQ, num_shape, ref_dim), where
                NQ is the number of points, num_shape is the number of shape functions,
                and ref_dim is the dimension of the reference element.
        """
        raise NotImplementedError()

    @classmethod
    def integral(
        cls,
        ctx: EntityContext,
        func: Callable[[Tensor], Tensor] | Callable[[tuple[Tensor, ...]], Tensor],
        q: int, index: Index | None
    ) -> Tensor:
        """Integral of a barycentric function."""
        raise NotImplementedError()

    @classmethod
    def jacobi_matrix(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        """Jacobi matrix of the transformation from reference to physical element.

        Parameters:
            ctx (EntityContext): The entity context containing the mesh block and sector information.
            bcs (tuple[Tensor, ...]): Barycentric coordinates of evaluation points, with shape (NQ, num_bc).
            index (Index | None): The index of the entity in the sector, or None for all entities.

        Returns:
            Tensor: The Jacobi matrix with shape (NC, NQ, GD, ref_dim), where
                NC is the number of cells, NQ is the number of points,
                GD is the geometric dimension, and ref_dim is the number of
                reference coordinates (typically equal to the topological
                dimension of the entity).
        """
        raise NotImplementedError()

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute the measure of the entity."""
        raise NotImplementedError()

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute the normal vector of the entity."""
        raise NotImplementedError()

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device = None) -> "Quadrature":
        """Quadrature formula for the entity."""
        raise NotImplementedError()

    @classmethod
    def shape_function(cls, bcs: tuple[Tensor, ...], p: tuple[int, ...]) -> Tensor:
        """Shape functions."""
        raise NotImplementedError()

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute the tangent vector of the entity."""
        raise NotImplementedError()
