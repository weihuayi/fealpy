
from typing import Any, Callable, Concatenate, Literal, overload

from ....backend import bm, Tensor, Index
from ...storage import EntityContext, Relation
from ...topology.boundary import BoundaryInfo
from ..entity_schema import EntitySchema


class ShapedEntitySchema(EntitySchema):
    """Entity schema with a fixed shape (e.g., a triangle or quadrilateral)."""
    @classmethod
    def boundary(cls, ctx: EntityContext) -> BoundaryInfo:
        """Boundary information of the entity."""
        from ...topology.boundary import BoundaryInferencer

        if ctx.block._cache_boundary_info is None:
            ctx.block._cache_boundary_info = {}

        return BoundaryInferencer.infer_entity(
            ctx.block,
            ctx.sector.schema_name,
            ctx.block._cache_boundary_info
        )

    @classmethod
    def local_entity(cls, tgt_name: str, /, indexing: Literal["o", "s"] = "o") -> list[list[int]]:
        if indexing == "o":
            if tgt_name in cls.OFace:
                return cls.OFace[tgt_name]
        elif indexing == "s":
            if tgt_name in cls.SFace:
                return cls.SFace[tgt_name]
        else:
            raise ValueError(f"indexing must be 'o' or 's', got {indexing!r}")
        raise ValueError(f"local entity {tgt_name!r} is not defined for {cls.name!r}")

    @classmethod
    def relation(cls, ctx: EntityContext, tgt_name: str) -> Relation:
        src_name = ctx.sector.schema_name
        relation = ctx.block.relations.get((src_name, tgt_name))

        from ..registry import SCHEMA_REGISTRY
        tgt_schema = SCHEMA_REGISTRY[tgt_name]

        if relation is None and ((tgt_name, src_name) in ctx.block.relations):
            relation = ctx.block.relations[(tgt_name, src_name)].inverse()
            ctx.block.relations[(src_name, tgt_name)] = relation
            return relation

        if relation is None and (cls.top_dim > tgt_schema.top_dim):
            from ...topology import TopRelationConnector
            try:
                TopRelationConnector.connect(ctx.block, src_name, tgt_name)
                relation = ctx.block.relations.get((src_name, tgt_name))
            except ValueError:
                pass

        if relation is None:
            from ...topology import TopRelationInferer
            try:
                TopRelationInferer.infer(ctx.block, src_name, tgt_name)
                relation = ctx.block.relations.get((src_name, tgt_name))
            except ValueError:
                pass

        if relation is None:
            raise ValueError(f"relation from {src_name!r} to {tgt_name!r} not found")

        return relation

    @classmethod
    def size(cls, ctx: EntityContext) -> int:
        """Number of entities in the sector."""
        return ctx.sector.indices.shape[0]

    @classmethod
    def global_permutations(cls, ctx: EntityContext, tgt_name: str, indexing: Literal["o", "s"] = "o") -> Tensor:
        from ..utils import argpermute
        cell_indices = ctx.sector.indices
        local_face = cls.local_entity(tgt_name, indexing=indexing)
        face_indices = ctx.block.get_sector(tgt_name).indices
        cell_to_face = cls.relation(ctx, tgt_name).tgt_indices
        return argpermute(
            cell_indices[:, local_face],
            face_indices[cell_to_face],
            dtype=bm.uint8
        )

    @classmethod
    def vo_to_do(cls, order: tuple[int, ...]) -> dict[tuple[int, ...], Tensor]:
        from ...ipoints import multi_index_sort
        result: dict[tuple[int, ...], Tensor] = {}

        for v_o in cls.orientation:
            mi = cls.multi_index(order, internal=True, tensorprod=False)
            d_o = multi_index_sort(mi[:, v_o])
            result[tuple(v_o)] = d_o

        return result

    ### [Geometric Computations] ###

    @classmethod
    def barycentric[**P, R](
        cls,
        ctx: EntityContext,
        func: Callable[Concatenate[Tensor, P], R],
        index: Index | None
    ) -> Callable[Concatenate[Tensor | tuple[Tensor, ...], P], R]:
        """Compute the barycentric coordinates of the entity."""
        from functools import wraps
        from ....decorator import barycentric
        @wraps(func)
        @barycentric
        def wrapper(bcs: Tensor | tuple[Tensor, ...], *args, **kwargs) -> R:
            if not isinstance(bcs, tuple):
                bcs = (bcs,)
            points = cls.bc_to_point(ctx, bcs, index) # [NC, NQ, GD]
            return func(points, *args, **kwargs)
        return wrapper

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        return int(ctx.block.positions.shape[1])

    @classmethod
    def integral(cls, ctx: EntityContext, func: Callable[[Tensor], Tensor], q: int, index: Index | None) -> Tensor:
        """Integral of a barycentric function."""
        quadrature = cls.quadrature_formula(q)
        bcs, ws = quadrature.get_quadrature_points_and_weights()
        if not isinstance(bcs, tuple):
            bcs = (bcs,)

        if not getattr(func, "coordtype", None) == "barycentric":
            func = cls.barycentric(ctx, func, index)
        values = func(bcs) # type: ignore
        J = cls.jacobi_matrix(ctx, bcs, index)
        det = bm.linalg.det # type: ignore

        if J.shape[-2] == J.shape[-1]:
            factor = bm.abs(det(J)) # [NC, NQ]
        else:
            # For non-square Jacobians, compute the square root of the determinant of J^T * J
            JTJ = bm.einsum("...ji, ...jk -> ...ik", J, J) # [NC, NQ, ref_dim, ref_dim]
            factor = bm.sqrt(det(JTJ)) # [NC, NQ]

        if cls.ref_measure is not None:
            factor = cls.ref_measure * factor

        return bm.einsum("cq, q, cq... -> c...", factor, ws, values)


@overload
def _require_bcs_tuple(bcs: Any, name: str) -> tuple[Tensor, ...]: ...
@overload
def _require_bcs_tuple(bcs: Any, name: str, n: Literal[1]) -> tuple[Tensor]: ...
@overload
def _require_bcs_tuple(bcs: Any, name: str, n: Literal[2]) -> tuple[Tensor, Tensor]: ...
@overload
def _require_bcs_tuple(bcs: Any, name: str, n: Literal[3]) -> tuple[Tensor, Tensor, Tensor]: ...
def _require_bcs_tuple(bcs: Any, name: str, n: int | None = None) -> tuple[Tensor, ...]:
    if not isinstance(bcs, tuple):
        raise TypeError(f"{name} expects barycentric coordinates as a tuple of tensors, got {type(bcs).__name__}")

    if n is None:
        return bcs

    if len(bcs) == n:
        pass
    elif len(bcs) == 1:
        bcs = (bcs[0],) * n
    else:
        raise ValueError(f"{name} expects {n} barycentric tensors, got {len(bcs)}")

    return bcs


@overload
def _require_order_tuple(p: Any, name: str) -> tuple[int, ...]: ...
@overload
def _require_order_tuple(p: Any, name: str, n: Literal[1]) -> tuple[int]: ...
@overload
def _require_order_tuple(p: Any, name: str, n: Literal[2]) -> tuple[int, int]: ...
@overload
def _require_order_tuple(p: Any, name: str, n: Literal[3]) -> tuple[int, int, int]: ...
def _require_order_tuple(p: Any, name: str, n: int | None = None) -> tuple[int, ...]:
    """Ensure that the polynomial degree is a tuple of integers.

    Parameters:
        p (tuple[int, ...]): The input polynomial degree(s).
        name (str): The name of the function for error messages.
        n (int | None): The expected number of polynomial degrees.
            If None, no check is performed. If an integer, the length of the
            tuple must be either 1 or n.
            1-length means that the same degree is used for all dimensions,
            returning a tuple of length n with the repeated degree.

    Returns:
        tuple[int, ...]: A tuple of polynomial degrees with length n (if n is not None).
    """
    if not isinstance(p, tuple):
        raise TypeError(f"{name} expects polynomial degrees as a tuple of integers, got {type(p).__name__}")

    if not all(isinstance(pi, int) for pi in p):
        raise TypeError(f"{name} expects polynomial degrees as integers, got {p}")

    if not all(pi >= 0 for pi in p):
        raise ValueError(f"{name} expects non-negative polynomial degrees, got {p}")

    if n is None:
        return p

    if len(p) == n:
        pass
    elif len(p) == 1:
        p = (p[0],) * n
    else:
        raise ValueError(f"{name} expects {n} polynomial degrees, got {len(p)}")

    return p
