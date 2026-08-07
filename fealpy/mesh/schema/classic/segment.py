
from ....backend import bm
from ....backend import Index, Tensor
from ...ipoints import MultiIndex as _MI, multi_index_tensorprod
from .base import (
    EntityContext,
    ShapedEntitySchema,
    _require_bcs_tuple,
    _require_order_tuple,
)

__all__ = ["SegmentSchema"]


class SegmentSchema(ShapedEntitySchema):
    """Schema implementation for 1D simplex entities embedded in physical space."""

    name = "segment"
    top_dim = 1
    OFace = {
        "point": [[0], [1]]
    }
    SFace = {
        "point": [[0], [1]]
    }
    orientation = [(0, 1), (1, 0)]

    @classmethod
    def _entity(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Return the selected segment connectivity with an explicit entity axis.

        Parameters:
            ctx (EntityContext): The mesh block and segment sector.
            index (Index | None): Selected segment entities, or ``None`` for all segments.

        Returns:
            Tensor: The segment connectivity tensor with shape ``(NC, 2)``.
        """
        segment = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(segment.shape) == 1:
            segment = bm.reshape(segment, (1, -1))
        return segment

    @classmethod
    def _points(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Return the endpoint coordinates of the selected segments.

        Parameters:
            ctx (EntityContext): The mesh block and segment sector.
            index (Index | None): Selected segment entities, or ``None`` for all segments.

        Returns:
            Tensor: Endpoint coordinates with shape ``(NC, 2, GD)``.
        """
        return ctx.block.positions[cls._entity(ctx, index)]

    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        """Return interpolation multi-indices on the reference segment.

        Parameters:
            order (tuple[int, ...]): Polynomial degree on the segment.
            internal (bool, optional): If ``True``, return only interior multi-indices.
            tensorprod (bool, optional): If ``True``, convert the simplex ordering to the
                tensor-product-compatible ordering used by interpolation-point utilities.

        Returns:
            Tensor: The multi-index tensor with one column per segment endpoint.
        """
        p = _require_order_tuple(order, "segment multi_index", 1)[0]
        if internal:
            mi = _MI.multi_index_inner(p, 2)
        else:
            mi = _MI.multi_index_matrix(p, 2)
        if tensorprod:
            return multi_index_tensorprod(mi)
        return mi

    @classmethod
    def num_multi_index(cls, order: tuple[int, ...], *, internal: bool = False) -> int:
        """Return the number of interpolation multi-indices on the segment.

        Parameters:
            order (tuple[int, ...]): Polynomial degree on the segment.
            internal (bool, optional): If ``True``, count only interior multi-indices.

        Returns:
            int: The number of local interpolation multi-indices.
        """
        p = _require_order_tuple(order, "segment num_multi_index", 1)[0]
        if internal:
            return p - 1 if p > 1 else 0
        return p + 1

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute segment barycenters as the average of the two endpoints.

        Parameters:
            ctx (EntityContext): The mesh block and segment sector.
            index (Index | None): Selected segment entities, or ``None`` for all segments.

        Returns:
            Tensor: Segment barycenters with shape ``(NC, GD)``.
        """
        points = cls._points(ctx, index)
        return bm.mean(points, axis=1)

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        """Map segment barycentric coordinates to physical points.

        Parameters:
            ctx (EntityContext): The mesh block and segment sector.
            bcs (tuple[Tensor, ...]): Segment barycentric coordinates with one tensor of
                shape ``(NQ, 2)``.
            index (Index | None): Selected segment entities, or ``None`` for all segments.

        Returns:
            Tensor: Physical points with shape ``(NC, NQ, GD)``.
        """
        bcs = _require_bcs_tuple(bcs, "segment bc_to_point", 1)
        if bcs[0].shape[-1] != 2:
            raise ValueError(f"segment barycentric coordinates expect last dimension 2, got {bcs[0].shape[-1]}")

        points = cls._points(ctx, index)
        return bm.einsum("...j,cjd->c...d", bcs[0], points)

    @classmethod
    def shape_function(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        """Evaluate scalar Lagrange shape functions on the reference segment.

        Parameters:
            bcs (tuple[Tensor, ...]): Segment barycentric coordinates with one tensor of
                shape ``(NQ, 2)``.
            p (tuple[int, ...]): Polynomial degree on the segment.

        Returns:
            Tensor: Shape-function values with shape ``(NQ, num_shape)``.
        """
        bcs = _require_bcs_tuple(bcs, "segment shape_function", 1)
        p = _require_order_tuple(p, "segment shape_function", 1)
        if bcs[0].shape[-1] != 2:
            raise ValueError(f"segment shape_function expects last dimension 2, got {bcs[0].shape[-1]}")
        return bm.simplex_shape_function(bcs[0], p[0])

    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None,
        bcs: tuple[Tensor, ...] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        """Return gradients of the segment barycentric coordinates.

        Parameters:
            ctx (EntityContext): The mesh block and segment sector.
            index (Index | None): Selected segment entities, or ``None`` for all segments.
            bcs (tuple[Tensor, ...] | None, optional): Evaluation points in barycentric
                form. When provided, the result is broadcast to shape
                ``(NC, NQ, 2, GD_or_2)``.
            ref (bool, optional): If ``True``, return gradients with respect to the two
                barycentric coordinates on the reference segment. If ``False``, return
                physical gradients with respect to cartesian coordinates.

        Returns:
            Tensor: Gradients of ``lambda_0`` and ``lambda_1``. Without ``bcs``, the
            shape is ``(NC, 2, GD_or_2)``. With ``bcs``, the gradients are broadcast
            along the quadrature axis.
        """
        points = cls._points(ctx, index)
        nc = int(points.shape[0])
        if ref:
            grad = bm.broadcast_to(
                bm.eye(2, dtype=ctx.block.positions.dtype)[None, :, :],
                (nc, 2, 2),
            )
        else:
            tangent = points[:, 1, :] - points[:, 0, :]
            sqnorm = bm.sum(tangent * tangent, axis=1, keepdims=True)
            g1 = tangent / sqnorm
            g0 = -g1
            grad = bm.stack([g0, g1], axis=1)
        if bcs is None:
            return grad
        bcs = _require_bcs_tuple(bcs, "segment grad_lambda", 1)
        nq = int(bcs[0].shape[0])
        return bm.broadcast_to(grad[:, None, :, :], (nc, nq, grad.shape[1], grad.shape[2]))

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None):
        """Return a 1D Gauss-Legendre quadrature rule on the reference segment.

        Parameters:
            q (int): Quadrature order.
            qtype (str | None, optional): Quadrature family. Only ``"legendre"`` is
                supported for segments.
            device (optional): Backend device on which the quadrature data is created.

        Returns:
            Quadrature: The quadrature object for the reference segment.
        """
        if qtype not in (None, "legendre"):
            raise ValueError(f"unsupported segment quadrature type: {qtype!r}")
        from fealpy.quadrature import GaussLegendreQuadrature
        return GaussLegendreQuadrature(q, device=device)

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Return the physical lengths of the selected segments.

        Parameters:
            ctx (EntityContext): The mesh block and segment sector.
            index (Index | None): Selected segment entities, or ``None`` for all segments.

        Returns:
            Tensor: Segment lengths with shape ``(NC,)``.
        """
        points = cls._points(ctx, index)
        tangent = points[:, 1, :] - points[:, 0, :]
        return bm.linalg.norm(tangent, axis=1)

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Return normal directions orthogonal to each segment tangent.

        In 2D, one normal direction is returned for each segment. In 3D, two
        mutually orthogonal normal directions are constructed.

        Parameters:
            ctx (EntityContext): The mesh block and segment sector.
            index (Index | None): Selected segment entities, or ``None`` for all segments.

        Returns:
            Tensor: Normal directions with shape ``(NC, GD - 1, GD)`` for ``GD <= 3``.
        """
        points = cls._points(ctx, index)
        tangent = points[:, 1, :] - points[:, 0, :]
        gd = points.shape[-1]
        sqnorm = bm.sum(tangent * tangent, axis=1)

        if bm.any(sqnorm == 0):
            raise ValueError("degenerate segment has no well-defined normal directions")

        if gd == 1:
            return bm.zeros((points.shape[0], 0, gd), dtype=ctx.block.positions.dtype)

        if gd == 2:
            normal = bm.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
            return normal[:, None, :]

        if gd == 3:
            axis = bm.argmin(bm.abs(tangent), axis=1)
            ref_basis = bm.eye(gd, dtype=ctx.block.positions.dtype, device=bm.get_device(tangent))
            ref = ref_basis[axis]
            n1 = bm.cross(tangent, ref, axis=-1)
            n2 = bm.cross(tangent, n1, axis=-1)
            return bm.stack([n1, n2], axis=1)

        raise NotImplementedError(f"segment normal is only implemented for GD <= 3, got {gd}")

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Return the non-unit physical tangent vectors of the selected segments.

        Parameters:
            ctx (EntityContext): The mesh block and segment sector.
            index (Index | None): Selected segment entities, or ``None`` for all segments.

        Returns:
            Tensor: Tangent vectors with shape ``(NC, 1, GD)``.
        """
        points = cls._points(ctx, index)
        return (points[:, 1, :] - points[:, 0, :])[:, None, :]

    @classmethod
    def transform(cls, ctx: EntityContext, func, kind: str = "value"):
        """Wrap a cartesian function so it can be evaluated on segment barycentric points.

        Parameters:
            ctx (EntityContext): The mesh block and segment sector.
            func: A callable defined on physical points.
            kind (str, optional): Transformation kind. Only ``"value"`` is supported.

        Returns:
            Callable: A wrapper that accepts segment barycentric coordinates.
        """
        points = ctx.block.positions[ctx.sector.indices]

        def wrapper(bc: Tensor) -> Tensor:
            x = bm.einsum("...j,cjd->c...d", bc, points)
            value = func(x)
            if kind == "value":
                return value
            raise NotImplementedError(f"Unsupported segment transform kind: {kind!r}")

        return wrapper

    @classmethod
    def grad_shape_function_barycentric(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        """Evaluate shape-function gradients with respect to barycentric coordinates.

        Parameters:
            bcs (tuple[Tensor, ...]): Segment barycentric coordinates with one tensor of
                shape ``(NQ, 2)``.
            p (tuple[int, ...]): Polynomial degree on the segment.

        Returns:
            Tensor: A tensor with shape ``(NQ, num_shape, 2)`` whose last axis stores
                derivatives with respect to ``lambda_0`` and ``lambda_1``.
        """
        bcs = _require_bcs_tuple(bcs, "segment grad_shape_function_barycentric", 1)
        p = _require_order_tuple(p, "segment grad_shape_function_barycentric", 1)
        if bcs[0].shape[-1] != 2:
            raise ValueError(
                f"segment grad_shape_function_barycentric expects last dimension 2, got {bcs[0].shape[-1]}")

        mi = _MI.multi_index_matrix(p[0], 2)
        return bm.simplex_grad_shape_function(bcs[0], p[0], mi)
    
    @classmethod
    def grad_shape_function_reference(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        """Evaluate shape-function gradients with respect to the reference coordinate.

        The reference segment uses one coordinate ``u`` with barycentric relation
        ``(lambda_0, lambda_1) = (1 - u, u)``.

        Parameters:
            bcs (tuple[Tensor, ...]): Segment barycentric coordinates with one tensor of
                shape ``(NQ, 2)``.
            p (tuple[int, ...]): Polynomial degree on the segment.

        Returns:
            Tensor: A tensor with shape ``(NQ, num_shape, 1)``.
        """
        bcs = _require_bcs_tuple(bcs, "segment grad_shape_function_reference", 1)
        p = _require_order_tuple(p, "segment grad_shape_function_reference", 1)

        Dlambda = bm.array([-1, 1], dtype=bcs[0].dtype, device=bm.get_device(bcs[0]))
        grad_bary = cls.grad_shape_function_barycentric(bcs, p)
        return bm.einsum("...ij,j->...i", grad_bary, Dlambda)[..., None]

    @classmethod
    def jacobi_matrix(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        index: Index | None,
    ) -> Tensor:
        """Compute the Jacobian of the reference-to-physical segment map.

        Parameters:
            ctx (EntityContext): The mesh block and segment sector.
            bcs (tuple[Tensor, ...]): Barycentric evaluation points on the reference
                segment.
            index (Index | None): Selected segment entities, or ``None`` for all segments.

        Returns:
            Tensor: The Jacobian tensor with shape ``(NC, NQ, GD, 1)``. Its last axis
            stores the derivative of the physical map with respect to the reference
            coordinate ``u``.
        """
        bcs = _require_bcs_tuple(bcs, "segment jacobi_matrix", 1)
        points = cls._points(ctx, index)
        gphi = cls.grad_shape_function_reference(bcs, p=(1,))
        return bm.einsum("cid,qin->cqdn", points, gphi)
