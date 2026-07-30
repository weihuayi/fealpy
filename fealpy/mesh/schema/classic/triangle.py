
from ....backend import bm
from ....backend import Index, Tensor
from ...ipoints import MultiIndex as _MI, multi_index_tensorprod
from .base import (
    EntityContext,
    ShapedEntitySchema,
    _require_bcs_tuple,
    _require_order_tuple,
)

__all__ = ["TriangleSchema"]


class TriangleSchema(ShapedEntitySchema):
    name = "tri"
    top_dim = 2
    OFace = {
        "segment": [[1, 2], [2, 0], [0, 1]],
        "point": [[0], [1], [2]],
    }
    SFace = {
        "segment": [[1, 2], [0, 2], [0, 1]],
        "point": [[0], [1], [2]],
    }
    orientation = [
        (0, 1, 2), (1, 2, 0), (2, 0, 1),
        (0, 2, 1), (2, 1, 0), (1, 0, 2),
    ]
    ref_measure = 0.5

    @classmethod
    def _selected_triangles(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tri = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tri.shape) == 1:
            tri = bm.reshape(tri, (1, -1))
        return tri

    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        p = _require_order_tuple(order, "triangle multi_index", 1)[0]
        if internal:
            mi = _MI.multi_index_inner(p, 3)
        else:
            mi = _MI.multi_index_matrix(p, 3)
        if tensorprod:
            return multi_index_tensorprod(mi)
        return mi

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tri = cls._selected_triangles(ctx, index)
        points = ctx.block.positions[tri]
        return bm.mean(points, axis=1)

    @classmethod
    def shape_function(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "triangle shape_function", 1)
        p = _require_order_tuple(p, "triangle shape_function", 1)
        if bcs[0].shape[-1] != 3:
            raise ValueError(f"triangle shape_function expects last dimension 3, got {bcs[0].shape[-1]}")
        mi = cls.multi_index(p)
        return bm.simplex_shape_function(bcs[0], p[0], mi)

    @classmethod
    def grad_shape_function_barycentric(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "triangle grad_shape_function_barycentric", 1)
        p = _require_order_tuple(p, "triangle grad_shape_function_barycentric", 1)
        if bcs[0].shape[-1] != 3:
            raise ValueError(
                "triangle grad_shape_function_barycentric expects "
                f"last dimension 3, got {bcs[0].shape[-1]}"
            )

        mi = cls.multi_index(p)
        return bm.simplex_grad_shape_function(bcs[0], p[0], mi)

    @classmethod
    def grad_shape_function_reference(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "triangle grad_shape_function_reference", 1)
        p = _require_order_tuple(p, "triangle grad_shape_function_reference", 1)

        Dlambda = bm.array(
            [
                [-1.0, -1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=bcs[0].dtype,
            device=bm.get_device(bcs[0]),
        )
        grad_bary = cls.grad_shape_function_barycentric(bcs, p)
        return bm.einsum("...ij,jk->...ik", grad_bary, Dlambda)

    @classmethod
    def jacobi_matrix(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        index: Index | None,
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "triangle jacobi_matrix", 1)
        tri = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tri.shape) == 1:
            tri = bm.reshape(tri, (1, -1))

        gphi = cls.grad_shape_function_reference(bcs, p=(1,))
        return bm.einsum("cim,qin->cqmn", ctx.block.positions[tri], gphi)

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "triangle bc_to_point", 1)
        bc = bcs[0]
        tri = cls._selected_triangles(ctx, index)
        points = ctx.block.positions[tri]
        return bm.einsum("...j,cjd->c...d", bc, points)

    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None,
        bcs: tuple[Tensor, ...] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        tri = cls._selected_triangles(ctx, index)
        node = ctx.block.positions
        if ref:
            grad = bm.broadcast_to(
                bm.eye(3, dtype=node.dtype)[None, :, :],
                (tri.shape[0], 3, 3),
            )
        else:
            gd = int(node.shape[1])
            if gd == 2:
                grad = bm.triangle_grad_lambda_2d(tri, node)
            elif gd == 3:
                grad = bm.triangle_grad_lambda_3d(tri, node)
            else:
                raise ValueError(f"unsupported geometric dimension: {gd}")
        if bcs is None:
            return grad
        bcs = _require_bcs_tuple(bcs, "triangle grad_lambda", 1)
        nq = int(bcs[0].shape[0])
        return bm.broadcast_to(grad[:, None, :, :], (tri.shape[0], nq, grad.shape[1], grad.shape[2]))

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None):
        if qtype != "legendre":
            raise ValueError(f"unsupported quadrature type: {qtype}")
        if q > 9:
            from ....quadrature.stroud_quadrature import StroudQuadrature
            return StroudQuadrature(2, q)
        from ....quadrature import TriangleQuadrature
        return TriangleQuadrature(q, device=device)

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tri = cls._selected_triangles(ctx, index)
        node = ctx.block.positions
        gd = int(node.shape[1])
        if gd == 2:
            return bm.simplex_measure(tri, node)
        if gd == 3:
            points = node[tri]
            v1 = points[:, 1, :] - points[:, 0, :]
            v2 = points[:, 2, :] - points[:, 0, :]
            normal = bm.cross(v1, v2)
            return bm.linalg.vector_norm(normal, axis=1) * 0.5
        raise ValueError(f"unsupported geometric dimension: {gd}")

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tri = cls._selected_triangles(ctx, index)
        points = ctx.block.positions[tri]
        gd = int(points.shape[2])
        if gd == 2:
            return bm.zeros((points.shape[0], 0, gd), **bm.context(points))
        if gd == 3:
            v1 = points[:, 1, :] - points[:, 0, :]
            v2 = points[:, 2, :] - points[:, 0, :]
            normal = bm.cross(v1, v2)
            return bm.expand_dims(normal, axis=1)
        raise ValueError(f"unsupported geometric dimension: {gd}")

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tri = cls._selected_triangles(ctx, index)
        points = ctx.block.positions[tri]
        t0 = points[:, 1, :] - points[:, 0, :]
        t1 = points[:, 2, :] - points[:, 0, :]
        return bm.stack([t0, t1], axis=1)
