
from ....backend import bm
from ....backend import Index, Tensor
from ...ipoints import MultiIndex as _MI, multi_index_tensorprod
from .base import (
    EntityContext,
    ShapedEntitySchema,
    _require_bcs_tuple,
    _require_order_tuple,
)

__all__ = ["PrismSchema"]


class PrismSchema(ShapedEntitySchema):
    name = "prism"
    top_dim = 3
    OFace = {
        "tri": [[0, 2, 1], [3, 4, 5]],
        "quad": [[0, 1, 4, 3], [1, 2, 5, 4], [0, 3, 5, 2]],
        "segment": [
            [0, 1], [1, 2], [0, 2],
            [0, 3], [1, 4], [2, 5],
            [3, 4], [4, 5], [3, 5],
        ],
        "point": [[0], [1], [2], [3], [4], [5]],
    }
    SFace = {
        "tri": [[0, 1, 2], [3, 4, 5]],
        "quad": [[0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5]],
        "segment": [
            [0, 1], [1, 2], [0, 2],
            [0, 3], [1, 4], [2, 5],
            [3, 4], [4, 5], [3, 5],
        ],
        "point": [[0], [1], [2], [3], [4], [5]],
    }
    ref_measure = 0.5

    @classmethod
    def _entity(cls, ctx: EntityContext, index: Index | None = None) -> Tensor:
        entity = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(entity.shape) == 1:
            entity = entity[None, :]
        return entity

    @classmethod
    def _points(cls, ctx: EntityContext, index: Index | None = None) -> Tensor:
        prism = cls._entity(ctx, index)
        return ctx.block.positions[prism]

    @classmethod
    def _tp_points(cls, ctx: EntityContext, index: Index | None = None) -> Tensor:
        prism = cls._entity(ctx, index)
        # ``shape_function`` orders the six nodes as triangle vertices on the
        # bottom layer followed by the corresponding vertices on the top
        # layer.  This is also the schema's contract order.
        return ctx.block.positions[prism]

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute barycenters of prism entities.

        Return shape: (NC, GD).
        """
        points = cls._points(ctx, index)
        return bm.mean(points, axis=1)

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute the volume of a prism.

        ∫_K dx = ∫_{\hat K} sqrt(det(G)) dξ, where G = J^T J.
        """
        qf = cls.quadrature_formula(2)
        bcs, ws = qf.get_quadrature_points_and_weights()
        G = cls.first_fundamental_form(ctx, bcs, index=index)
        l = bm.sqrt(bm.linalg.det(G))
        return 0.5 * bm.einsum("q,cq->c", ws, l)

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Prism volume entities have no normal directions in 3D."""
        prism = cls._entity(ctx, index)
        GD = cls.geo_dimension(ctx)
        return bm.zeros((prism.shape[0], 0, GD), dtype=ctx.block.positions.dtype)

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute tangent directions of prism entities.

        Return shape: (NC, 3, GD).
        """
        points = cls._points(ctx, index)
        return points[:, [1, 2, 3], :] - points[:, [0], :]

    # quadrature
    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None):
        from ....quadrature import (
            GaussLegendreQuadrature,
            TensorProductQuadrature,
            TriangleQuadrature,
        )

        qf0 = TriangleQuadrature(q, device=device)
        qf1 = GaussLegendreQuadrature(q, device=device)
        return TensorProductQuadrature((qf0, qf1))

    # shape function
    @classmethod
    def shape_function(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "prism shape_function", 2)
        p = _require_order_tuple(p, "prism shape_function", 2)

        mi0 = _MI.multi_index_matrix(p[0], 3)
        mi1 = _MI.multi_index_matrix(p[1], 2)
        # Keep quadrature points triangle-major, matching
        # TensorProductQuadrature.  Reorder the interleaved basis values to
        # the schema contract: bottom triangle vertices, then top vertices.
        phi = bm.tensorprod(
            bm.simplex_shape_function(bcs[0], p[0], mi0),
            bm.simplex_shape_function(bcs[1], p[1], mi1),
        )
        return phi[:, [0, 2, 4, 1, 3, 5]]

    @classmethod
    def grad_shape_function(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...],
        *,
        index: Index | None = None, variables: str = "u",
        mi: Tensor | None = None
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "prism grad_shape_function", 2)
        p = _require_order_tuple(p, "prism grad_shape_function", 2)
        Dlambda0 = bm.array([[-1, -1], [1, 0], [0, 1]], dtype=bcs[0].dtype)
        Dlambda1 = bm.array([[-1], [1]], dtype=bcs[1].dtype)

        phi0 = bm.simplex_shape_function(bcs[0], p[0], mi)
        phi1 = bm.simplex_shape_function(bcs[1], p[1], mi)
        R0 = bm.simplex_grad_shape_function(bcs[0], p[0], mi)
        R1 = bm.simplex_grad_shape_function(bcs[1], p[1], mi)

        gphi0 = bm.einsum("...ij,jn->...in", R0, Dlambda0)
        gphi1 = bm.einsum("...ij,jn->...in", R1, Dlambda1)
        ref = bm.concatenate([
            gphi0[:, None, :, None, :] * phi1[None, :, None, :, None],
            phi0[:, None, :, None, None] * gphi1[None, :, None, :, :],
        ], axis=-1).reshape(-1, phi0.shape[1] * phi1.shape[1], 3)
        ref = ref[:, [0, 2, 4, 1, 3, 5], :]

        if variables == "u":
            return ref
        if variables == "x":
            return cls.transform_grad(ctx, bcs, ref, index)
        raise ValueError(f"Unsupported variables: {variables!r}")

    @classmethod
    def grad_shape_function_barycentric(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "prism grad_shape_function_barycentric", 2)
        p = _require_order_tuple(p, "prism grad_shape_function_barycentric", 2)

        if bcs[0].shape[-1] != 3:
            raise ValueError(
                "prism grad_shape_function_barycentric expects "
                f"triangle barycentric tensor with last dimension 3, got {bcs[0].shape[-1]}"
            )
        if bcs[1].shape[-1] != 2:
            raise ValueError(
                "prism grad_shape_function_barycentric expects "
                f"interval barycentric tensor with last dimension 2, got {bcs[1].shape[-1]}"
            )

        mi0 = _MI.multi_index_matrix(p[0], 3)
        mi1 = _MI.multi_index_matrix(p[1], 2)

        phi0 = bm.simplex_shape_function(bcs[0], p[0], mi0)
        phi1 = bm.simplex_shape_function(bcs[1], p[1], mi1)

        R0 = bm.simplex_grad_shape_function(bcs[0], p[0], mi0)
        R1 = bm.simplex_grad_shape_function(bcs[1], p[1], mi1)

        num_shape = phi0.shape[-1] * phi1.shape[-1]

        g0 = R0[:, None, :, None, :] * phi1[None, :, None, :, None]
        g1 = phi0[:, None, :, None, None] * R1[None, :, None, :, :]

        gphi = bm.concat([g0, g1], axis=-1)
        gphi = bm.reshape(gphi, (-1, num_shape, 5))
        # The intermediate tensor-product expansion is interleaved by
        # triangle vertex and interval endpoint.  The schema contract keeps
        # all bottom vertices first, followed by all top vertices.
        return gphi[:, [0, 2, 4, 1, 3, 5], :]

    @classmethod
    def grad_shape_function_reference(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "prism grad_shape_function_reference", 2)
        p = _require_order_tuple(p, "prism grad_shape_function_reference", 2)

        grad_bary = cls.grad_shape_function_barycentric(bcs, p)
        return bm.stack(
            [
                -grad_bary[..., 0] + grad_bary[..., 1],
                -grad_bary[..., 0] + grad_bary[..., 2],
                -grad_bary[..., 3] + grad_bary[..., 4],
            ],
            axis=-1,
        )
    
    @classmethod
    def jacobi_matrix(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        index: Index | None
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "prism jacobi_matrix", 2)

        points = cls._tp_points(ctx, index)
        gphi = cls.grad_shape_function_reference(bcs, p=(1, 1))

        return bm.einsum("cim,qin->cqmn", points, gphi)

    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None = None,
        bcs: tuple[Tensor, ...] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        prism = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(prism.shape) == 1:
            prism = bm.reshape(prism, (1, -1))
        if bcs is None:
            bcs = (
                bm.asarray([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=ctx.block.positions.dtype),
                bm.asarray([[0.5, 0.5]], dtype=ctx.block.positions.dtype),
            )
            squeeze_q = True
        else:
            bcs = _require_bcs_tuple(bcs, "prism grad_lambda", 2)
            squeeze_q = False
        ref3 = cls.grad_shape_function(ctx, bcs, p=(1, 1), index=index, variables="u")
        z3 = bm.zeros((ref3.shape[0], ref3.shape[1], 2), dtype=ref3.dtype)
        ref5 = bm.concatenate([ref3[:, :, 0:2], z3[:, :, 0:1], ref3[:, :, 2:3], z3[:, :, 1:2]], axis=-1)
        if ref:
            grad = bm.broadcast_to(ref5[None, :, :, :], (prism.shape[0], ref5.shape[0], 6, 5))
            return grad[:, 0, :, :] if squeeze_q else grad
        G, J = cls.first_fundamental_form(ctx, bcs, index=index, return_jacobi=True)
        Ginv = bm.linalg.inv(G)
        grad = bm.einsum("cqdk,cqkl,qil->cqid", J, Ginv, ref3)
        return grad[:, 0, :, :] if squeeze_q else grad

    # ipoint
    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        """Compute the multi-index matrix on reference prism.

        Return tensor-product multi-index of triangle and interval.
        """
        p0, p1 = _require_order_tuple(order, "prism multi_index", 2)

        if internal:
            mi0 = _MI.multi_index_inner(p0, 3)
            mi1 = _MI.multi_index_inner(p1, 2)
        else:
            mi0 = _MI.multi_index_matrix(p0, 3)
            mi1 = _MI.multi_index_matrix(p1, 2)

        shape = (mi1.shape[0], mi0.shape[0])
        mi0 = bm.broadcast_to(mi0[None, :, :], shape + (3,))
        mi1 = bm.broadcast_to(mi1[:, None, :], shape + (2,))

        mi = bm.concat([mi0, mi1], axis=-1).reshape(-1, 5)
        if tensorprod:
            return multi_index_tensorprod(mi, (3,))
        return mi

    @classmethod
    def bc_to_point(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        index: Index | None
    ) -> Tensor:
        """Convert barycentric coordinates to Cartesian coordinates.

        x = sum_i phi_i x_i on the physical prism.
        """
        phi = cls.shape_function(bcs, (1, 1))
        points = cls._tp_points(ctx, index)
        return bm.einsum("cim,qi->cqm", points, phi)

    @classmethod
    def first_fundamental_form(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        index: Index | None = None,
        return_jacobi: bool = False,
        return_grad: bool = False
    ):
        """Compute the first fundamental form of the Lagrange prism.

        G = J^T J, where J is the Jacobian matrix of the reference-to-physical map.
        """
        J = cls.jacobi_matrix(ctx, bcs, index=index)
        gphi = cls.grad_shape_function(ctx, bcs, p=(1, 1), variables="u")
        TD = J.shape[-1]
        shape = J.shape[0:-2] + (TD, TD)
        data = [[0 for _ in range(TD)] for _ in range(TD)]

        for i in range(TD):
            data[i][i] = bm.einsum("...d,...d->...", J[..., i], J[..., i])
            for j in range(i + 1, TD):
                data[i][j] = bm.einsum("...d,...d->...", J[..., i], J[..., j])
                data[j][i] = data[i][j]

        data = [val.reshape(val.shape + (1,)) for row in data for val in row]
        G = bm.concatenate(data, axis=-1).reshape(shape)

        if not return_jacobi and not return_grad:
            return G
        if return_jacobi and not return_grad:
            return G, J
        if not return_jacobi and return_grad:
            return G, gphi
        return G, J, gphi
