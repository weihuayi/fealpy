
from ....backend import bm
from ....backend import Index, Tensor
from ...ipoints import MultiIndex as _MI, multi_index_tensorprod
from .base import (
    EntityContext,
    ShapedEntitySchema,
    _require_bcs_tuple,
    _require_order_tuple,
)

__all__ = ["HexahedronSchema"]


class HexahedronSchema(ShapedEntitySchema):
    name = "hex"
    top_dim = 3
    OFace = {
        "quad": [
            [0, 3, 2, 1], [4, 5, 6, 7],
            [0, 1, 5, 4], [2, 3, 7, 6],
            [0, 4, 7, 3], [1, 2, 6, 5],
        ],
        "segment": [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [0, 4], [1, 5], [2, 6], [3, 7],
            [4, 5], [5, 6], [6, 7], [7, 4],
        ],
        "point": [[0], [1], [2], [3], [4], [5], [6], [7]],
    }
    SFace = {
        "quad": [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 4, 5], [2, 3, 6, 7],
            [0, 3, 4, 7], [1, 2, 5, 6],
        ],
        "segment": [
            [0, 1], [1, 2], [2, 3], [0, 3],
            [0, 4], [1, 5], [2, 6], [3, 7],
            [4, 5], [5, 6], [6, 7], [4, 7],
        ],
        "point": [[0], [1], [2], [3], [4], [5], [6], [7]],
    }
    _tp_to_contract = [0, 1, 3, 2, 4, 5, 7, 6]

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        return bm.mean(ctx.block.positions[cell], axis=1)

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "hexahedron bc_to_point", 3)
        for bc in bcs:
            if bc.shape[-1] != 2:
                raise ValueError(
                    f"hexahedron barycentric coordinate tensors expect last dimension 2, got {bc.shape[-1]}"
                )

        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        # Contract order is bottom/top cyclic; tensor-product order is different.
        points = ctx.block.positions[cell[:, [0, 1, 3, 2, 4, 5, 7, 6]]]
        points = bm.reshape(points, (-1, 2, 2, 2, cls.geo_dimension(ctx)))
        u, v, w = bcs
        result = bm.einsum("ia,jb,kc,ncbae->nkjie", u, v, w, points)
        return bm.reshape(result, (result.shape[0], -1, result.shape[-1]))

    @classmethod
    def shape_function(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "hexahedron shape_function", 3)
        p = _require_order_tuple(p, "hexahedron shape_function", 3)

        mi0 = _MI.multi_index_matrix(p[0], 2)
        mi1 = _MI.multi_index_matrix(p[1], 2)
        mi2 = _MI.multi_index_matrix(p[2], 2)
        phi = bm.tensorprod(
            bm.simplex_shape_function(bcs[2], p[2], mi2),
            bm.simplex_shape_function(bcs[1], p[1], mi1),
            bm.simplex_shape_function(bcs[0], p[0], mi0),
        )
        return phi[..., cls._tp_to_contract] if phi.shape[-1] == 8 else phi

    @classmethod
    def grad_shape_function_barycentric(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "hexahedron grad_shape_function_barycentric", 3)
        p = _require_order_tuple(p, "hexahedron grad_shape_function_barycentric", 3)
        for bc in bcs:
            if bc.shape[-1] != 2:
                raise ValueError(
                    "hexahedron grad_shape_function_barycentric expects "
                    "three interval barycentric tensors"
                )

        mi0 = _MI.multi_index_matrix(p[0], 2)
        mi1 = _MI.multi_index_matrix(p[1], 2)
        mi2 = _MI.multi_index_matrix(p[2], 2)
        phi0 = bm.simplex_shape_function(bcs[0], p[0], mi0)
        phi1 = bm.simplex_shape_function(bcs[1], p[1], mi1)
        phi2 = bm.simplex_shape_function(bcs[2], p[2], mi2)
        R0 = bm.simplex_grad_shape_function(bcs[0], p[0], mi0)
        R1 = bm.simplex_grad_shape_function(bcs[1], p[1], mi1)
        R2 = bm.simplex_grad_shape_function(bcs[2], p[2], mi2)
        num_shape = phi0.shape[-1] * phi1.shape[-1] * phi2.shape[-1]

        gphi0 = bm.einsum("ka,jb,icr->kijabcr", phi2, phi1, R0)
        gphi1 = bm.einsum("ka,jbr,ic->kijabcr", phi2, R1, phi0)
        gphi2 = bm.einsum("kar,jb,ic->kijabcr", R2, phi1, phi0)
        gphi = bm.concat(
            [
                bm.reshape(gphi0, (-1, num_shape, 2)),
                bm.reshape(gphi1, (-1, num_shape, 2)),
                bm.reshape(gphi2, (-1, num_shape, 2)),
            ],
            axis=-1,
        )
        return gphi[..., cls._tp_to_contract, :] if gphi.shape[-2] == 8 else gphi

    @classmethod
    def grad_shape_function_reference(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "hexahedron grad_shape_function_reference", 3)
        p = _require_order_tuple(p, "hexahedron grad_shape_function_reference", 3)

        grad_bary = cls.grad_shape_function_barycentric(bcs, p)
        return bm.stack(
            [
                -grad_bary[..., 0] + grad_bary[..., 1],
                -grad_bary[..., 2] + grad_bary[..., 3],
                -grad_bary[..., 4] + grad_bary[..., 5],
            ],
            axis=-1,
        )

    @classmethod
    def jacobi_matrix(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        index: Index | None,
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "hexahedron jacobi_matrix", 3)
        node = ctx.block.positions
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        gphi = cls.grad_shape_function_reference(bcs, p=(1, 1, 1))
        return bm.einsum("cim,qin->cqmn", node[cell], gphi)

    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        order = _require_order_tuple(order, "hexahedron multi_index", 3)
        px, py, pz = order

        if internal:
            ix = _MI.multi_index_inner(px, 2)
            iy = _MI.multi_index_inner(py, 2)
            iz = _MI.multi_index_inner(pz, 2)
        else:
            ix = _MI.multi_index_matrix(px, 2)
            iy = _MI.multi_index_matrix(py, 2)
            iz = _MI.multi_index_matrix(pz, 2)
        shape = (iz.shape[0], iy.shape[0], ix.shape[0], 2)
        multi_index0 = bm.broadcast_to(ix[None, None, :, :], shape).reshape(-1, 2)
        multi_index1 = bm.broadcast_to(iy[None, :, None, :], shape).reshape(-1, 2)
        multi_index2 = bm.broadcast_to(iz[:, None, None, :], shape).reshape(-1, 2)
        mi = bm.concat([multi_index0, multi_index1, multi_index2], axis=-1)
        if tensorprod:
            return multi_index_tensorprod(mi, (2, 4))
        return mi

    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None,
        bcs: tuple[Tensor, ...] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        if bcs is None:
            bcs = (
                bm.asarray([[0.5, 0.5]], dtype=ctx.block.positions.dtype),
                bm.asarray([[0.5, 0.5]], dtype=ctx.block.positions.dtype),
                bm.asarray([[0.5, 0.5]], dtype=ctx.block.positions.dtype),
            )
            squeeze_q = True
        else:
            bcs = _require_bcs_tuple(bcs, "hexahedron grad_lambda", 3)
            squeeze_q = False
        u, v, w = bcs
        u0, u1 = u[:, 0], u[:, 1]
        v0, v1 = v[:, 0], v[:, 1]
        w0, w1 = w[:, 0], w[:, 1]
        z = bm.zeros_like(u0)

        def ref_row(ua, ub, va, vb, wa, wb):
            return bm.stack([ua, ub, va, vb, wa, wb], axis=-1)

        ref_grad = bm.stack([
            ref_row(v0*w0, z, u0*w0, z, u0*v0, z),
            ref_row(z, v0*w0, u1*w0, z, u1*v0, z),
            ref_row(v1*w0, z, z, u0*w0, u0*v1, z),
            ref_row(z, v1*w0, z, u1*w0, u1*v1, z),
            ref_row(v0*w1, z, u0*w1, z, z, u0*v0),
            ref_row(z, v0*w1, u1*w1, z, z, u1*v0),
            ref_row(v1*w1, z, z, u0*w1, z, u0*v1),
            ref_row(z, v1*w1, z, u1*w1, z, u1*v1),
        ], axis=1)
        if ref:
            ref_grad = ref_grad[..., cls._tp_to_contract, :]
            grad = bm.broadcast_to(ref_grad[None, :, :, :], (cell.shape[0], ref_grad.shape[0], 8, 6))
            return grad[:, 0, :, :] if squeeze_q else grad

        dphi = bm.stack([
            bm.stack([-v0*w0, -u0*w0, -u0*v0], axis=-1),
            bm.stack([ v0*w0, -u1*w0, -u1*v0], axis=-1),
            bm.stack([-v1*w0,  u0*w0, -u0*v1], axis=-1),
            bm.stack([ v1*w0,  u1*w0, -u1*v1], axis=-1),
            bm.stack([-v0*w1, -u0*w1,  u0*v0], axis=-1),
            bm.stack([ v0*w1, -u1*w1,  u1*v0], axis=-1),
            bm.stack([-v1*w1,  u0*w1,  u0*v1], axis=-1),
            bm.stack([ v1*w1,  u1*w1,  u1*v1], axis=-1),
        ], axis=1)
        dphi = dphi[:, cls._tp_to_contract, :]
        points = ctx.block.positions[cell]
        J = bm.einsum("qit,cid->cqtd", dphi, points)
        Jt = bm.einsum("cqtd->cqdt", J)
        metric = bm.einsum("cqtd,cqsd->cqts", J, J)
        metric_inv = bm.linalg.inv(metric)
        grad = bm.einsum("cqdt,cqts,qis->cqid", Jt, metric_inv, dphi)
        return grad[:, 0, :, :] if squeeze_q else grad

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        gd = cls.geo_dimension(ctx)
        if gd != 3:
            raise ValueError(f"hexahedron geometry requires GD == 3, got {gd}")

        qf = cls.quadrature_formula(2)
        bcs, ws = qf.get_quadrature_points_and_weights()
        u, v, w = bcs
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        points = ctx.block.positions[cell[:, cls._tp_to_contract]]
        points = bm.reshape(points, (-1, 2, 2, 2, 3))
        du = bm.broadcast_to(bm.asarray([-1.0, 1.0], dtype=ctx.block.positions.dtype)[None, :], u.shape)
        dv = bm.broadcast_to(bm.asarray([-1.0, 1.0], dtype=ctx.block.positions.dtype)[None, :], v.shape)
        dw = bm.broadcast_to(bm.asarray([-1.0, 1.0], dtype=ctx.block.positions.dtype)[None, :], w.shape)

        ju = bm.einsum("ia,jb,kc,ncbae->nijke", du, v, w, points)
        jv = bm.einsum("ia,jb,kc,ncbae->nijke", u, dv, w, points)
        jw = bm.einsum("ia,jb,kc,ncbae->nijke", u, v, dw, points)
        jac = bm.stack([ju, jv, jw], axis=-1)
        det = bm.abs(bm.linalg.det(jac))
        weight = bm.reshape(ws, (2, 2, 2))
        return bm.einsum("ijk,nijk->n", weight, det)

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        gd = cls.geo_dimension(ctx)
        if gd != 3:
            raise ValueError(f"hexahedron geometry requires GD == 3, got {gd}")
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        return bm.zeros((cell.shape[0], 0, 3), dtype=ctx.block.positions.dtype)

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None):
        if qtype not in (None, "legendre"):
            raise ValueError(f"unsupported hexahedron quadrature type: {qtype!r}")
        from fealpy.quadrature import GaussLegendreQuadrature, TensorProductQuadrature

        qf = GaussLegendreQuadrature(q, device=device)
        return TensorProductQuadrature((qf, qf, qf))

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        gd = cls.geo_dimension(ctx)
        if gd != 3:
            raise ValueError(f"hexahedron geometry requires GD == 3, got {gd}")
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        points = ctx.block.positions[cell]
        t0 = points[:, 1, :] - points[:, 0, :]
        t1 = points[:, 3, :] - points[:, 0, :]
        t2 = points[:, 4, :] - points[:, 0, :]
        return bm.stack([t0, t1, t2], axis=1)
