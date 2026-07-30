
from ....backend import bm
from ....backend import Index, Tensor
from ...ipoints import MultiIndex as _MI, multi_index_tensorprod
from .base import (
    EntityContext,
    ShapedEntitySchema,
    _require_bcs_tuple,
    _require_order_tuple,
)

__all__ = ["QuadrilateralSchema"]


class QuadrilateralSchema(ShapedEntitySchema):
    name = "quad"
    top_dim = 2
    OFace = {
        "segment": [[0, 1], [1, 2], [2, 3], [3, 0]],
        "point": [[0], [1], [2], [3]],
    }
    SFace = {
        "segment": [[0, 1], [1, 2], [2, 3], [0, 3]],
        "point": [[0], [1], [2], [3]],
    }
    orientation = [
        (0, 1, 2, 3), (2, 0, 3, 1), (3, 2, 1, 0), (1, 3, 0, 2),
        (2, 3, 0, 1), (0, 2, 1, 3), (1, 0, 3, 2), (3, 1, 2, 0),
    ]
    ccw = [0, 1, 3, 2]
    _tp_to_contract = [0, 1, 3, 2]

    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        px, py = _require_order_tuple(order, "quadrilateral multi_index", 2)

        if internal:
            ix = _MI.multi_index_inner(px, 2)
            iy = _MI.multi_index_inner(py, 2)
        else:
            ix = _MI.multi_index_matrix(px, 2)
            iy = _MI.multi_index_matrix(py, 2)

        shape = (iy.shape[0], ix.shape[0], 2)
        multi_index0 = bm.broadcast_to(ix[None, :, :], shape).reshape(-1, 2)
        multi_index1 = bm.broadcast_to(iy[:, None, :], shape).reshape(-1, 2)
        mi = bm.concat([multi_index0, multi_index1], axis=1)
        if tensorprod:
            return multi_index_tensorprod(mi, (2,))
        return mi

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        quad = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[quad]
        return bm.mean(points, axis=1)

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "quadrilateral bc_to_point", 2)
        quad = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[quad[:, [0, 1, 3, 2]]]
        bc0 = bcs[0].reshape(-1, 2)
        bc1 = bcs[1].reshape(-1, 2)
        bc = bm.einsum("im,jn->ijmn", bc1, bc0).reshape(-1, 4)
        return bm.einsum("qj,cjd->cqd", bc, points)

    @classmethod
    def shape_function(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "quadrilateral shape_function", 2)
        p = _require_order_tuple(p, "quadrilateral shape_function", 2)

        mi0 = _MI.multi_index_matrix(p[0], 2)
        mi1 = _MI.multi_index_matrix(p[1], 2)

        return bm.tensorprod(
            bm.simplex_shape_function(bcs[1], p[1], mi1),
            bm.simplex_shape_function(bcs[0], p[0], mi0),
        )

    @classmethod
    def grad_shape_function_barycentric(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "quadrilateral grad_shape_function", 2)
        p = _require_order_tuple(p, "quadrilateral grad_shape_function", 2)

        mi0 = _MI.multi_index_matrix(p[0], 2)
        mi1 = _MI.multi_index_matrix(p[1], 2)
        phi0 = bm.simplex_shape_function(bcs[0], p=p[0], mi=mi0)
        phi1 = bm.simplex_shape_function(bcs[1], p=p[1], mi=mi1)
        R0 = bm.simplex_grad_shape_function(bcs[0], p=p[0], mi=mi0)
        R1 = bm.simplex_grad_shape_function(bcs[1], p=p[1], mi=mi1)
        num_shape_functions = cls.num_multi_index(p)

        gphi0 = bm.einsum('im, jng -> ijmng', phi1, R0).reshape(-1, num_shape_functions, 2)
        gphi1 = bm.einsum('img, jn -> ijmng', R1, phi0).reshape(-1, num_shape_functions, 2)
        gphi = gphi0[..., None, :] + gphi1[..., None, :]
        return bm.reshape(gphi, (-1, num_shape_functions, 4))

    @classmethod
    def grad_shape_function_reference(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "quadrilateral grad_shape_function_reference", 2)
        p = _require_order_tuple(p, "quadrilateral grad_shape_function_reference", 2)

        Dlambda = bm.array([-1, 1], dtype=bm.float64, device=bm.get_device(bcs[0]))
        mi0 = _MI.multi_index_matrix(p[0], 2)
        mi1 = _MI.multi_index_matrix(p[1], 2)
        phi0 = bm.simplex_shape_function(bcs[0], p=p[0], mi=mi0)
        phi1 = bm.simplex_shape_function(bcs[1], p=p[1], mi=mi1)
        R0 = bm.simplex_grad_shape_function(bcs[0], p=p[0], mi=mi0)
        R1 = bm.simplex_grad_shape_function(bcs[1], p=p[1], mi=mi1)
        dphi0 = bm.einsum('...ij, j->...i', R0, Dlambda)
        dphi1 = bm.einsum('...ij, j->...i', R1, Dlambda)

        num_shape_functions = phi0.shape[-1]**2

        gphi0 = bm.einsum('im, jn -> ijmn', dphi0, phi1).reshape(-1, num_shape_functions, 1)
        gphi1 = bm.einsum('im, jn -> ijmn', phi0, dphi1).reshape(-1, num_shape_functions, 1)
        return bm.concat((gphi0, gphi1), axis=-1)

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None):
        from ....quadrature import GaussLegendreQuadrature, TensorProductQuadrature
        qf = GaussLegendreQuadrature(q, device=device)
        return TensorProductQuadrature((qf, qf))

    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None,
        bcs: tuple[Tensor, ...] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        quad = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(quad.shape) == 1:
            quad = bm.reshape(quad, (1, -1))
        if bcs is None:
            if not ref:
                points = ctx.block.positions[quad]
                vr = 0.5 * ((points[:, 1, :] - points[:, 0, :]) + (points[:, 2, :] - points[:, 3, :]))
                vs = 0.5 * ((points[:, 3, :] - points[:, 0, :]) + (points[:, 2, :] - points[:, 1, :]))
                jac = bm.stack([vr, vs], axis=-1)
                jac_t = bm.einsum("nij->nji", jac)
                metric = bm.einsum("nik,nkj->nij", jac_t, jac)
                metric_inv = bm.linalg.inv(metric)
                grads = bm.einsum("nik,nkj->nij", metric_inv, jac_t)
                ref_grads = bm.asarray([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=points.dtype)
                return bm.einsum("ld, ndg->nlg", ref_grads, grads)
            bcs = (
                bm.asarray([[0.5, 0.5]], dtype=ctx.block.positions.dtype),
                bm.asarray([[0.5, 0.5]], dtype=ctx.block.positions.dtype),
            )
            squeeze_q = True
        else:
            bcs = _require_bcs_tuple(bcs, "quadrilateral grad_lambda", 2)
            squeeze_q = False
        u, v = bcs
        u0, u1 = u[:, 0], u[:, 1]
        v0 = bm.broadcast_to(v[:, 0], u0.shape)
        v1 = bm.broadcast_to(v[:, 1], u0.shape)
        z_u = bm.zeros_like(u0)
        z_v = bm.zeros_like(v0)
        ref_u = bm.stack([
            bm.stack([v0, z_u, u0, z_u], axis=-1),
            bm.stack([z_u, v0, u1, z_u], axis=-1),
            bm.stack([v1, z_u, z_u, u0], axis=-1),
            bm.stack([z_u, v1, z_u, u1], axis=-1),
        ], axis=1)
        if ref:
            grad = bm.broadcast_to(ref_u[None, :, :, :], (quad.shape[0], ref_u.shape[0], 4, 4))
            return grad[:, 0, :, :] if squeeze_q else grad

        dphi_duv = bm.stack([
            bm.stack([-v0, -u0], axis=-1),
            bm.stack([ v0, -u1], axis=-1),
            bm.stack([-v1,  u0], axis=-1),
            bm.stack([ v1,  u1], axis=-1),
        ], axis=1)
        points = ctx.block.positions[quad]
        J = bm.einsum("qit,cid->cqtd", dphi_duv, points)
        Jt = bm.einsum("cqtd->cqdt", J)
        metric = bm.einsum("cqtd,cqsd->cqts", J, J)
        metric_inv = bm.linalg.inv(metric)
        grad = bm.einsum("cqdt,cqts,qis->cqid", Jt, metric_inv, dphi_duv)
        return grad[:, 0, :, :] if squeeze_q else grad

    @classmethod
    def jacobi_matrix(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        index: Index | None,
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "quadrilateral jacobi_matrix", 2)
        node = ctx.block.positions
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        gphi = cls.grad_shape_function_reference(bcs, p=(1, 1)) # (NQ, num_shape, ref_dim)
        cell = cell[:, cls._tp_to_contract]
        J = bm.einsum('cim, qin -> cqmn', node[cell], gphi) # (NC, NQ, GD, ref_dim)

        return J

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        quad = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[quad]
        v0 = points[:, 1, :] - points[:, 0, :]
        v1 = points[:, 2, :] - points[:, 0, :]
        v2 = points[:, 3, :] - points[:, 0, :]
        if points.shape[-1] == 2:
            cross01 = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
            cross12 = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            return 0.5 * (bm.abs(cross01) + bm.abs(cross12))
        cross0 = bm.cross(v0, v1)
        cross1 = bm.cross(v1, v2)
        area0 = bm.sqrt(bm.sum(cross0 * cross0, axis=1))
        area1 = bm.sqrt(bm.sum(cross1 * cross1, axis=1))
        return 0.5 * (area0 + area1)

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        quad = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[quad]
        gd = int(points.shape[2])
        if gd == 2:
            return bm.zeros((points.shape[0], 0, gd), **bm.context(points))
        if gd == 3:
            v1 = points[:, 1, :] - points[:, 0, :]
            v2 = points[:, 2, :] - points[:, 0, :]
            v3 = points[:, 3, :] - points[:, 0, :]
            normal = bm.cross(v1, v2) + bm.cross(v2, v3)
            return bm.expand_dims(normal, axis=1)
        raise ValueError(f"unsupported geometric dimension: {gd}")

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        quad = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[quad]
        vr = 0.5 * ((points[:, 1, :] - points[:, 0, :]) + (points[:, 2, :] - points[:, 3, :]))
        vs = 0.5 * ((points[:, 3, :] - points[:, 0, :]) + (points[:, 2, :] - points[:, 1, :]))
        return bm.stack([vr, vs], axis=1)
