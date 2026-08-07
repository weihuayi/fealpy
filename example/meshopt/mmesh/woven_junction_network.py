import argparse
import csv
import types
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from fealpy.backend import bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.mesh import TriangleMesh
from fealpy.mmesh import Config, MMesher, MeshQuality


SQUARE_VERTICES = bm.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=bm.float64,
)


def parse_list(text, cast=float):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def build_unit_square_mesh(n, diagonal="checkerboard"):
    xs = np.linspace(0.0, 1.0, n + 1)
    ys = np.linspace(0.0, 1.0, n + 1)
    nodes = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)

    def node_id(i, j):
        return j * (n + 1) + i

    cells = []
    for j in range(n):
        for i in range(n):
            v00 = node_id(i, j)
            v10 = node_id(i + 1, j)
            v01 = node_id(i, j + 1)
            v11 = node_id(i + 1, j + 1)
            if diagonal == "slash":
                flip = False
            elif diagonal == "backslash":
                flip = True
            else:
                flip = ((i + j) & 1) == 1
            if flip:
                cells.extend([(v00, v10, v01), (v10, v11, v01)])
            else:
                cells.extend([(v00, v10, v11), (v00, v11, v01)])

    mesh = TriangleMesh(bm.asarray(nodes, dtype=bm.float64), bm.asarray(cells, dtype=bm.int32))
    mesh.meshdata["vertices"] = SQUARE_VERTICES
    return mesh


def junction_coords(nj):
    return np.arange(1, nj + 1, dtype=np.float64) / (nj + 1)


def woven_width(pressure_a, ell0):
    return ell0 / pressure_a


def woven_junction_metric_raw(points, *, nj, pressure_a, ell0=0.8):
    points = np.asarray(points, dtype=np.float64)
    x = points[..., 0]
    y = points[..., 1]
    coords = junction_coords(nj)
    w = woven_width(pressure_a, ell0)

    phi_h = np.zeros_like(x, dtype=np.float64)
    phi_v = np.zeros_like(x, dtype=np.float64)
    for c in coords:
        phi_h += np.exp(-((y - c) / w) ** 4)
        phi_v += np.exp(-((x - c) / w) ** 4)

    amp = pressure_a * pressure_a - 1.0
    M = np.zeros(points.shape[:-1] + (2, 2), dtype=np.float64)
    M[..., 0, 0] = 1.0 + amp * phi_v
    M[..., 1, 1] = 1.0 + amp * phi_h
    return M


def woven_masks(points, *, nj, pressure_a, ell0=0.8):
    points = np.asarray(points, dtype=np.float64)
    x = points[..., 0]
    y = points[..., 1]
    coords = junction_coords(nj)
    w = woven_width(pressure_a, ell0)

    dx = np.min(np.abs(x[..., None] - coords), axis=-1)
    dy = np.min(np.abs(y[..., None] - coords), axis=-1)
    core = (dx <= 2.0 * w) & (dy <= 2.0 * w)
    horizontal = (dy <= 2.0 * w) & ~core
    vertical = (dx <= 2.0 * w) & ~core
    arms = horizontal | vertical
    network = core | arms
    return {
        "core": core,
        "horizontal_arms": horizontal,
        "vertical_arms": vertical,
        "arms": arms,
        "network": network,
    }


def reference_scale(args, nj, pressure_a):
    n = args.ref_n
    grid = (np.arange(n, dtype=np.float64) + 0.5) / n
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    M_raw = woven_junction_metric_raw(points, nj=nj, pressure_a=pressure_a, ell0=args.ell0)
    sigma_raw = float(np.mean(np.sqrt(np.linalg.det(M_raw))))
    return 1.0 / sigma_raw


def install_woven_monitor(adaptiver, *, args, nj, pressure_a, scale):
    key = "_woven_junction_metric_registered"
    if not getattr(adaptiver.__class__, key, False):
        @adaptiver.__class__.monitor.register("woven_junction_metric")
        def monitor(self):
            node = np.asarray(bm.to_numpy(self.mesh.node))
            M_raw = woven_junction_metric_raw(
                node,
                nj=self._woven_nj,
                pressure_a=self._woven_pressure_a,
                ell0=self._woven_ell0,
            )
            self.M_node = bm.asarray(self._woven_scale * M_raw, dtype=bm.float64)
            self.M = bm.mean(self.M_node[self.cell], axis=1)
            self._woven_sigma = 1.0

        setattr(adaptiver.__class__, key, True)

    adaptiver._woven_nj = nj
    adaptiver._woven_pressure_a = pressure_a
    adaptiver._woven_ell0 = args.ell0
    adaptiver._woven_scale = scale
    adaptiver.monitor.set("woven_junction_metric")


def install_identity_interpolation(adaptiver):
    key = "_woven_identity_interpolation_registered"
    if not getattr(adaptiver.__class__, key, False):
        @adaptiver.__class__.interpolate.register("identity")
        def interpolate(self, moved_node):
            return self.uh

        setattr(adaptiver.__class__, key, True)
    adaptiver.interpolate.set("identity")


def install_fixed_boundary(adaptiver):
    bd = np.asarray(bm.to_numpy(adaptiver.mesh.boundary_node_flag()), dtype=bool)
    idx = np.flatnonzero(bd)
    mask = np.ones(2 * adaptiver.NN, dtype=np.float64)
    mask[idx] = 0.0
    mask[adaptiver.NN + idx] = 0.0
    projector = sp.diags(mask, format="csr")
    raw_vector = adaptiver.vector_construction
    raw_jac = adaptiver.JAC_functional

    def fixed_vector(self, A, g, trA, E_hat, *args, **kwargs):
        ret = raw_vector(A, g, trA, E_hat, *args, **kwargs)
        if isinstance(ret, tuple):
            v, local = ret
            return bm.set_at(v, bm.asarray(idx), 0.0), local
        return bm.set_at(ret, bm.asarray(idx), 0.0)

    def fixed_jac(*pargs, **kwargs):
        return projector @ raw_jac(*pargs, **kwargs).tocsr()

    adaptiver.vector_construction = types.MethodType(fixed_vector, adaptiver)
    adaptiver.JAC_functional = fixed_jac
    adaptiver.theta = types.MethodType(lambda self, M: 1.0, adaptiver)


def install_huang_mu(adaptiver, mu):
    adaptiver._woven_mu = float(mu)

    def I_func(self, trA, rho, g):
        d = self.GD
        gamma = self.gamma
        mu_ = self._woven_mu
        return bm.sum(self.cm * rho * (
            mu_ * trA ** (d * gamma / 2)
            + d ** (d * gamma / 2) * (1.0 - 2.0 * mu_) * g ** (gamma / 2)
        ))

    def TdA(self, trA):
        d = self.GD
        gamma = self.gamma
        mu_ = self._woven_mu
        return (mu_ * (d * gamma / 2) * trA ** (d * gamma / 2 - 1))[..., None, None] * self.I_p

    def Tdg(self, g):
        d = self.GD
        gamma = self.gamma
        mu_ = self._woven_mu
        return d ** (d * gamma / 2) * (1.0 - 2.0 * mu_) * (gamma / 2) * g ** (gamma / 2 - 1)

    adaptiver.I_func = types.MethodType(I_func, adaptiver)
    adaptiver.TdA = types.MethodType(TdA, adaptiver)
    adaptiver.Tdg = types.MethodType(Tdg, adaptiver)


def edge_matrix(mesh, node=None):
    if node is None:
        node = mesh.entity("node")
    cell = mesh.entity("cell")
    x0 = node[cell[:, 0]]
    E = node[cell[:, 1:]] - x0[:, None, :]
    return bm.permute_dims(E, axes=(0, 2, 1))


def local_alignment(mesh, logic_mesh, M):
    E = edge_matrix(mesh)
    Ehat = edge_matrix(logic_mesh)
    J = E @ bm.linalg.inv(Ehat)
    A = bm.permute_dims(J, axes=(0, 2, 1)) @ M @ J
    trA = bm.trace(A, axis1=-2, axis2=-1)
    detA = bm.linalg.det(A)
    return trA / (2.0 * bm.sqrt(detA))


def metric_volume_ratio(mesh, M):
    cm = mesh.entity_measure("cell")
    rho = bm.sqrt(bm.linalg.det(M))
    return cm * rho * mesh.number_of_cells()


def weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or np.sum(weights) <= 0.0:
        return np.nan
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    return float(values[np.searchsorted(cdf, quantile * cdf[-1], side="left")])


def weighted_cvar(values, weights, quantile=0.95):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or np.sum(weights) <= 0.0:
        return np.nan
    q = weighted_quantile(values, weights, quantile)
    tail = values >= q
    if not np.any(tail):
        return q
    return float(np.sum(weights[tail] * values[tail]) / np.sum(weights[tail]))


def compression_response_state(adaptiver, Xi_new, *, args, nj, pressure_a, mu):
    mesh = adaptiver.mesh
    cache = adaptiver._ivp_cache
    E_hat = edge_matrix(mesh, bm.asarray(Xi_new, dtype=bm.float64))
    M_inv = bm.linalg.inv(adaptiver.M)
    A = adaptiver.A(cache["E_K"], E_hat, M_inv)
    detA = np.asarray(bm.to_numpy(bm.linalg.det(A)), dtype=np.float64)
    trA = np.asarray(bm.to_numpy(bm.trace(A, axis1=-2, axis2=-1)), dtype=np.float64)
    theta = float(cache["theta"])
    d = adaptiver.GD
    gamma = adaptiver.gamma
    p = d * gamma / 2.0

    q = theta ** (d / 2.0) / np.sqrt(np.maximum(detA, np.finfo(np.float64).tiny))
    ell = np.maximum(np.log(np.maximum(q, np.finfo(np.float64).tiny)), 0.0)
    S = trA ** p
    pi_tl = gamma * ((d * theta) ** p - S)
    mu_eff = float(mu) if np.isfinite(mu) else 1.0 / 3.0
    T_h = mu_eff * S + (1.0 - 2.0 * mu_eff) * (d ** p) * detA ** (gamma / 2.0)
    pi_h = -gamma * T_h

    bary = np.asarray(bm.to_numpy(bm.mean(mesh.entity("node")[mesh.entity("cell")], axis=1)))
    core = woven_masks(bary, nj=nj, pressure_a=pressure_a, ell0=args.ell0)["core"]
    weights = np.asarray(
        bm.to_numpy(cache["cm"] * cache["rho"]),
        dtype=np.float64,
    )
    core_weights = weights[core]
    Wc = float(np.sum(core_weights))
    if Wc <= 0.0 or not np.any(core):
        core_L_peak = np.nan
        core_L_cvar95 = np.nan
        U_tl = np.nan
        U_h = np.nan
        X_h_to_tl = np.nan
        tl_positive_fraction = np.nan
        compressed_fraction = np.nan
    else:
        ell_c = ell[core]
        pi_tl_c = pi_tl[core]
        pi_h_c = pi_h[core]
        core_L_peak = float(np.max(ell_c))
        core_L_cvar95 = weighted_cvar(ell_c, core_weights, 0.95)
        U_tl = float(np.sum(core_weights * ell_c * np.maximum(-pi_tl_c, 0.0)) / Wc)
        U_h = float(np.sum(core_weights * ell_c * np.maximum(-pi_h_c, 0.0)) / Wc)
        X_h_to_tl = float(np.sum(core_weights * ell_c * (pi_tl_c - pi_h_c)) / Wc)
        compressed = ell_c > 0.0
        compressed_fraction = float(np.sum(core_weights[compressed]) / Wc)
        tl_positive_fraction = float(np.sum(core_weights[pi_tl_c > 0.0]) / Wc)

    det_Ehat = np.asarray(bm.to_numpy(bm.linalg.det(E_hat)), dtype=np.float64)
    det0 = adaptiver._woven_det_Ehat0
    positive = det_Ehat > 0.0
    delta_adm = float(np.min(det_Ehat / det0)) if det_Ehat.size else np.nan

    return {
        "L_peak": core_L_peak,
        "L_cvar95": core_L_cvar95,
        "U_TL": U_tl,
        "U_H": U_h,
        "X_H_to_TL": X_h_to_tl,
        "delta_adm": delta_adm,
        "min_det_Ehat": float(np.min(det_Ehat)) if det_Ehat.size else np.nan,
        "positive_Ehat": bool(np.all(positive)),
        "core_compressed_weight_fraction": compressed_fraction,
        "core_tl_positive_weight_fraction": tl_positive_fraction,
        "core_q_max": float(np.max(q[core])) if np.any(core) else np.nan,
        "core_q_p95": float(np.quantile(q[core], 0.95)) if np.any(core) else np.nan,
    }


def install_compression_response_probe(adaptiver, *, args, method, nj, pressure_a, mu):
    Ehat0 = edge_matrix(adaptiver.logic_mesh)
    adaptiver._woven_det_Ehat0 = np.asarray(
        bm.to_numpy(bm.linalg.det(Ehat0)),
        dtype=np.float64,
    )
    stats = {
        "path_L_peak": 0.0,
        "path_L_cvar95": 0.0,
        "path_U_TL": 0.0,
        "path_U_H": 0.0,
        "path_X_H_to_TL": 0.0,
        "path_delta_adm": np.inf,
        "path_min_det_Ehat": np.inf,
        "path_positive_Ehat": True,
        "last_L_peak": np.nan,
        "last_L_cvar95": np.nan,
        "last_U_TL": np.nan,
        "last_U_H": np.nan,
        "last_X_H_to_TL": np.nan,
        "last_delta_adm": np.nan,
        "last_min_det_Ehat": np.nan,
        "last_positive_Ehat": np.nan,
        "last_core_compressed_weight_fraction": np.nan,
        "last_core_tl_positive_weight_fraction": np.nan,
        "last_core_q_max": np.nan,
        "last_core_q_p95": np.nan,
        "accepted_state_count": 0,
        "compression_probe_method": method,
    }
    raw_linear_interpolate = adaptiver.linear_interpolate

    def probed_linear_interpolate(self, Xi, Xi_new, X):
        Xnew = raw_linear_interpolate(Xi, Xi_new, X)
        state = compression_response_state(
            self, Xi_new, args=args, nj=nj, pressure_a=pressure_a, mu=mu
        )
        stats["accepted_state_count"] += 1
        stats["path_L_peak"] = max(stats["path_L_peak"], state["L_peak"])
        stats["path_L_cvar95"] = max(stats["path_L_cvar95"], state["L_cvar95"])
        stats["path_U_TL"] = max(stats["path_U_TL"], state["U_TL"])
        stats["path_U_H"] = max(stats["path_U_H"], state["U_H"])
        stats["path_X_H_to_TL"] = max(stats["path_X_H_to_TL"], state["X_H_to_TL"])
        stats["path_delta_adm"] = min(stats["path_delta_adm"], state["delta_adm"])
        stats["path_min_det_Ehat"] = min(stats["path_min_det_Ehat"], state["min_det_Ehat"])
        stats["path_positive_Ehat"] = bool(stats["path_positive_Ehat"] and state["positive_Ehat"])
        for key, value in state.items():
            stats[f"last_{key}"] = value
        return Xnew

    adaptiver.linear_interpolate = types.MethodType(probed_linear_interpolate, adaptiver)
    adaptiver._woven_compression_stats = stats


def summarize_region(out, prefix, mask, z, ali):
    mask = np.asarray(mask, dtype=bool)
    out[f"{prefix}_count"] = int(np.sum(mask))
    if not np.any(mask):
        out[f"{prefix}_S"] = np.nan
        out[f"{prefix}_P95"] = np.nan
        out[f"{prefix}_E_eq"] = np.nan
        out[f"{prefix}_Q_ali"] = np.nan
        return
    z_region = z[mask]
    ali_region = ali[mask]
    out[f"{prefix}_S"] = float(np.max(z_region))
    out[f"{prefix}_P95"] = float(np.quantile(z_region, 0.95))
    out[f"{prefix}_E_eq"] = float(np.sqrt(np.mean((z_region - 1.0) ** 2)))
    out[f"{prefix}_Q_ali"] = float(np.sqrt(np.mean(ali_region * ali_region)))


def collect_metrics(adaptiver, args, nj, pressure_a):
    adaptiver.monitor()
    mesh = adaptiver.mesh
    M = adaptiver.M
    mq = MeshQuality(mesh, adaptiver.logic_mesh, M)
    z = np.asarray(bm.to_numpy(metric_volume_ratio(mesh, M)))
    ali = np.asarray(bm.to_numpy(local_alignment(mesh, adaptiver.logic_mesh, M)))
    bary = np.asarray(bm.to_numpy(bm.mean(mesh.entity("node")[mesh.entity("cell")], axis=1)))
    masks = woven_masks(bary, nj=nj, pressure_a=pressure_a, ell0=args.ell0)
    cm = mesh.entity_measure("cell")

    out = {
        "num_cells": mesh.number_of_cells(),
        "num_junctions": nj * nj,
        "width": woven_width(pressure_a, args.ell0),
        "Q_eq": float(mq.Q_eq()),
        "Q_ali": float(mq.Q_ali()),
        "Q_geo": float(mq.Q_geo()),
        "min_cell_area": float(bm.min(cm)),
        "min_z": float(np.min(z)),
        "max_z": float(np.max(z)),
        "P95_z": float(np.quantile(z, 0.95)),
    }
    summarize_region(out, "core", masks["core"], z, ali)
    summarize_region(out, "arms", masks["arms"], z, ali)
    summarize_region(out, "horizontal_arms", masks["horizontal_arms"], z, ali)
    summarize_region(out, "vertical_arms", masks["vertical_arms"], z, ali)
    summarize_region(out, "network", masks["network"], z, ali)
    out["passes_core_s_tol"] = np.nan
    out["passes_arms_q_tol"] = np.nan
    if args.s_core_tol > 0.0:
        out["passes_core_s_tol"] = bool(out["core_S"] <= args.s_core_tol)
    if args.q_arms_tol > 0.0:
        out["passes_arms_q_tol"] = bool(out["arms_Q_ali"] <= args.q_arms_tol)
    out["passes_positive_area"] = bool(out["min_cell_area"] > 0.0)
    return out


def bdf_stats(adaptiver, args):
    out = {
        "bdf_max_steps": args.bdf_max_steps,
        "bdf_total_step_count": getattr(adaptiver, "_bdf_total_step_count", 0),
        "bdf_total_accepted_count": getattr(adaptiver, "_bdf_total_accepted_count", 0),
        "bdf_total_rejected_count": getattr(adaptiver, "_bdf_total_rejected_count", 0),
        "bdf_total_nonfinite_error_count": getattr(adaptiver, "_bdf_total_nonfinite_error_count", 0),
        "bdf_total_nonlinear_iteration_count": getattr(
            adaptiver, "_bdf_total_nonlinear_iteration_count", 0
        ),
        "bdf_stage_count": getattr(adaptiver, "_bdf_stage_count", 0),
        "bdf_last_step_count": getattr(adaptiver, "_last_bdf_step_count", np.nan),
        "bdf_last_accepted_count": getattr(adaptiver, "_last_bdf_accepted_count", np.nan),
        "bdf_last_rejected_count": getattr(adaptiver, "_last_bdf_rejected_count", np.nan),
        "bdf_last_nonfinite_error_count": getattr(adaptiver, "_last_bdf_nonfinite_error_count", np.nan),
        "bdf_last_nonlinear_iteration_count": getattr(
            adaptiver, "_last_bdf_nonlinear_iteration_count", np.nan
        ),
        "bdf_last_max_nonlinear_iteration_count": getattr(
            adaptiver, "_last_bdf_max_nonlinear_iteration_count", np.nan
        ),
        "bdf_last_total_time": getattr(adaptiver, "_last_bdf_total_time", np.nan),
        "bdf_last_h": getattr(adaptiver, "_last_bdf_h", np.nan),
        "bdf_last_min_h": getattr(adaptiver, "_last_bdf_min_h", np.nan),
        "bdf_last_scaled_error": getattr(adaptiver, "_last_bdf_scaled_error", np.nan),
    }
    total_steps = out["bdf_total_step_count"]
    out["bdf_mean_nonlinear_iteration_per_step"] = (
        out["bdf_total_nonlinear_iteration_count"] / total_steps
        if total_steps else np.nan
    )
    stats = getattr(adaptiver, "_smw_stats", None)
    if stats is not None:
        for key in (
            "factor_count",
            "symbolic_factor_count",
            "numeric_factor_count",
            "rank2_updates",
            "rank1_updates",
            "rank_stack_updates",
            "max_rank",
            "line_fail",
            "small_solve_fail",
            "b0_solve_count",
            "b0_solve_call_count",
            "multi_rhs_solve_count",
            "saved_w_solves",
            "last_res",
        ):
            out[f"smw_{key}"] = stats.get(key, np.nan)
        out["smw_factor_backend"] = stats.get("factor_backend", "")
    return out


def cleanup_adaptiver(adaptiver):
    cache = getattr(adaptiver, "_smw_mumps_cache", None)
    if cache is not None and cache.get("ctx") is not None:
        cache["ctx"].destroy()
        cache["ctx"] = None


def compression_stats(adaptiver):
    stats = dict(getattr(adaptiver, "_woven_compression_stats", {}))
    if stats.get("accepted_state_count", 0) == 0:
        for key in ("path_delta_adm", "path_min_det_Ehat"):
            if not np.isfinite(stats.get(key, np.nan)):
                stats[key] = np.nan
    return stats


def run_case(args, method, nj, pressure_a, mu):
    mesh = build_unit_square_mesh(args.nx, diagonal=args.diagonal)
    space = LagrangeFESpace(mesh, p=1)
    uh = space.function()
    uh[:] = 0.0

    config = Config()
    config.active_method = method
    config.is_pre = False
    config.tau = args.tau
    config.t_max = args.t_max
    config.gamma = args.gamma
    config.mol_times = args.mol_times
    config.alpha = args.alpha

    mm = MMesher(mesh=mesh, uh=uh, space=space, beta=0.5, config=config)
    mm.initialize()
    install_identity_interpolation(mm.instance)
    mm.set_mol_method("huangs_method")
    adaptiver = mm.instance
    adaptiver.total_steps = args.total_steps
    adaptiver.t_span = args.stage_time
    adaptiver.step = args.stage_substeps
    adaptiver.bdf_max_steps = None if args.bdf_max_steps <= 0 else args.bdf_max_steps
    adaptiver._bdf_total_step_count = 0
    adaptiver._bdf_total_accepted_count = 0
    adaptiver._bdf_total_rejected_count = 0
    adaptiver._bdf_stage_count = 0
    install_fixed_boundary(adaptiver)
    if method == "EAGAdaptiveHuang":
        install_huang_mu(adaptiver, mu)

    scale = reference_scale(args, nj, pressure_a)
    install_woven_monitor(adaptiver, args=args, nj=nj, pressure_a=pressure_a, scale=scale)
    install_compression_response_probe(
        adaptiver, args=args, method=method, nj=nj, pressure_a=pressure_a, mu=mu
    )

    print(
        f"Running woven method={method} nx={args.nx} nj={nj} "
        f"A={pressure_a:g} mu={mu:.6g}"
    )
    success = True
    failure_reason = ""
    failure_message = ""
    try:
        ret = adaptiver.mesh_redistributor(method=args.integrator, return_info=False)
        adaptiver._construct(ret["X"])
    except Exception as exc:
        success = False
        message = str(exc)
        failure_reason = (
            "bdf_step_limit_exceeded"
            if "BDF step limit exceeded" in message
            else exc.__class__.__name__
        )
        failure_message = message
        print(
            f"FAILED woven method={method} nx={args.nx} nj={nj} "
            f"A={pressure_a:g} mu={mu:.6g}: {failure_reason}: {failure_message}"
        )
    finally:
        cleanup_adaptiver(adaptiver)

    try:
        metrics = collect_metrics(adaptiver, args, nj, pressure_a)
    except Exception as exc:
        metrics = {
            "num_cells": adaptiver.mesh.number_of_cells(),
            "num_junctions": nj * nj,
            "width": woven_width(pressure_a, args.ell0),
            "metrics_failure": exc.__class__.__name__,
            "metrics_failure_message": str(exc),
        }
    metrics.update({
        "method": method,
        "functional": "trace-log" if method == "MetricTensorAdaptive" else "huang",
        "success": success,
        "failure_reason": failure_reason,
        "failure_message": failure_message,
        "mu": np.nan if method == "MetricTensorAdaptive" else mu,
        "nj": nj,
        "pressure_lambda": pressure_a,
        "pressure_A": pressure_a,
        "ell0": args.ell0,
        "gamma": args.gamma,
        "tau": args.tau,
        "stage_time": args.stage_time,
        "total_steps": args.total_steps,
        "target_time": args.stage_time * args.total_steps,
        "integrator": args.integrator,
        "nx": args.nx,
        "diagonal": args.diagonal,
    })
    metrics.update(bdf_stats(adaptiver, args))
    metrics.update(compression_stats(adaptiver))
    if success and not bool(metrics.get("passes_positive_area", True)):
        success = False
        failure_reason = "physical_mesh_inverted"
        failure_message = "min_cell_area <= 0 after mesh redistribution"
    if success and not bool(metrics.get("path_positive_Ehat", True)):
        success = False
        failure_reason = "logic_mesh_inverted"
        failure_message = "det(E_hat) <= 0 along accepted redistribution path"
    metrics["success"] = success
    metrics["failure_reason"] = failure_reason
    metrics["failure_message"] = failure_message

    if args.output_vtu and success:
        z = metric_volume_ratio(adaptiver.mesh, adaptiver.M)
        adaptiver.mesh.celldata["z_metric_volume"] = z
        name = f"woven_{method}_nx{args.nx}_nj{nj}_A{pressure_a:g}"
        if method == "EAGAdaptiveHuang":
            name += f"_mu{mu:g}"
        adaptiver.mesh.to_vtk(str(args.output_dir / f"{name}.vtu"))
    return metrics


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "method", "functional", "success", "failure_reason", "failure_message",
        "mu", "nj", "num_junctions", "pressure_A", "width",
        "pressure_lambda", "ell0", "gamma", "tau", "stage_time", "total_steps",
        "target_time", "integrator", "nx", "num_cells", "diagonal",
        "bdf_max_steps", "bdf_total_step_count", "bdf_total_accepted_count",
        "bdf_total_rejected_count", "bdf_total_nonfinite_error_count",
        "bdf_total_nonlinear_iteration_count",
        "bdf_mean_nonlinear_iteration_per_step",
        "bdf_stage_count", "bdf_last_step_count", "bdf_last_accepted_count",
        "bdf_last_rejected_count", "bdf_last_nonfinite_error_count",
        "bdf_last_nonlinear_iteration_count",
        "bdf_last_max_nonlinear_iteration_count",
        "bdf_last_total_time", "bdf_last_h", "bdf_last_min_h", "bdf_last_scaled_error",
        "smw_factor_count", "smw_symbolic_factor_count", "smw_numeric_factor_count",
        "smw_rank2_updates", "smw_rank1_updates", "smw_rank_stack_updates",
        "smw_max_rank", "smw_line_fail", "smw_small_solve_fail",
        "smw_b0_solve_count", "smw_b0_solve_call_count",
        "smw_multi_rhs_solve_count", "smw_saved_w_solves", "smw_last_res",
        "smw_factor_backend",
        "accepted_state_count", "path_L_peak", "path_L_cvar95",
        "path_U_TL", "path_U_H", "path_X_H_to_TL",
        "path_delta_adm", "path_min_det_Ehat", "path_positive_Ehat",
        "last_L_peak", "last_L_cvar95", "last_U_TL", "last_U_H",
        "last_X_H_to_TL", "last_delta_adm", "last_min_det_Ehat",
        "last_core_compressed_weight_fraction", "last_core_tl_positive_weight_fraction",
        "last_core_q_max", "last_core_q_p95",
        "Q_eq", "Q_ali", "Q_geo", "min_cell_area",
        "core_S", "core_P95", "core_E_eq", "core_Q_ali", "core_count",
        "arms_Q_ali", "arms_E_eq", "arms_S", "arms_P95", "arms_count",
        "passes_core_s_tol", "passes_arms_q_tol", "passes_positive_area",
    ]
    fields = [field for field in preferred if field in fields] + [
        field for field in fields if field not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Woven junction network pressure test for trace-log and Huang metrics."
    )
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--diagonal", choices=("checkerboard", "slash", "backslash"), default="checkerboard")
    parser.add_argument("--methods", default="MetricTensorAdaptive,EAGAdaptiveHuang")
    parser.add_argument("--njs", default="2,3,4,5")
    parser.add_argument("--pressure-as", default="32,64,128")
    parser.add_argument("--ell0", type=float, default=0.8)
    parser.add_argument("--mus", default="0.1,0.3333333333333333,0.45")
    parser.add_argument("--ref-n", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--t-max", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mol-times", type=int, default=4)
    parser.add_argument("--integrator", default="BDF_SMW")
    parser.add_argument("--total-steps", type=int, default=10)
    parser.add_argument("--stage-time", type=float, default=0.1)
    parser.add_argument("--stage-substeps", type=int, default=10)
    parser.add_argument(
        "--bdf-max-steps",
        type=int,
        default=5000,
        help="Maximum BDF outer attempts per mesh redistribution stage; <=0 disables the limit.",
    )
    parser.add_argument("--s-core-tol", type=float, default=-1.0)
    parser.add_argument("--q-arms-tol", type=float, default=-1.0)
    parser.add_argument("--output-vtu", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/mmesh/woven_junction_network"),
    )
    parser.add_argument("--csv-name", default="summary.csv")
    args = parser.parse_args()

    methods = parse_list(args.methods, str)
    njs = parse_list(args.njs, int)
    pressure_as = parse_list(args.pressure_as, float)
    mus = parse_list(args.mus, float)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for nj in njs:
        for pressure_a in pressure_as:
            for method in methods:
                method_mus = mus if method == "EAGAdaptiveHuang" else [np.nan]
                for mu in method_mus:
                    rows.append(run_case(args, method, nj, pressure_a, mu))
                    print(rows[-1])

    csv_path = args.output_dir / args.csv_name
    write_csv(csv_path, rows)
    print(f"Wrote summary to {csv_path}")


if __name__ == "__main__":
    main()
