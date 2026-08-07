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


def parse_density_pairs(text):
    pairs = []
    for item in text.split(";"):
        if not item.strip():
            continue
        left, right = item.split(",")
        pairs.append((float(left), float(right)))
    return pairs


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


def _outer_metric(rho, q, e1, e2, kappa):
    lam1 = kappa ** (-0.5 * q)
    lam2 = kappa ** (0.5 * q)
    M = np.zeros(rho.shape + (2, 2), dtype=np.float64)
    M[..., 0, 0] = rho * (lam1 * e1[..., 0] * e1[..., 0] + lam2 * e2[..., 0] * e2[..., 0])
    M[..., 0, 1] = rho * (lam1 * e1[..., 0] * e1[..., 1] + lam2 * e2[..., 0] * e2[..., 1])
    M[..., 1, 0] = M[..., 0, 1]
    M[..., 1, 1] = rho * (lam1 * e1[..., 1] * e1[..., 1] + lam2 * e2[..., 1] * e2[..., 1])
    return M


def metric_a_raw(
    points,
    *,
    phi,
    kappa,
    a=16.0,
    ell1=0.30,
    ell2=0.055,
    winding=0.0,
    core=0.08,
):
    x = points[..., 0] - 0.5
    y = points[..., 1] - 0.5
    c = np.cos(phi)
    s = np.sin(phi)
    y1 = c * x + s * y
    y2 = -s * x + c * y
    xi1 = y1 / ell1
    xi2 = y2 / ell2
    r2 = xi1 * xi1 + xi2 * xi2
    g = np.exp(-r2)
    rho = 1.0 + a * g
    if winding == 0.0:
        q = g
        beta = phi
    else:
        # The anisotropy is faded out in a tiny core, so the rotating line field
        # does not create a hard direction singularity at the ellipse center.
        ramp = r2 / (r2 + core * core)
        q = g * ramp
        beta = phi + winding * np.arctan2(xi2, xi1)

    e1 = np.zeros(points.shape, dtype=np.float64)
    e2 = np.zeros(points.shape, dtype=np.float64)
    cb = np.cos(beta)
    sb = np.sin(beta)
    e1[..., 0] = cb
    e1[..., 1] = sb
    e2[..., 0] = -sb
    e2[..., 1] = cb
    return _outer_metric(rho, q, e1, e2, kappa), {"A": g >= 0.10}


def metric_b_raw(points, *, kappa, a=18.0, r0=0.25, eps=0.035):
    dx = points[..., 0] - 0.5
    dy = points[..., 1] - 0.5
    r = np.sqrt(dx * dx + dy * dy)
    q = np.exp(-(((r - r0) / eps) ** 2))
    rho = 1.0 + a * q
    inv_r = 1.0 / np.maximum(r, 1.0e-14)
    n = np.stack([dx * inv_r, dy * inv_r], axis=-1)
    t = np.stack([-n[..., 1], n[..., 0]], axis=-1)
    near_center = r <= 1.0e-14
    if np.any(near_center):
        n[near_center] = np.array([1.0, 0.0])
        t[near_center] = np.array([0.0, 1.0])
    return _outer_metric(rho, q, t, n, kappa), {"B": np.abs(r - r0) <= 2.0 * eps}


def metric_c_raw(points, *, kappa, a_l=18.0, a_r=24.0):
    x = points[..., 0]
    y = points[..., 1]
    g_l = np.exp(-(((x - 0.28) / 0.24) ** 2 + ((y - 0.50) / 0.035) ** 2))
    g_r = np.exp(-(((x - 0.74) ** 2 + (y - 0.50) ** 2) / (0.075 ** 2)))
    rho = 1.0 + a_l * g_l + a_r * g_r
    q = g_l
    e1 = np.zeros(points.shape, dtype=np.float64)
    e2 = np.zeros(points.shape, dtype=np.float64)
    e1[..., 0] = 1.0
    e2[..., 1] = 1.0
    return _outer_metric(rho, q, e1, e2, kappa), {"L": g_l >= 0.10, "R": g_r >= 0.10}


def raw_metric(points, args, example, kappa, phi, density_pair):
    if example == "A":
        return metric_a_raw(
            points,
            phi=phi,
            kappa=kappa,
            a=args.ellipse_a,
            winding=args.ellipse_winding,
            core=args.ellipse_core,
        )
    if example == "B":
        return metric_b_raw(points, kappa=kappa, a=args.ring_a, r0=args.r0, eps=args.ring_eps)
    a_l, a_r = density_pair
    return metric_c_raw(points, kappa=kappa, a_l=a_l, a_r=a_r)


def reference_integrals(args, example, kappa, phi, density_pair):
    n = args.ref_n
    grid = (np.arange(n, dtype=np.float64) + 0.5) / n
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    M_raw, patches = raw_metric(points, args, example, kappa, phi, density_pair)
    rho_raw = np.sqrt(np.linalg.det(M_raw))
    sigma_raw = float(np.mean(rho_raw))
    scale = 1.0 / sigma_raw
    patch_targets = {}
    for name, mask in patches.items():
        patch_targets[name] = float(np.mean(rho_raw[mask]) * np.mean(mask) / sigma_raw)
    return scale, patch_targets


def install_metric_monitor(adaptiver, *, args, example, kappa, phi, density_pair, scale, patch_targets):
    key = "_functional_discriminating_metric_registered"
    if not getattr(adaptiver.__class__, key, False):
        @adaptiver.__class__.monitor.register("functional_discriminating_metric")
        def monitor(self):
            node = np.asarray(bm.to_numpy(self.mesh.node))
            M_node_raw, _ = raw_metric(
                node,
                self._fd_args,
                self._fd_example,
                self._fd_kappa,
                self._fd_phi,
                self._fd_density_pair,
            )
            M_node = self._fd_scale * M_node_raw
            self.M_node = bm.asarray(M_node, dtype=bm.float64)
            self.M = bm.mean(self.M_node[self.cell], axis=1)
            self._fd_sigma = 1.0

        setattr(adaptiver.__class__, key, True)

    adaptiver._fd_args = args
    adaptiver._fd_example = example
    adaptiver._fd_kappa = kappa
    adaptiver._fd_phi = phi
    adaptiver._fd_density_pair = density_pair
    adaptiver._fd_scale = scale
    adaptiver._fd_patch_targets = patch_targets
    adaptiver.monitor.set("functional_discriminating_metric")


def install_identity_interpolation(adaptiver):
    key = "_fd_identity_interpolation_registered"
    if not getattr(adaptiver.__class__, key, False):
        @adaptiver.__class__.interpolate.register("identity")
        def interpolate(self, moved_node):
            return self.uh
        setattr(adaptiver.__class__, key, True)
    adaptiver.interpolate.set("identity")


def install_fixed_boundary(adaptiver):
    bd = np.asarray(bm.to_numpy(adaptiver.mesh.boundary_node_flag()), dtype=bool)
    mask = np.ones(2 * adaptiver.NN, dtype=np.float64)
    idx = np.flatnonzero(bd)
    mask[idx] = 0.0
    mask[adaptiver.NN + idx] = 0.0
    projector = sp.diags(mask, format="csr")
    raw_vector = adaptiver.vector_construction
    raw_jac = adaptiver.JAC_functional

    def fixed_vector(self, A, g, trA, E_hat, *args, **kwargs):
        ret = raw_vector(A, g, trA, E_hat, *args, **kwargs)
        if isinstance(ret, tuple):
            v, local = ret
            v = bm.set_at(v, bm.asarray(idx), 0.0)
            return v, local
        v = ret
        v = bm.set_at(v, bm.asarray(idx), 0.0)
        return v

    def fixed_jac(*pargs, **kwargs):
        return projector @ raw_jac(*pargs, **kwargs).tocsr()

    adaptiver.vector_construction = types.MethodType(fixed_vector, adaptiver)
    adaptiver.JAC_functional = fixed_jac
    adaptiver.theta = types.MethodType(lambda self, M: 1.0, adaptiver)


def install_huang_mu(adaptiver, mu):
    adaptiver._fd_mu = float(mu)

    def I_func(self, trA, rho, g):
        d = self.GD
        gamma = self.gamma
        mu_ = self._fd_mu
        return bm.sum(self.cm * rho * (
            mu_ * trA ** (d * gamma / 2)
            + d ** (d * gamma / 2) * (1.0 - 2.0 * mu_) * g ** (gamma / 2)
        ))

    def TdA(self, trA):
        d = self.GD
        gamma = self.gamma
        mu_ = self._fd_mu
        return (mu_ * (d * gamma / 2) * trA ** (d * gamma / 2 - 1))[..., None, None] * self.I_p

    def Tdg(self, g):
        d = self.GD
        gamma = self.gamma
        mu_ = self._fd_mu
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


def metric_volume_ratio_ref(mesh, M):
    cm = mesh.entity_measure("cell")
    rho = bm.sqrt(bm.linalg.det(M))
    return cm * rho * mesh.number_of_cells()


def local_alignment(mesh, logic_mesh, M):
    E = edge_matrix(mesh)
    Ehat = edge_matrix(logic_mesh)
    J = E @ bm.linalg.inv(Ehat)
    A = bm.permute_dims(J, axes=(0, 2, 1)) @ M @ J
    trA = bm.trace(A, axis1=-2, axis2=-1)
    detA = bm.linalg.det(A)
    return trA / (2.0 * bm.sqrt(detA))


def collect_metrics(adaptiver, args, example, kappa, phi, density_pair):
    adaptiver.monitor()
    mesh = adaptiver.mesh
    M = adaptiver.M
    mq = MeshQuality(mesh, adaptiver.logic_mesh, M)
    z = np.asarray(bm.to_numpy(metric_volume_ratio_ref(mesh, M)))
    ali = np.asarray(bm.to_numpy(local_alignment(mesh, adaptiver.logic_mesh, M)))
    bary = np.asarray(bm.to_numpy(bm.mean(mesh.entity("node")[mesh.entity("cell")], axis=1)))
    _, patch_masks = raw_metric(bary, args, example, kappa, phi, density_pair)

    out = {
        "num_cells": mesh.number_of_cells(),
        "Q_eq": float(mq.Q_eq()),
        "Q_ali": float(mq.Q_ali()),
        "Q_geo": float(mq.Q_geo()),
        "min_cell_area": float(bm.min(mesh.entity_measure("cell"))),
    }
    targets = getattr(adaptiver, "_fd_patch_targets", {})
    alloc_terms = []
    for name, mask in patch_masks.items():
        mask = np.asarray(mask, dtype=bool)
        prefix = f"patch_{name}"
        if np.any(mask):
            z_patch = z[mask]
            ali_patch = ali[mask]
            out[f"{prefix}_S"] = float(np.max(z_patch))
            out[f"{prefix}_E_eq"] = float(np.sqrt(np.mean((z_patch - 1.0) ** 2)))
            out[f"{prefix}_P95"] = float(np.quantile(z_patch, 0.95))
            out[f"{prefix}_Q_ali"] = float(np.sqrt(np.mean(ali_patch * ali_patch)))
            out[f"{prefix}_count"] = int(np.sum(mask))
            if name in targets and targets[name] > 0.0:
                ratio = float((np.sum(mask) / mesh.number_of_cells()) / targets[name])
                out[f"{prefix}_R_N"] = ratio
                alloc_terms.append(abs(np.log(ratio)))
        else:
            out[f"{prefix}_S"] = np.nan
            out[f"{prefix}_E_eq"] = np.nan
            out[f"{prefix}_P95"] = np.nan
            out[f"{prefix}_Q_ali"] = np.nan
            out[f"{prefix}_count"] = 0
            out[f"{prefix}_R_N"] = np.nan
    out["B_alloc"] = float(max(alloc_terms)) if alloc_terms else np.nan
    return out


def run_case(args, example, method, kappa, phi, density_pair, mu):
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
    install_fixed_boundary(adaptiver)
    if method == "EAGAdaptiveHuang":
        install_huang_mu(adaptiver, mu)

    scale, patch_targets = reference_integrals(args, example, kappa, phi, density_pair)
    install_metric_monitor(
        adaptiver,
        args=args,
        example=example,
        kappa=kappa,
        phi=phi,
        density_pair=density_pair,
        scale=scale,
        patch_targets=patch_targets,
    )

    print(
        f"Running example={example} method={method} nx={args.nx} "
        f"kappa={kappa:g} phi={phi:.6g} mu={mu:.6g}"
    )
    ret = adaptiver.mesh_redistributor(method=args.integrator, return_info=False)
    adaptiver._construct(ret["X"])

    metrics = collect_metrics(adaptiver, args, example, kappa, phi, density_pair)
    a_l, a_r = density_pair
    metrics.update({
        "example": example,
        "method": method,
        "functional": "trace-log" if method == "MetricTensorAdaptive" else "huang",
        "mu": np.nan if method == "MetricTensorAdaptive" else mu,
        "kappa": kappa,
        "phi": phi,
        "a_L": a_l,
        "a_R": a_r,
        "gamma": args.gamma,
        "tau": args.tau,
        "nx": args.nx,
        "diagonal": args.diagonal,
    })

    if args.output_vtu:
        z = metric_volume_ratio_ref(adaptiver.mesh, adaptiver.M)
        adaptiver.mesh.celldata["z_metric_volume"] = z
        name = f"fdm_{example}_{method}_nx{args.nx}_kappa{kappa:g}"
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
        "example", "method", "functional", "mu", "kappa", "phi", "a_L", "a_R",
        "gamma", "tau", "nx", "num_cells", "diagonal",
        "Q_eq", "Q_ali", "Q_geo", "min_cell_area", "B_alloc",
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
        description="Functional-discriminating static metric examples on the unit square."
    )
    parser.add_argument("--examples", default="A,B,C")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--diagonal", choices=("checkerboard", "slash", "backslash"), default="checkerboard")
    parser.add_argument("--methods", default="MetricTensorAdaptive,EAGAdaptiveHuang")
    parser.add_argument("--kappas", default="256")
    parser.add_argument("--phis", default="0,0.2617993877991494,0.5235987755982988,0.7853981633974483")
    parser.add_argument("--mus", default="0.1,0.2,0.3,0.3333333333333333,0.4,0.45")
    parser.add_argument("--density-pairs", default="18,24")
    parser.add_argument("--ellipse-a", type=float, default=16.0)
    parser.add_argument("--ellipse-winding", type=float, default=0.0)
    parser.add_argument("--ellipse-core", type=float, default=0.08)
    parser.add_argument("--ring-a", type=float, default=18.0)
    parser.add_argument("--r0", type=float, default=0.25)
    parser.add_argument("--ring-eps", type=float, default=0.035)
    parser.add_argument("--ref-n", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--t-max", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mol-times", type=int, default=4)
    parser.add_argument("--integrator", default="BDF_SMW")
    parser.add_argument("--total-steps", type=int, default=20)
    parser.add_argument("--stage-time", type=float, default=0.1)
    parser.add_argument("--stage-substeps", type=int, default=10)
    parser.add_argument("--output-vtu", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/mmesh/functional_discriminating_metrics"),
    )
    parser.add_argument("--csv-name", default="summary.csv")
    args = parser.parse_args()

    examples = parse_list(args.examples, str)
    methods = parse_list(args.methods, str)
    kappas = parse_list(args.kappas, float)
    phis = parse_list(args.phis, float)
    mus = parse_list(args.mus, float)
    density_pairs = parse_density_pairs(args.density_pairs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for example in examples:
        example = example.upper()
        example_phis = phis if example == "A" else [0.0]
        example_density_pairs = density_pairs if example == "C" else [(np.nan, np.nan)]
        for kappa in kappas:
            for phi in example_phis:
                for density_pair in example_density_pairs:
                    for method in methods:
                        method_mus = mus if method == "EAGAdaptiveHuang" else [np.nan]
                        for mu in method_mus:
                            rows.append(run_case(args, example, method, kappa, phi, density_pair, mu))
                            print(rows[-1])

    csv_path = args.output_dir / args.csv_name
    write_csv(csv_path, rows)
    print(f"Wrote summary to {csv_path}")


if __name__ == "__main__":
    main()
