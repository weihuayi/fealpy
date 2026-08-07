import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fealpy.backend import bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.mesh import TriangleMesh
from fealpy.mmesh import Config, MMesher, MeshQuality
from fealpy.mmesh.tool import Poissondata


CASE_DATA = {
    3: {
        "name": "Sinshape",
        "example": "Example 5.1",
        "u": "tanh( -100 * ( y - 0.5 - 0.25*sin(2*pi*x) ) )",
    },
    4: {
        "name": "Xshape",
        "example": "Example 5.2",
        "u": "tanh(100*(1-x-y)) - tanh(100*(x-y))",
    },
}

FUNCTIONALS = {
    "ours": {
        "label": "Trace-log",
        "method": "MetricTensorAdaptive",
        "gamma": 1.25,
        "tau": 0.004,
        "suffix": "ours",
    },
    "huang": {
        "label": "Huang",
        "method": "EAGAdaptiveHuang",
        "gamma": 1.5,
        "tau": 0.01,
        "mu": 1.0 / 3.0,
        "suffix": "huang",
    },
}

FUNCTIONAL_ORDER = {"Huang": 0, "Trace-log": 1, "Ours": 1}
FUNCTIONAL_STYLE = {
    "Trace-log": {
        "color": "#111111",
        "linecolor": "#111111",
        "edgecolor": "#111111",
        "hatch": "",
        "linestyle": "-",
        "marker": "o",
    },
    "Ours": {
        "color": "#111111",
        "linecolor": "#111111",
        "edgecolor": "#111111",
        "hatch": "",
        "linestyle": "-",
        "marker": "o",
    },
    "Huang": {
        "color": "#B8B8B8",
        "linecolor": "#4A4A4A",
        "edgecolor": "#555555",
        "hatch": "////",
        "linestyle": "--",
        "marker": "s",
    },
}


def display_functional(name):
    return "Trace-log" if name == "Ours" else name


def to_numpy(value):
    try:
        return np.asarray(bm.to_numpy(value))
    except Exception:
        return np.asarray(value)


def parse_csv_ints(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_csv_names(text):
    return [part.strip().lower() for part in text.split(",") if part.strip()]


def parse_case_total_steps(text, default_steps):
    mapping = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        case_text, steps_text = part.split(":", 1)
        mapping[int(case_text.strip())] = int(steps_text.strip())
    return mapping or {"default": default_steps}


def parse_case_floats(text):
    mapping = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        case_text, value_text = part.split(":", 1)
        mapping[int(case_text.strip())] = float(value_text.strip())
    return mapping


def make_pde(case):
    return Poissondata(CASE_DATA[case]["u"], ["x", "y"])


def metric_anisotropy(M):
    eig = np.linalg.eigvalsh(to_numpy(M))
    eig = np.maximum(eig, 1e-300)
    ratio = np.sqrt(eig[:, -1] / eig[:, 0])
    return {
        "anisotropy_mean": float(np.mean(ratio)),
        "anisotropy_p95": float(np.percentile(ratio, 95)),
        "anisotropy_max": float(np.max(ratio)),
    }


def quality_metric_arrays(mesh):
    return {
        "Q_eq": 1.0 / np.asarray(mesh.celldata["Q_eq_K"]).ravel(),
        "Q_ali": 1.0 / np.asarray(mesh.celldata["Q_ali_K"]).ravel(),
        "Q_geo": np.asarray(mesh.celldata["Q_geo_K"]).ravel(),
    }


def rms(value):
    value = np.asarray(value)
    value = value[np.isfinite(value)]
    return float(np.sqrt(np.mean(value * value))) if value.size else np.nan


def quality_arrays_from_row(row):
    with np.load(row["quality_npz"]) as data:
        arrays = {key: np.asarray(data[key], dtype=float) for key in ("Q_eq", "Q_ali", "Q_geo")}

    # Older result files stored Q_eq and Q_ali as visual inverse quantities.
    # Choose the orientation whose RMS matches the table/global metric.
    for key, csv_key in [("Q_eq", "Q_eq"), ("Q_ali", "Q_ali")]:
        target = float(row[csv_key])
        arr = arrays[key]
        inv = np.divide(1.0, arr, out=np.full_like(arr, np.nan), where=arr != 0.0)
        if abs(rms(inv) - target) < abs(rms(arr) - target):
            arrays[key] = inv
    return arrays


def run_one(args, case, nx, functional_key):
    fcfg = dict(FUNCTIONALS[functional_key])
    if functional_key == "ours" and args.ours_tau is not None:
        fcfg["tau"] = args.ours_tau
    outer_steps = args.case_total_steps.get(case, args.total_steps)
    stage_t_max = args.case_t_max.get(case, args.final_time / outer_steps)
    pde = make_pde(case)
    mesh = TriangleMesh.from_box_cross_mesh([0, 1, 0, 1], nx=nx, ny=nx)
    vertices = bm.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=bm.float64)
    mesh.meshdata["vertices"] = vertices

    config = Config()
    config.t_max = stage_t_max
    config.pde = pde
    config.active_method = fcfg["method"]
    config.mol_times = args.mol_times
    config.is_pre = False
    config.tau = fcfg["tau"]
    config.gamma = fcfg["gamma"]

    space = LagrangeFESpace(mesh, p=1)
    uh = space.interpolate(pde.solution)
    error0 = float(mesh.error(uh, pde.solution))

    setup_start = time.perf_counter()
    mm = MMesher(mesh=mesh, uh=uh, space=space, beta=args.beta, config=config)
    mm.initialize()
    mm.set_interpolation_method("solution")
    mm.set_monitor(args.monitor)
    mm.set_mol_method(args.mol_method)
    adaptiver = mm.instance
    setup_time = time.perf_counter() - setup_start

    adapt_start = time.perf_counter()
    ret = adaptiver.mesh_redistributor(
        total_steps=outer_steps,
        h=fcfg["tau"],
        method=args.integrator,
        return_info=True,
        return_timemesh=False,
    )
    adapt_time = time.perf_counter() - adapt_start

    mesh.node = ret["X"]
    uh = space.interpolate(pde.solution)
    error1 = float(mesh.error(uh, pde.solution))

    logic_mesh = adaptiver.logic_mesh
    M = adaptiver.M
    mq = MeshQuality(mesh, logic_mesh, M)
    q_eq = float(mq.Q_eq())
    q_ali = float(mq.Q_ali())
    q_geo = float(mq.Q_geo())
    cm = to_numpy(mesh.entity_measure("cell"))
    aniso = metric_anisotropy(M)

    I_h, I_t, cm_min = ret["info"]
    history = {
        "case": case,
        "case_name": CASE_DATA[case]["name"],
        "functional": fcfg["label"],
        "method": fcfg["method"],
        "nx": nx,
        "ny": nx,
        "NC": int(mesh.number_of_cells()),
        "tau": float(fcfg["tau"]),
        "gamma": float(fcfg["gamma"]),
        "mu": float(fcfg.get("mu", np.nan)),
        "integrator": args.integrator,
        "outer_steps": int(outer_steps),
        "stage_t_max": float(stage_t_max),
        "final_time": float(stage_t_max * outer_steps),
        "I_h": [float(x) for x in I_h],
        "delta_I_h": [float(x) for x in to_numpy(I_t).ravel()],
        "min_cell_volume": [float(x) for x in cm_min],
    }

    integrator_suffix = "" if args.integrator == "BDF_SMW" else f"_{args.integrator.lower()}"
    stem = f"{CASE_DATA[case]['name']}_case{case}_nx{nx}_{fcfg['suffix']}{integrator_suffix}"
    history_json = args.out_dir / f"history_{stem}.json"
    history_csv = args.out_dir / f"history_{stem}.csv"
    history_json.write_text(json.dumps(history, indent=2), encoding="utf-8")
    with history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["outer_step", "I_h", "delta_I_h", "min_cell_volume"])
        for i, val in enumerate(history["I_h"]):
            dI = history["delta_I_h"][i - 1] if i > 0 else np.nan
            writer.writerow([i + 1, val, dI, history["min_cell_volume"][i]])

    if args.write_vtu:
        mesh.to_vtk(str(args.out_dir / f"mesh_{stem}.vtu"))

    quality_png = ""
    quality_npz = ""
    if args.quality_size == nx:
        quality_png = str(args.out_dir / f"quality_{stem}.png")
        plot_quality_histograms(mesh, quality_png)
        quality_npz = str(args.out_dir / f"quality_{stem}.npz")
        np.savez_compressed(args.out_dir / f"quality_{stem}.npz", **quality_metric_arrays(mesh))

    row = {
        "case": case,
        "case_name": CASE_DATA[case]["name"],
        "functional": fcfg["label"],
        "method": fcfg["method"],
        "nx": nx,
        "ny": nx,
        "NC": int(mesh.number_of_cells()),
        "tau": float(fcfg["tau"]),
        "gamma": float(fcfg["gamma"]),
        "mu": float(fcfg.get("mu", np.nan)),
        "integrator": args.integrator,
        "outer_steps": int(outer_steps),
        "stage_t_max": float(stage_t_max),
        "final_time": float(stage_t_max * outer_steps),
        "Q_eq": q_eq,
        "Q_ali": q_ali,
        "Q_geo": q_geo,
        "e_L2_before": error0,
        "e_L2": error1,
        "min_cell_volume": float(np.min(cm)),
        "time_setup_s": setup_time,
        "time_adapt_s": adapt_time,
        "time_total_s": setup_time + adapt_time,
        "bdf_steps": int(getattr(adaptiver, "_bdf_total_step_count", 0)),
        "accepted_updates": int(getattr(adaptiver, "_bdf_total_accepted_count", 0)),
        "rejected_updates": int(getattr(adaptiver, "_bdf_total_rejected_count", 0)),
        "nonlinear_iterations": int(
            getattr(adaptiver, "_bdf_total_nonlinear_iteration_count", 0)
        ),
        "nonlinear_max_per_update": int(
            getattr(adaptiver, "_last_bdf_max_nonlinear_iteration_count", 0)
        ),
        "smw_symbolic_factor_count": int(
            getattr(adaptiver, "_smw_stats", {}).get("symbolic_factor_count", 0)
        ),
        "smw_numeric_factor_count": int(
            getattr(adaptiver, "_smw_stats", {}).get("numeric_factor_count", 0)
        ),
        "history_json": str(history_json),
        "history_csv": str(history_csv),
        "quality_png": quality_png,
        "quality_npz": quality_npz,
    }
    row.update(aniso)
    return row


def plot_quality_histograms(mesh, savefig):
    metrics = {
        r"$Q_{\mathrm{eq}}$": 1.0 / np.asarray(mesh.celldata["Q_eq_K"]).ravel(),
        r"$Q_{\mathrm{ali}}$": 1.0 / np.asarray(mesh.celldata["Q_ali_K"]).ravel(),
        r"$Q_{\mathrm{geo}}$": np.asarray(mesh.celldata["Q_geo_K"]).ravel(),
    }
    global_values = {
        r"$Q_{\mathrm{eq}}$": float(np.sqrt(np.mean(metrics[r"$Q_{\mathrm{eq}}$"] ** 2))),
        r"$Q_{\mathrm{ali}}$": float(np.sqrt(np.mean(metrics[r"$Q_{\mathrm{ali}}$"] ** 2))),
        r"$Q_{\mathrm{geo}}$": float(np.sqrt(np.mean(metrics[r"$Q_{\mathrm{geo}}$"] ** 2))),
    }
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.35), dpi=220, constrained_layout=True)
    for ax, (title, data), color in zip(axes, metrics.items(), colors):
        data = data[np.isfinite(data)]
        ax.hist(data, bins=50, color=color, edgecolor="#333333", linewidth=0.45, alpha=0.72)
        if data.size:
            mean_v = float(np.mean(data))
            median_v = float(np.median(data))
            std_v = float(np.std(data))
            min_v = float(np.min(data))
            max_v = float(np.max(data))
            p5, p95 = np.percentile(data, [5, 95])
            ax.axvline(mean_v, color="#111111", linestyle="-", linewidth=1.15, label="mean")
            ax.axvline(median_v, color="#555555", linestyle="--", linewidth=1.05, label="median")
            ax.axvline(p5, color="#777777", linestyle=":", linewidth=0.95, label="p5/p95")
            ax.axvline(p95, color="#777777", linestyle=":", linewidth=0.95)
            text = (
                f"n={data.size}\n"
                f"min={min_v:.3f}\n"
                f"max={max_v:.3f}\n"
                f"std={std_v:.3g}\n"
                f"global={global_values[title]:.3f}"
            )
            ax.annotate(
                text,
                xy=(0.97, 0.95),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=8.2,
                bbox={
                    "boxstyle": "round,pad=0.32",
                    "fc": "white",
                    "ec": "#777777",
                    "alpha": 0.92,
                    "linewidth": 0.55,
                },
            )
            ax.legend(frameon=False, loc="best", fontsize=8.2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(savefig, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_history(rows, out_dir):
    by_key = {}
    for row in rows:
        if not Path(row["history_json"]).exists():
            continue
        key = (row["case"], row["nx"])
        by_key.setdefault(key, []).append(row)

    for (case, nx), case_rows in sorted(by_key.items()):
        if len(case_rows) < 2:
            continue
        histories = []
        for row in case_rows:
            with open(row["history_json"], encoding="utf-8") as f:
                histories.append((row, json.load(f)))

        for field, ylabel, out_tag in [
            ("I_h", r"Discrete functional value", "energy"),
            ("min_cell_volume", r"Minimum cell volume", "min_cell_volume"),
        ]:
            fig, ax = plt.subplots(figsize=(4.8, 3.35), dpi=220, constrained_layout=True)
            for row, hist in histories:
                x = np.arange(1, len(hist[field]) + 1)
                functional = display_functional(row["functional"])
                style = FUNCTIONAL_STYLE.get(
                    row["functional"],
                    {"color": "#444444", "linestyle": "-", "marker": "o"},
                )
                ax.plot(
                    x,
                    hist[field],
                    color=style.get("linecolor", style["color"]),
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    ms=4.0,
                    lw=1.25,
                    markerfacecolor="white",
                    markeredgecolor=style.get("linecolor", style["color"]),
                    markeredgewidth=0.9,
                    label=functional,
                )
            ax.set_xlabel("outer step")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.58)
            ax.legend(frameon=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.savefig(
                out_dir / f"{out_tag}_{CASE_DATA[case]['name']}_case{case}_nx{nx}.png",
                dpi=260,
                bbox_inches="tight",
                pil_kwargs={"optimize": True},
            )
            plt.close(fig)


def plot_quality_comparisons(rows, out_dir):
    metric_labels = {
        "Q_eq": r"$Q_{\mathrm{eq}}$",
        "Q_ali": r"$Q_{\mathrm{ali}}$",
        "Q_geo": r"$Q_{\mathrm{geo}}$",
    }
    by_key = {}
    for row in rows:
        npz = row.get("quality_npz", "")
        if not npz or not Path(npz).exists():
            continue
        key = (row["case"], row["nx"])
        by_key.setdefault(key, []).append(row)

    for (case, nx), case_rows in sorted(by_key.items()):
        by_functional = {display_functional(row["functional"]): row for row in case_rows}
        if not {"Trace-log", "Huang"}.issubset(by_functional):
            continue

        loaded = {
            name: quality_arrays_from_row(by_functional[name])
            for name in ["Trace-log", "Huang"]
        }

        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), dpi=220, constrained_layout=True)
        for ax, metric in zip(axes, metric_labels):
            arrays = {
                name: values[metric][np.isfinite(values[metric])]
                for name, values in loaded.items()
            }
            finite = np.concatenate([arr for arr in arrays.values() if arr.size])
            if finite.size == 0:
                continue
            lo, hi = np.percentile(finite, [0.2, 99.8])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = float(np.min(finite)), float(np.max(finite))
            if lo == hi:
                hi = lo + 1.0
            bins = np.linspace(lo, hi, 33)
            centers = 0.5 * (bins[:-1] + bins[1:])
            width = 0.42 * (bins[1] - bins[0])

            for offset, name in [(-0.5, "Trace-log"), (0.5, "Huang")]:
                counts, _ = np.histogram(arrays[name], bins=bins)
                style = FUNCTIONAL_STYLE[name]
                ax.bar(
                    centers + offset * width,
                    counts,
                    width=width,
                    color=style["color"],
                    edgecolor=style.get("edgecolor", "#222222"),
                    linewidth=0.35,
                    alpha=0.92,
                    hatch=style.get("hatch", ""),
                    label=name,
                )

            rms = {
                name: float(np.sqrt(np.mean(arrays[name] ** 2)))
                for name in arrays
                if arrays[name].size
            }
            ax.text(
                0.03,
                0.96,
                f"RMS {rms['Trace-log']:.4f} / {rms['Huang']:.4f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.0,
            )
            ax.set_title(metric_labels[metric], fontsize=10.5)
            ax.set_xlabel("value", fontsize=8.4)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
            ax.tick_params(labelsize=7.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].set_ylabel("cell count", fontsize=8.4)
        axes[0].legend(
            loc="upper right",
            bbox_to_anchor=(1.0, 0.86),
            frameon=False,
            fontsize=7.5,
        )
        nc = int(by_functional["Trace-log"].get("NC", 2 * nx * nx))
        fig.suptitle(
            f"{CASE_DATA[case]['example']} ({CASE_DATA[case]['name']}), NC={nc}",
            fontsize=10.5,
        )
        fig.savefig(
            out_dir / f"quality_compare_{CASE_DATA[case]['name']}_case{case}_nx{nx}.png",
            dpi=260,
            bbox_inches="tight",
            pil_kwargs={"optimize": True},
        )
        plt.close(fig)


def write_metrics_csv(rows, path):
    fields = [
        "case",
        "case_name",
        "functional",
        "method",
        "nx",
        "ny",
        "NC",
        "tau",
        "gamma",
        "mu",
        "integrator",
        "outer_steps",
        "stage_t_max",
        "final_time",
        "Q_eq",
        "Q_ali",
        "Q_geo",
        "e_L2_before",
        "e_L2",
        "min_cell_volume",
        "time_setup_s",
        "time_adapt_s",
        "time_total_s",
        "bdf_steps",
        "accepted_updates",
        "rejected_updates",
        "nonlinear_iterations",
        "nonlinear_max_per_update",
        "anisotropy_mean",
        "anisotropy_p95",
        "anisotropy_max",
        "smw_symbolic_factor_count",
        "smw_numeric_factor_count",
        "history_json",
        "history_csv",
        "quality_png",
        "quality_npz",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_metrics_csv(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)
            for key in ["case", "nx", "ny", "NC"]:
                parsed[key] = int(float(parsed[key]))
            rows.append(parsed)
    return rows


def latex_table(rows, case):
    case_rows = [row for row in rows if row["case"] == case]
    order = FUNCTIONAL_ORDER
    case_rows.sort(key=lambda r: (order.get(r["functional"], 99), r["NC"]))
    lines = []
    lines.append(r"\begin{tabular}{c c c c c c c c c}")
    lines.append(r"\hline")
    lines.append(
        r"functional & size(NC) & $Q_{eq}$ & $Q_{ali}$ & $Q_{geo}$ & "
        r"$e_{L_2}$ & time(s) & nonlinear iters & $A_{95}$ \\"
    )
    lines.append(r"\hline")
    last = None
    display_names = sorted(
        {display_functional(row["functional"]) for row in case_rows},
        key=lambda name: order.get(name, 99),
    )
    counts = {
        name: sum(1 for row in case_rows if display_functional(row["functional"]) == name)
        for name in display_names
    }
    seen = {name: 0 for name in display_names}
    for row in case_rows:
        name = display_functional(row["functional"])
        if last is not None and name != last:
            lines.append(r"\hline")
        seen[name] += 1
        if counts.get(name, 1) > 1 and seen[name] == 1:
            first = rf"\multirow{{{counts[name]}}}{{*}}{{{name}}}"
        elif counts.get(name, 1) > 1:
            first = ""
        else:
            first = name
        lines.append(
            rf"{first} & {row['NC']} & {row['Q_eq']:.5f} & {row['Q_ali']:.5f} & "
            rf"{row['Q_geo']:.5f} & {row['e_L2']:.5g} & {row['time_adapt_s']:.6f} & "
            rf"{row['nonlinear_iterations']} & {row['anisotropy_p95']:.5f}\\"
        )
        last = name
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def write_latex_tables(rows, path):
    parts = []
    for case in sorted({row["case"] for row in rows}):
        parts.append(f"% {CASE_DATA[case]['example']} ({CASE_DATA[case]['name']})")
        parts.append(latex_table(rows, case))
        parts.append("")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="BDF-SMW benchmark for mmesher_with_solution.py cases 3 and 4."
    )
    parser.add_argument("--cases", default="3,4")
    parser.add_argument("--sizes", default="20,40,80", help="nx values; ny=nx.")
    parser.add_argument("--functionals", default="ours,huang", choices=None)
    parser.add_argument("--out-dir", type=Path, default=Path("example/meshopt/mmesh/results/function_cases_bdf_smw"))
    parser.add_argument("--total-steps", type=int, default=10)
    parser.add_argument(
        "--case-total-steps",
        default="3:20,4:10",
        help="Comma-separated case:outer_steps overrides.",
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=1.0,
        help="Target total pseudo-time; default gives stage_t_max=1/outer_steps.",
    )
    parser.add_argument(
        "--case-t-max",
        default="",
        help="Optional comma-separated case:stage_t_max overrides.",
    )
    parser.add_argument("--mol-times", type=int, default=6)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--monitor", default="linear_int_error")
    parser.add_argument("--mol-method", default="huangs_method")
    parser.add_argument("--quality-size", type=int, default=80)
    parser.add_argument(
        "--ours-tau",
        type=float,
        default=None,
        help="Override tau only for MetricTensorAdaptive/Ours.",
    )
    parser.add_argument(
        "--integrator",
        default="BDF_SMW",
        choices=["BDF_SMW", "BDF_LFP"],
        help="Implicit BDF nonlinear solver branch used inside mesh_redistributor.",
    )
    parser.add_argument("--write-vtu", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate history plots from out-dir/metrics.csv without rerunning cases.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one unrecorded small Ours case before timing to absorb JIT/MUMPS cold-start cost.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        rows = read_metrics_csv(args.out_dir / "metrics.csv")
        plot_history(rows, args.out_dir)
        plot_quality_comparisons(rows, args.out_dir)
        print(f"[done] regenerated history plots in {args.out_dir}")
        return

    cases = parse_csv_ints(args.cases)
    sizes = parse_csv_ints(args.sizes)
    functionals = parse_csv_names(args.functionals)
    args.case_total_steps = parse_case_total_steps(args.case_total_steps, args.total_steps)
    args.case_t_max = parse_case_floats(args.case_t_max)

    unknown = sorted(set(functionals) - set(FUNCTIONALS))
    if unknown:
        raise ValueError(f"Unknown functionals: {unknown}")

    rows = []
    if args.warmup:
        warm_case = cases[0]
        warm_nx = min(sizes)
        print(f"[warmup] case={warm_case} nx={warm_nx} functional=ours", flush=True)
        run_one(args, warm_case, warm_nx, "ours")
    for case in cases:
        if case not in CASE_DATA:
            raise ValueError(f"Only cases {sorted(CASE_DATA)} are supported, got {case}.")
        for nx in sizes:
            for functional in functionals:
                print(f"[run] case={case} nx={nx} functional={functional}", flush=True)
                rows.append(run_one(args, case, nx, functional))
                write_metrics_csv(rows, args.out_dir / "metrics.csv")
                write_latex_tables(rows, args.out_dir / "tables.tex")

    plot_history(rows, args.out_dir)
    plot_quality_comparisons(rows, args.out_dir)
    write_metrics_csv(rows, args.out_dir / "metrics.csv")
    write_latex_tables(rows, args.out_dir / "tables.tex")
    print(f"[done] wrote {args.out_dir / 'metrics.csv'}")
    print(f"[done] wrote {args.out_dir / 'tables.tex'}")


if __name__ == "__main__":
    main()
