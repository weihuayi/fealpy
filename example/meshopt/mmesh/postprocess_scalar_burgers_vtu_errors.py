import argparse
import re
from pathlib import Path

import meshio

from fealpy.backend import bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.mesh import TriangleMesh
from fealpy.mmesh.pde.scalar_burgers_data import ScalarBurgersData


RE = 100
U_EXPR = f"1/(1+ exp((x+y-t)/({1 / RE})))"
VAR = ["x", "y", "t"]
DOMAIN = [0, 1, 0, 1]
TIME_SPAN = [0, 2]


def _step_number(path: Path) -> int:
    match = re.search(r"step(\d+)\.vtu$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse step number from {path.name!r}")
    return int(match.group(1))


def _load_triangle_solution(path: Path):
    data = meshio.read(path)
    if "triangle" not in data.cells_dict:
        raise ValueError(f"{path} does not contain triangle cells")
    if "u" not in data.point_data:
        raise ValueError(f"{path} does not contain point data named 'u'")

    node = bm.asarray(data.points[:, :2], dtype=bm.float64)
    cell = bm.asarray(data.cells_dict["triangle"], dtype=bm.int32)
    uh_data = bm.asarray(data.point_data["u"], dtype=bm.float64)
    mesh = TriangleMesh(node, cell)
    space = LagrangeFESpace(mesh, p=1)
    uh = space.function()
    uh[:] = uh_data
    return mesh, uh


def compute_errors(vtu_dir: Path, pattern: str, dt: float):
    pde = ScalarBurgersData(U_EXPR, VAR, DOMAIN, TIME_SPAN, Re=RE)
    files = sorted(vtu_dir.glob(pattern), key=_step_number)
    if not files:
        raise FileNotFoundError(f"No VTU files matching {pattern!r} under {vtu_dir}")

    rows = []
    for i, path in enumerate(files, start=1):
        step = _step_number(path)
        t = step * dt
        mesh, uh = _load_triangle_solution(path)
        l2_error = mesh.error(uh, lambda p, t=t: pde.solution(p, t), power=2)
        h1_error = mesh.error(uh.grad_value, lambda p, t=t: pde.gradient(p, t), power=2)
        rows.append((step, t, float(l2_error), float(h1_error), path.name))
        if i == 1 or i == len(files) or i % 50 == 0:
            print(
                f"[{i:4d}/{len(files)}] step={step:04d}, "
                f"t={t:.6f}, L2={float(l2_error):.6e}, H1={float(h1_error):.6e}"
            )
    return rows


def write_report(rows, output: Path, vtu_dir: Path, pattern: str, dt: float):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write("# Scalar Burgers VTU error history\n")
        f.write(f"# vtu_dir: {vtu_dir}\n")
        f.write(f"# pattern: {pattern}\n")
        f.write(f"# dt: {dt:.16g}\n")
        f.write("# columns: step time L2_error H1_seminorm_error file\n")
        for step, t, l2_error, h1_error, name in rows:
            f.write(f"{step:6d} {t:.16e} {l2_error:.16e} {h1_error:.16e} {name}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Read scalar Burgers VTU files and write L2/H1 error history."
    )
    parser.add_argument(
        "--vtu-dir",
        type=Path,
        default=Path("experiments/results/mmesh/scalar_burgers/huang_g15_tau001_linear_int_error"),
    )
    parser.add_argument("--pattern", default="burgers_solution_step*.vtu")
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <vtu-dir>/burgers_vtu_error_history.txt",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = args.vtu_dir / "burgers_vtu_error_history.txt"

    rows = compute_errors(args.vtu_dir, args.pattern, args.dt)
    write_report(rows, output, args.vtu_dir, args.pattern, args.dt)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
