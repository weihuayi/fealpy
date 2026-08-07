import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import HexahedronMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.csm.material.hyperelastic_material import HyperElasticMaterial
from fealpy.csm.fem.hyperelastic_lfem_model import HyperElasticLFEMModel


OUTPUT_DIR = Path(__file__).resolve().parent / "yeoh_benchmark_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def dof(node, direction, NN):
    return direction * NN + node


def compute_D1_from_poisson(C10, nu):
    inv_D1 = 2.0 * C10 * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu))
    return 1.0 / inv_D1


def make_material(nu):
    C10 = 0.18
    C20 = -2.0e-3
    C30 = 5.0e-5
    D1 = compute_D1_from_poisson(C10, nu)

    return HyperElasticMaterial(
        C10=C10,
        C20=C20,
        C30=C30,
        D1=D1,
    ), D1


def yeoh_analytical_stress(stretch):
    """
    Incompressible Yeoh analytical engineering stress.

    lambda = stretch
    sigma = 2 * (lambda - lambda^(-2)) * dW/dI1
    """
    C10 = 0.18
    C20 = -2.0e-3
    C30 = 5.0e-5

    lam = np.asarray(stretch, dtype=float)
    I1 = lam**2 + 2.0 / lam
    x = I1 - 3.0

    dWdI1 = C10 + 2.0 * C20 * x + 3.0 * C30 * x * x

    return 2.0 * (lam - lam**(-2.0)) * dWdI1


def create_mesh_and_space():
    node = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    cell = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    mesh = HexahedronMesh(node, cell)
    scalar_space = LagrangeFESpace(mesh, p=1)
    GD = mesh.geo_dimension()
    space = TensorFunctionSpace(scalar_space, shape=(GD, -1))

    return mesh, scalar_space, space


def make_displacement_bc(scalar_space, ubar):
    ipoints = np.asarray(scalar_space.interpolation_points(), dtype=float)
    NN = scalar_space.number_of_global_dofs()
    tol = 1.0e-12

    left_nodes = []
    right_nodes = []

    for i in range(NN):
        x = ipoints[i, 0]

        if abs(x - 0.0) < tol:
            left_nodes.append(i)

        if abs(x - 1.0) < tol:
            right_nodes.append(i)

    dbc = {}

    for n in left_nodes:
        dbc[dof(n, 0, NN)] = 0.0

    for n in right_nodes:
        dbc[dof(n, 0, NN)] = ubar

    # minimal constraints to remove rigid body motion
    dbc[dof(0, 1, NN)] = 0.0
    dbc[dof(0, 2, NN)] = 0.0
    dbc[dof(3, 2, NN)] = 0.0

    return dbc, right_nodes


def run_one_case(stretch, nu, q=2, verbose=False):
    mesh, scalar_space, space = create_mesh_and_space()

    NN = scalar_space.number_of_global_dofs()
    ubar = stretch - 1.0

    material, D1 = make_material(nu)
    dbc, right_nodes = make_displacement_bc(scalar_space, ubar)

    nsteps = max(5, int(abs(ubar) / 0.05) + 1)

    if nu > 0.499:
        nsteps = max(nsteps, 40)

    model = HyperElasticLFEMModel(
        space=space,
        material=material,
        dbc=dbc,
        q=q,
    )

    converged, uh = model.solve(
        nsteps=nsteps,
        tol=1.0e-8,
        maxit=40,
        line_search=True,
        verbose=verbose,
    )

    reaction = np.asarray(model.reaction_force(), dtype=float)
    right_ux_dofs = [dof(n, 0, NN) for n in right_nodes]

    force_x = float(np.sum(reaction[right_ux_dofs]))

    A0 = 1.0
    engineering_stress = force_x / A0
    engineering_strain = stretch - 1.0

    analytical_stress = float(yeoh_analytical_stress(stretch))
    abs_error = abs(engineering_stress - analytical_stress)

    if abs(analytical_stress) > 1.0e-14:
        rel_error = abs_error / abs(analytical_stress)
    else:
        rel_error = 0.0

    F, _, _ = model.compute_F(uh)
    J = np.asarray(material.compute_J(F), dtype=float)

    return {
        "nu": nu,
        "D1": D1,
        "stretch": stretch,
        "engineering_strain": engineering_strain,
        "reaction_force": force_x,
        "engineering_stress": engineering_stress,
        "analytical_stress": analytical_stress,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "converged": converged,
        "nsteps": nsteps,
        "J_min": float(np.min(J)),
        "J_max": float(np.max(J)),
        "J_mean": float(np.mean(J)),
    }


def run_sweep():
    nus = [0.495, 0.49999]

    # Altair benchmark reproduction range
    stretch_values = np.linspace(0.5, 3.0, 51)

    results = []

    for nu in nus:
        print("\n==============================")
        print(f"Running sweep for nu = {nu}")
        print("==============================")

        for stretch in stretch_values:
            r = run_one_case(stretch=stretch, nu=nu, q=2, verbose=False)
            results.append(r)

            print(
                f"nu={nu:.5f}, "
                f"strain={r['engineering_strain']:.4f}, "
                f"stress={r['engineering_stress']:.12e}, "
                f"ana={r['analytical_stress']:.12e}, "
                f"rel_err={r['rel_error']:.3e}, "
                f"J={r['J_mean']:.12e}, "
                f"conv={r['converged']}, "
                f"nsteps={r['nsteps']}"
            )

    return results


def save_csv(results, filename):
    fieldnames = [
        "nu",
        "D1",
        "stretch",
        "engineering_strain",
        "reaction_force",
        "engineering_stress",
        "analytical_stress",
        "abs_error",
        "rel_error",
        "converged",
        "nsteps",
        "J_min",
        "J_max",
        "J_mean",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved csv to: {filename}")


def plot_stress_strain(results, filename):
    plt.figure(figsize=(8, 5.5))

    for nu in sorted(set(r["nu"] for r in results)):
        rows = sorted(
            [r for r in results if r["nu"] == nu],
            key=lambda r: r["engineering_strain"],
        )

        x = [r["engineering_strain"] for r in rows]
        y = [r["engineering_stress"] for r in rows]

        plt.plot(x, y, marker="o", label=f"FEALPy, nu = {nu}")

    strain_min = min(r["engineering_strain"] for r in results)
    strain_max = max(r["engineering_strain"] for r in results)

    strain_ref = np.linspace(strain_min, strain_max, 400)
    stretch_ref = 1.0 + strain_ref
    stress_ref = yeoh_analytical_stress(stretch_ref)

    plt.plot(
        strain_ref,
        stress_ref,
        "k--",
        linewidth=2.0,
        label="Analytical incompressible Yeoh",
    )

    plt.axhline(0.0, linewidth=0.8)
    plt.axvline(0.0, linewidth=0.8)

    plt.xlabel("Engineering strain")
    plt.ylabel("Engineering stress")
    plt.title("RD-V: 0210 Yeoh Hyperelastic Material Benchmark")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

    print(f"Saved figure to: {filename}")


def plot_J(results, filename):
    plt.figure(figsize=(8, 5.5))

    for nu in sorted(set(r["nu"] for r in results)):
        rows = sorted(
            [r for r in results if r["nu"] == nu],
            key=lambda r: r["engineering_strain"],
        )

        x = [r["engineering_strain"] for r in rows]
        y = [max(abs(r["J_mean"] - 1.0), 1.0e-12) for r in rows]

        plt.semilogy(x, y, marker="o", label=f"FEALPy, nu = {nu}")

    plt.xlabel("Engineering strain")
    plt.ylabel("|J_mean - 1|")
    plt.title("Volume change check")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

    print(f"Saved figure to: {filename}")


if __name__ == "__main__":
    results = run_sweep()

    csv_file = OUTPUT_DIR / "yeoh_altair_rd_v_0210_results.csv"
    stress_fig = OUTPUT_DIR / "yeoh_altair_rd_v_0210_stress_strain.png"
    J_fig = OUTPUT_DIR / "yeoh_altair_rd_v_0210_J_check.png"

    save_csv(results, csv_file)
    plot_stress_strain(results, stress_fig)
    plot_J(results, J_fig)