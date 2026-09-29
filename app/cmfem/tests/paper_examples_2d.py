"""Reproduce the two-dimensional numerical examples from the paper."""

import argparse
import sys
from math import factorial
from pathlib import Path

import sympy as sp
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh, spsolve
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cm_fe_space2d import CmFESpace2d
from pde import DoubleLaplacePDE, ManufacturedPDE, TripleLaplacePDE


x, y = sp.symbols('x y')


def _print_row(n, dofs, errors, previous):
    """Print one error-and-rate row for the numerical tables."""
    if previous is None:
        rates = bm.full(len(errors), float('nan'))
    else:
        rates = bm.log2(previous/errors)
    error_text = ' '.join(f'{value:.6e}' for value in errors)
    rate_text = ' '.join('--' if bm.isnan(rate) else f'{rate:.3f}' for rate in rates)
    row = f'n={n:<2d} DoF={dofs:<6d} errors=[{error_text}] '
    row += f'rates=[{rate_text}]'
    print(row)


def _local_condition(space):
    """Compute the local-matrix data reported in Table 1."""
    D = space.D.to_dense()[0]
    orders = []
    for faces in space.lattice.simplex.subsimplices:
        for f in faces:
            for s, rows in enumerate(space.lattice.layers[f]):
                orders.extend([s]*len(rows))
    scale = bm.array([factorial(space.p)//factorial(space.p-s)
                      for s in orders])
    raw = scale[:, None]*D
    nnz = int(bm.sum(bm.abs(raw) > 1.0e-10))
    return D.shape[0], nnz, bm.linalg.cond(raw), bm.linalg.cond(D), bm.max(scale)


def local_conditioning_example():
    """Reproduce the three two-dimensional entries in Table 1."""
    node = bm.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    cell = bm.array(((0, 1, 2),), dtype=bm.int32)
    print('\nLocal DoF--Bernstein matrix: 2D')
    for m, p, r in ((1, 5, (2, 1, 0)),
                    (1, 7, (2, 1, 0)),
                    (2, 9, (4, 2, 0))):
        space = CmFESpace2d(TriangleMesh(node, cell), p, r)
        size, nnz, raw_cond, cond, ratio = _local_condition(space)
        print(f'(2,{m},{p}) N={size:<3d} nnz={nnz:<3d} '
              f'cond_raw={raw_cond:.4e} cond={cond:.4e} ratio={ratio}')


def interpolation_examples():
    """Reproduce the 2D C1 and C2 interpolation tests in Tables 2 and 3."""
    expression = sp.sin(4*x)*sp.cos(5*y)
    for p, r, m in [(7, (2, 1, 0), 1), (9, (4, 2, 0), 2)]:
        exact = ManufacturedPDE(expression, 2).derivatives(r[0])
        previous = None
        print(f'\nInterpolation: k={p}, m={m}')
        for n in (1, 2, 4, 8):
            mesh = TriangleMesh.from_box(nx=n, ny=n)
            space = CmFESpace2d(mesh, p, r)
            uh = space.interpolate(exact)
            errors = space.error_norms(uh, exact, m+1)
            _print_row(n, space.number_of_global_dofs(), errors, previous)
            previous = errors


def _polyharmonic_example(m, p, r, expression, ns):
    """Solve the two-dimensional polyharmonic problems in Tables 5 and 6."""
    pde_type = {1: DoubleLaplacePDE, 2: TripleLaplacePDE}[m]
    pde = pde_type(expression, dimension=2)
    exact = pde.derivatives(max(r[0], m+1))
    source = pde.source

    previous = None
    print(f'\nPolyharmonic: k={p}, m={m}')
    for n in ns:
        mesh = TriangleMesh.from_box(nx=n, ny=n)
        space = CmFESpace2d(mesh, p, r)
        matrix = space.stiffness_matrix().to_scipy()
        vector = space.source_vector(source, q=p+4)
        lift, boundary = space.boundary_interpolate(exact)
        free = ~boundary
        solution = lift.copy()
        free_matrix = matrix[free][:, free]
        right = vector[free]-(matrix@lift)[free]
        solution[free] = spsolve(free_matrix, right)
        errors = space.error_norms(solution, exact, m+1)
        _print_row(n, space.number_of_global_dofs(), errors, previous)
        previous = errors


def biharmonic_example():
    """Reproduce the two-dimensional C1 biharmonic test in Table 5."""
    expression = (sp.sin(2*sp.pi*x)*sp.sin(2*sp.pi*y))**2
    ns = (4, 8, 16, 32)
    _polyharmonic_example(1, 5, (2, 1, 0), expression, ns)


def triharmonic_example():
    """Reproduce the two-dimensional C2 triharmonic test in Table 6."""
    expression = sp.sin(2*sp.pi*x)*sp.sin(2*sp.pi*y)
    ns = (1, 2, 4, 8)
    _polyharmonic_example(2, 9, (4, 2, 0), expression, ns)


def conditioning_example():
    """Reproduce the 2D Jacobi-scaled stiffness conditions in Figure 4."""
    mesh = TriangleMesh.from_box(nx=2, ny=2)
    cases = ((1, range(5, 23), (2, 1, 0)),
             (2, range(9, 23), (4, 2, 0)),
             (3, range(13, 23), (6, 3, 0)))
    for m, degrees, r in cases:
        print(f'\nJacobi-scaled condition numbers: m={m}')
        for p in degrees:
            space = CmFESpace2d(mesh, p, r)
            matrix = space.stiffness_matrix().to_scipy()
            free = ~space.is_boundary_dof()
            matrix = matrix[free][:, free]
            scale = 1/bm.sqrt(matrix.diagonal())
            matrix = diags(scale)@matrix@diags(scale)
            largest = eigsh(matrix, k=1, which='LM', return_eigenvectors=False)[0]
            smallest = eigsh(matrix, k=1, sigma=0, which='LM',
                             return_eigenvectors=False)[0]
            print(f'k={p:<2d} free_DoF={matrix.shape[0]:<4d} kappa={largest/smallest:.6e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    choices = ('all', 'interpolation', 'biharmonic',
               'triharmonic', 'conditioning', 'local-conditioning')
    parser.add_argument('example', nargs='?', default='all', choices=choices)
    args = parser.parse_args()
    choice = args.example
    print('Error norm: Frobenius')
    examples = {
        'interpolation': interpolation_examples,
        'biharmonic': biharmonic_example,
        'triharmonic': triharmonic_example,
        'conditioning': conditioning_example,
        'local-conditioning': local_conditioning_example,
    }
    if choice == 'all':
        interpolation_examples()
        biharmonic_example()
        triharmonic_example()
        conditioning_example()
        local_conditioning_example()
    else:
        examples[choice]()
