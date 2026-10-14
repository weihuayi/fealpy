"""Reproduce the three-dimensional numerical examples from the paper."""

import argparse
import sys
from math import factorial
from pathlib import Path

import sympy as sp
from mumps import spsolve as mumps_spsolve
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cm_fe_space3d import CmFESpace3d
from pde import DoubleLaplacePDE, ManufacturedPDE


x, y, z = sp.symbols('x y z')


def _print_row(n, dofs, errors, previous):
    """Print one error-and-rate row for Tables 3 and 6."""
    rates = (bm.full(len(errors), float('nan')) if previous is None
             else bm.log2(previous/errors))
    error_text = ' '.join(f'{value:.6e}' for value in errors)
    rate_text = ' '.join('--' if bm.isnan(rate) else f'{rate:.3f}' for rate in rates)
    print(f'n={n:<2d} DoF={dofs:<7d} errors=[{error_text}] rates=[{rate_text}]', flush=True)


def normalization_example():
    """Reproduce the 3D local-matrix comparison in the response to Referee 1."""
    node = bm.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                     (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)))
    cell = bm.array(((0, 1, 2, 3),), dtype=bm.int32)
    print('\nLocal DoF--Bernstein matrix: 3D')
    for p in (9, 11):
        space = CmFESpace3d(TetrahedronMesh(node, cell), p, (4, 2, 1, 0))
        D = space.D.to_dense()[0]
        orders = []
        for faces in space.lattice.simplex.subsimplices:
            for f in faces:
                for s, rows in enumerate(space.lattice.layers[f]):
                    orders.extend([s]*len(rows))
        scale = bm.array([factorial(p)//factorial(p-s) for s in orders])
        raw = scale[:, None]*D
        size = D.shape[0]
        nnz = int(bm.sum(bm.abs(raw) > 1.0e-10))
        raw_cond = bm.linalg.cond(raw)
        cond = bm.linalg.cond(D)
        ratio = bm.max(scale)
        print(f'(3,1,{p}) N={size:<3d} nnz={nnz:<4d} '
              f'cond_raw={raw_cond:.4e} cond={cond:.4e} ratio={ratio}')


def _shape_mesh(eps, family='T'):
    """Construct a tetrahedron from the families in Section 6.3."""
    cell = bm.array(((0, 1, 2, 3),), dtype=bm.int32)
    if family == 'T':
        t = eps/2**0.5
        node = bm.array(((1.0, 0.0, t), (-1.0, 0.0, t),
                         (0.0, 1.0, -t), (0.0, -1.0, -t)))
    else:
        node = bm.array(((0.0, 0.0, 0.0), (eps, 0.0, 0.0),
                         (1.0, eps, 0.0), (0.0, 0.0, 1.0)))
    return TetrahedronMesh(node, cell)


def local_conditioning_example():
    """Study the shape and degree dependence of local D in Section 6.3."""
    cases = ((1, 9, (4, 2, 1, 0)), (1, 11, (4, 2, 1, 0)))
    for family in ('T', 'S'):
        for m, p, r in cases:
            for j in range(7):
                eps = 2.0**(-j)
                mesh = _shape_mesh(eps, family)
                space = CmFESpace3d(mesh, p, r)
                cond = bm.linalg.cond(space.D.to_dense()[0])
                print(f'shape: family={family} d=3 m={m} k={p:<2d} '
                      f'eps={eps:.6f} kappa={cond:.6e}', flush=True)

    for p in range(9, 19):
        for eps in (1.0, 1/8, 1/64):
            mesh = _shape_mesh(eps)
            space = CmFESpace3d(mesh, p, (4, 2, 1, 0))
            cond = bm.linalg.cond(space.D.to_dense()[0])
            print(f'degree: family=T d=3 m=1 k={p:<2d} '
                  f'eps={eps:.6f} kappa={cond:.6e}', flush=True)


def interpolation_example(ns, q=12, error_batch=512):
    """Reproduce the three-dimensional C1 interpolation test in Table 3."""
    expression = sp.sin(2*sp.pi*x)*sp.sin(2*sp.pi*y)*sp.sin(2*sp.pi*z)
    exact = ManufacturedPDE(expression, 3).derivatives(4)
    previous = None
    print('\n3D interpolation: k=11, m=1, r=(4,2,1,0)', flush=True)
    for n in ns:
        mesh = TetrahedronMesh.from_box(nx=n, ny=n, nz=n)
        space = CmFESpace3d(mesh, 11, (4, 2, 1, 0))
        uh = space.interpolate(exact)
        errors = space.error_norms(uh, exact, 2, q, error_batch)
        _print_row(n, space.number_of_global_dofs(), errors, previous)
        previous = errors


def biharmonic_example(ns, q=12, batch_size=1, stiffness_q=8,
                       error_batch=512):
    """Reproduce the three-dimensional C1 biharmonic test in Table 6."""
    expression = sp.sin(5*x)*sp.sin(5*y)*sp.sin(5*z)
    pde = DoubleLaplacePDE(expression, dimension=3)
    exact = pde.derivatives(4)
    source = pde.source

    previous = None
    print('\n3D biharmonic: k=9, m=1, r=(4,2,1,0)', flush=True)
    for n in ns:
        mesh = TetrahedronMesh.from_box(nx=n, ny=n, nz=n)
        space = CmFESpace3d(mesh, 9, (4, 2, 1, 0))

        matrix = space.stiffness_matrix(order=2, q=stiffness_q,
                                        batch_size=batch_size).to_scipy()
        vector = space.source_vector(source, q=q, batch_size=batch_size)

        # Apply the essential boundary data.
        lift, boundary = space.boundary_interpolate(exact)
        free = ~boundary
        solution = lift.copy()
        free_matrix = matrix[free][:, free]
        right = vector[free]-(matrix@lift)[free]

        # Solve the reduced linear system with MUMPS.
        solution[free] = mumps_spsolve(free_matrix, right)

        errors = space.error_norms(solution, exact, 2, q, error_batch)
        _print_row(n, space.number_of_global_dofs(), errors, previous)
        previous = errors


def conditioning_example(degrees=range(9, 19)):
    """Reproduce the 3D Jacobi-scaled stiffness conditions in Figure 7."""
    mesh = TetrahedronMesh.from_box(nx=2, ny=2, nz=2)
    print('\n3D Jacobi-scaled condition numbers: m=1', flush=True)
    for p in degrees:
        space = CmFESpace3d(mesh, p, (4, 2, 1, 0))
        matrix = space.stiffness_matrix().to_scipy()
        free = ~space.is_boundary_dof()
        matrix = matrix[free][:, free]
        scale = 1/bm.sqrt(matrix.diagonal())
        matrix = diags(scale)@matrix@diags(scale)
        largest = eigsh(matrix, k=1, which='LM',
                        return_eigenvectors=False)[0]
        smallest = eigsh(matrix, k=1, sigma=0, which='LM',
                         return_eigenvectors=False)[0]
        condition = largest/smallest
        print(f'k={p:<2d} free_DoF={matrix.shape[0]:<5d} '
              f'kappa={condition:.6e}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    examples = ('all', 'interpolation', 'biharmonic',
                'conditioning', 'normalization', 'local-conditioning')
    parser.add_argument('example', nargs='?', default='all', choices=examples)
    parser.add_argument('--max-n', type=int, default=2)
    parser.add_argument('--min-n', type=int, default=1)
    parser.add_argument('--quadrature', type=int, default=12)
    parser.add_argument('--stiffness-quadrature', dest='stiffness_q',
                        type=int, default=8)
    parser.add_argument('--batch-size', dest='batch', type=int, default=4)
    parser.add_argument('--error-batch-size', dest='error_batch',
                        type=int, default=512)
    args = parser.parse_args()
    levels = (1, 2, 4, 8)
    ns = tuple(n for n in levels if args.min_n <= n <= args.max_n)
    if args.example in ('all', 'interpolation'):
        interpolation_example(ns, args.quadrature, args.error_batch)
    if args.example in ('all', 'biharmonic'):
        biharmonic_example(ns, args.quadrature, args.batch,
                           args.stiffness_q, args.error_batch)
    if args.example in ('all', 'conditioning'):
        conditioning_example()
    if args.example in ('all', 'normalization'):
        normalization_example()
    if args.example in ('all', 'local-conditioning'):
        local_conditioning_example()
