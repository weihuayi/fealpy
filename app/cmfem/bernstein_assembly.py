"""Assemble smooth-element stiffness matrices from lambda derivatives."""

from fealpy.backend import backend_manager as bm

from symmetric_tensor import symmetry_multiplicity


def bernstein_stiffness(space, order, q=None):
    """Integrate lambda derivatives once and form all cell matrices in a batch."""
    if q is None:
        q = space.p+1
    quadrature = space.mesh.quadrature_formula(q, 'cell')
    bcs, weights = quadrature.get_quadrature_points_and_weights()

    gmphi, geo = space.bspace.grad_m_basis(bcs, order, variable='lambda')
    integral = bm.einsum('aqi,bqj,q->abij', gmphi, gmphi, weights)

    TD = space.mesh.top_dimension()
    multiplicity = symmetry_multiplicity(order, TD, dtype=bm.float64)
    geo = bm.einsum('acg,g,bcg->cab', geo, multiplicity, geo)

    measure = space.mesh.entity_measure('cell')
    matrix = bm.einsum('cab,abij,c->cij', geo, integral, measure)
    permutation = space.lattice.permutation
    return matrix[:, permutation, :][:, :, permutation]


def smooth_stiffness(space, order, q=None):
    """Transform local Bernstein matrices to the smooth basis with ``C``."""
    matrix = bernstein_stiffness(space, order, q)
    coefficient = space.coeff
    matrix = coefficient @ matrix
    matrix = matrix @ bm.swapaxes(coefficient, -1, -2)
    return (matrix+bm.swapaxes(matrix, -1, -2))/2
