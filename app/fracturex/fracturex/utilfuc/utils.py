import numpy as np
from typing import Optional

from fealpy.experimental.typing import TensorLike
from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.material.elastic_material import LinearElasticMaterial


def compute_strain(u: TensorLike, q=None) -> TensorLike:
    """
    Compute the strain tensor.

    Parameters
    ----------
    u : TensorLike
        The displacement field.

    Returns
    -------
    TensorLike
        The strain tensor.
    """
    q = self.q
    mesh = u.space.mesh
    qf = mesh.quadrature_formula(q, 'cell')
    bc, ws = qf.get_quadrature_points_and_weights() 
    guh = u.grad_value(bc)

    GD = guh.shape[-1]
    strain = bm.zeros_like(guh)
    for i in range(GD):
        for j in range(GD):
            strain[..., i, j] = 0.5 * (guh[..., i, j] + guh[..., j, i])
    return strain


