
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix


def to_global(matrix: NDArray, dof_idx_left: NDArray, dof_idx_right: NDArray, gdof: int):
    r"""Assembly face matrix to global matrix.

    Input:
        matrix: (NF, ldof, ldof)
        dof_idx_left: (NF, ldof)
        dof_idx_right: (NF, ldof)
        gdof: int
    """
    I = np.broadcast_to(dof_idx_left[:, :, None], matrix.shape)
    J = np.broadcast_to(dof_idx_right[:, None, :], matrix.shape)
    return coo_matrix((matrix.flat, (I.flat, J.flat)), shape=(gdof, gdof))
