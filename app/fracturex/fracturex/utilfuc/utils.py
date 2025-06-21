
from typing import Optional

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm


def flatten_symmetric_matrices(matrices):
    """
    Shape it as Flatten the symmetric matrix.
    """
    flatten_rules = {
        (2, 2): [ 
            (0, 0),  
            (1, 1), 
            (0, 1)  
        ],
        (3, 3): [
            (0, 0),  
            (1, 1),  
            (2, 2), 
            (0, 1), 
            (1, 2),
            (0, 2) 
        ]
    }

    matrix_shape = matrices.shape[-2:]

    if matrix_shape not in flatten_rules:
        raise ValueError("The shape of the matrix is not supported.")
    
    rules = flatten_rules[matrix_shape]

    flattened = bm.stack([matrices[..., i, j] for i, j in rules], axis=-1)

    return flattened



