
import numpy as np
from numpy.typing import NDArray

from .base import (
    Backend, ATTRIBUTE_MAPPING, CREATION_MAPPING, REDUCTION_MAPPING,
    UNARY_MAPPING, BINARY_MAPPING, OTHER_MAPPING
)


class NumpyBackend(Backend[NDArray], backend_name='numpy'):
    DATA_CLASS = np.ndarray

    ### Tensor creation methods ###
    # NOTE: all copied

    ### Reduction methods ###
    # NOTE: all copied

    ### Unary methods ###
    # NOTE: all copied

    ### Binary methods ###
    # NOTE: all copied

    ### Other methods ###

    @classmethod
    def nonzero(cls, a, /, as_tuple=True):
        cls.show_unsupported(not as_tuple, 'nonzero', 'as_tuple')
        return np.nonzero(a)

    @staticmethod
    def cat(iterable, axis=0, out=None) -> NDArray:
        return np.concatenate(iterable, axis=axis, out=out)

    ### FEALPy methods ###


NumpyBackend.attach_attributes(ATTRIBUTE_MAPPING, np)
creation_mapping = CREATION_MAPPING.copy()
creation_mapping['tensor'] = 'array'
NumpyBackend.attach_methods(CREATION_MAPPING, np)
NumpyBackend.attach_methods(REDUCTION_MAPPING, np)
NumpyBackend.attach_methods(UNARY_MAPPING, np)
NumpyBackend.attach_methods(BINARY_MAPPING, np)
NumpyBackend.attach_methods(OTHER_MAPPING, np)
