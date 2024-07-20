
import numpy as np
from numpy.typing import NDArray

from .base import Backend, ATTRIBUTE_MAPPING


class NumpyBackend(Backend[NDArray], backend_name='numpy'):
    DATA_CLASS = np.ndarray


attribute_mapping = ATTRIBUTE_MAPPING.copy()
attribute_mapping['device'] = None
NumpyBackend.attach_attributes(attribute_mapping, np)
