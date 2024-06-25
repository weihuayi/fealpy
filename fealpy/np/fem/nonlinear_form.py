
from typing import TypeVar, Optional

import numpy as np
from numpy.typing import NDArray

from scipy.sparse import csr_matrix

from .. import logger
from ..functionspace.space import FunctionSpace
from .form import Form


_FS = TypeVar('_FS', bound=FunctionSpace)

class NonlinearForm(Form[_FS]):
    pass
        
