
from typing import Optional

from ..typing import TensorLike
from ..backend import backend_manager as bm
from ..sparse import COOTensor
from .form import Form
from .. import logger


class NonlinearForm(Form):
    _M: Optional[COOTensor] = None

    