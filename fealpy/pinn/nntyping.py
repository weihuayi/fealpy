from typing import Callable, Optional, Union

from torch import Tensor
from numpy.typing import NDArray

dtype = Optional[bool]
TensorOrArray = Union[Tensor, NDArray]

TensorFunction = Callable[[Tensor], Tensor]
VectorFunction = Callable[[NDArray], NDArray]

Operator = Callable[[Tensor, TensorFunction], Tensor]
