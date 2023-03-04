from typing import Callable, Optional, Union

import torch
import numpy as np

dtype = Optional[bool]
TensorOrArray = Union[torch.Tensor, np.ndarray]

TensorFunction = Callable[[torch.Tensor], torch.Tensor]
VectorFunction = Callable[[np.ndarray], np.ndarray]
