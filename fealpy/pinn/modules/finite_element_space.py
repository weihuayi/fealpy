
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.nn import Module


class LagrangeFESapce(Module):
    def __init__(self, uh: NDArray) -> None:
        super().__init__()
        self.__uh = uh

    def forward(self, bc: Tensor):
        ...

# TODO: finish this
