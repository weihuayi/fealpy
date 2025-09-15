from typing import Callable

from .. import logger
from ..typing import TensorLike, Index, _S
from ..backend import backend_manager as bm


if bm.backend_name == "pytorch":
    ReLU = bm.relu
else:
    ReLU = lambda x: bm.maximum(x, 0.)

GradReLU = lambda x: bm.where(x > 0., bm.ones_like(x), bm.zeros_like(x))


class NeuralMesh2D:
    def __init__(
        self,
        weight: TensorLike,
        bias: TensorLike,
        /
    ):
        assert weight.ndim == 2
        assert bias.ndim == 1
        assert weight.shape[0] == bias.shape[0]
        assert weight.shape[1] == 2
        self.weight = weight
        self.bias = bias

    def geo_dimension(self) -> int:
        return 2

    def top_dimension(self) -> int:
        return 2

    def number_of_cells(self) -> int:
        return self.weight.shape[0]

    def number_of_cells(self) -> int:
        return self.weight.shape[0]

    def bc_to_point(self, bc: TensorLike, index: Index = _S) -> TensorLike:
        pass

    def point_to_bc(self, points: TensorLike, index: Index = _S) -> TensorLike:
        weight = self.weight[index, :]
        bias = self.bias[index]
        L0 = bm.einsum("cd, qd -> cq", weight, points) + bias[:, None]
        return L0

    def shape_function(
        self,
        bcs: TensorLike,
        p: int = 1,
        *,
        index: Index = _S
    ) -> TensorLike:
        if p == 1:
            return bcs[:, None] # (..., ldof)
        else:
            raise NotImplementedError

    def grad_shape_function(
        self,
        bcs: TensorLike,
        p: int = 1,
        *,
        index: Index = _S,
        variables: str = 'u'
    ) -> TensorLike:
        if p == 1:
            if variables == 'u':
                return 1
            elif variables == 'x':
                return self.weight[index, :]
            else:
                raise ValueError("Variables type is expected to be 'u' or 'x', "
                                 f"but got '{variables}'.")
        else:
            raise NotImplementedError
