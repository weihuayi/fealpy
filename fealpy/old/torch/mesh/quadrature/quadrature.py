
from typing import Tuple, Optional, Union

import torch

Tensor = torch.Tensor
_dtype = torch.dtype
_device = torch.device


class Quadrature():
    r"""Base class for quadrature generators."""
    def __init__(self, index: Optional[int]=None, *,
                 dtype: Optional[_dtype]=None,
                 device: Union[_device, str, None]=None) -> None:
        self.dtype = dtype
        self.device = device
        self.quadpts, self.weights = self.make(index)

    def __len__(self) -> int:
        return self.number_of_quadrature_points()

    def __getitem__(self, i: int) -> Tensor:
        return self.get_quadrature_point_and_weight(i)

    def make(self, index: int) -> Tensor:
        raise NotImplementedError

    def number_of_quadrature_points(self) -> int:
        return self.weights.shape[0]

    def get_quadrature_points_and_weights(self) -> Tuple[Tensor, Tensor]:
        """Get all quadrature points and weights in the formula.

        Returns:
            (Tensor, Tensor): Quadrature points and weights.
        """
        return self.quadpts, self.weights

    def get_quadrature_point_and_weight(self, i: int) -> Tuple[Tensor, Tensor]:
        """Get the i-th quadrature point and weight.

        Parameters:
            i (int): _description_

        Returns:
            (Tensor, Tensor): A quadrature point and weight.
        """
        return self.quadpts[i, :], self.weights[i]
