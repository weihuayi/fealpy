
from typing import Dict, Optional, Union

import torch

Tensor = torch.Tensor
_dtype = torch.dtype
_device = torch.device


class Quadrature():
    r"""Base class for quadrature generators."""
    def __init__(self, in_memory: bool=True,
                 dtype: Optional[_dtype]=None,
                 device: Union[_device, str, None]=None) -> None:
        self._cache: Optional[Dict[int, Tensor]] = {} if in_memory else None
        self._latest_order = -1
        self.dtype = dtype
        self.device = device

    def __len__(self) -> int:
        return self.number_of_quadrature_points()

    def _get_latest_order(self, order: Optional[int]=None) -> int:
        if order is not None:
            if (not isinstance(order, int)) or (order <= 0):
                raise ValueError("The order must be a positive integer.")
            self._latest_order = order
            return order
        if self._latest_order == -1:
            raise ValueError("The order has not been specified yet.")
        return self._latest_order

    def __getitem__(self, order: int) -> Tensor:
        return self.get(order)

    def __getattr__(self, name: str):
        if name not in {'quadpts', 'weights'}:
            return super().__getattr__(name)
        else:
            if name == 'quadpts':
                return self.get_quadrature_points_and_weights()[0]
            else:
                return self.get_quadrature_points_and_weights()[1]

    def get(self, order: int, *, refresh: bool=False) -> Tensor:
        self._latest_order = order
        if self._cache is not None:
            if (order in self._cache) and (not refresh):
                return self._cache[order]
            else:
                result = self.make(order)
                self._cache[order] = result
                return result
        else:
            return self.make(order)

    def clear(self) -> None:
        self._cache.clear()

    def make(self, order: int) -> Tensor:
        raise NotImplementedError

    def number_of_quadrature_points(self, order: Optional[int]=None) -> int:
        if order is None:
            order = self._get_latest_order(order)
        qw = self.get(order)
        return qw.shape[0]

    def get_quadrature_points_and_weights(self, order: Optional[int]=None) -> Tensor:
        if order is None:
            order = self._get_latest_order(order)
        return self[order][:, :-1], self[order][:, -1]

    def get_quadrature_point_and_weight(self, i: int, order: Optional[int]=None) -> Tensor:
        if order is None:
            order = self._get_latest_order(order)
        return self[order][i, :-1], self[order][i, -1]
