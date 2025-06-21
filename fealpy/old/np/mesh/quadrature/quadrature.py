
from typing import Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray

_dtype = np.dtype



class Quadrature():
    r"""Base class for quadrature generators."""
    def __init__(self, index: Optional[int]=None, in_memory: bool=True,
                 dtype: Optional[_dtype]=None
                 ) -> None:
        self._cache: Optional[Dict[int, NDArray]] = {} if in_memory else None
        self._latest_index = index
        self.dtype = dtype

    def __len__(self) -> int:
        return self.number_of_quadrature_points()

    def _get_latest_index(self, index: Optional[int]=None) -> int:
        if index is not None:
            if (not isinstance(index, int)) or (index <= 0):
                raise ValueError("The index must be a positive integer.")
            self._latest_index = index
            return index
        if self._latest_index == -1:
            raise ValueError("The index has not been specified yet.")
        return self._latest_index

    def __getitem__(self, index: int) -> NDArray:
        return self.get(index)

    def __getattr__(self, name: str):
        if name not in {'quadpts', 'weights'}:
            return super().__getattr__(name)
        else:
            if name == 'quadpts':
                return self.get_quadrature_points_and_weights()[0]
            else:
                return self.get_quadrature_points_and_weights()[1]

    def get(self, index: int, *, refresh: bool=False) -> NDArray:
        self._latest_index = index
        if self._cache is not None:
            if (index in self._cache) and (not refresh):
                return self._cache[index]
            else:
                result = self.make(index)
                self._cache[index] = result
                return result
        else:
            return self.make(index)

    def clear(self) -> None:
        self._cache.clear()

    def make(self, index: int) -> NDArray:
        raise NotImplementedError

    def number_of_quadrature_points(self, index: Optional[int]=None) -> int:
        if index is None:
            index = self._get_latest_index(index)
        qw = self.get(index)
        return qw.shape[0]

    def get_quadrature_points_and_weights(self, index: Optional[int]=None) -> NDArray:
        if index is None:
            index = self._get_latest_index(index)
        return self[index][:, :-1], self[index][:, -1]

    def get_quadrature_point_and_weight(self, i: int, index: Optional[int]=None) -> NDArray:
        if index is None:
            index = self._get_latest_index(index)
        return self[index][i, :-1], self[index][i, -1]
