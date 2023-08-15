
from typing import Tuple, List

import torch
from torch import device, Tensor

from .sampler import Sampler


class Collocator(Sampler):
    """
    Generate collocation points.
    """
    def __init__(self, box: Tuple[float], nums: Tuple[int],
                 dtype=torch.float64,
                 device: device=None, requires_grad: bool=False) -> None:
        """
        @brief Prepare to generate collocation points.

        @param box: tuple[float]. The collocate area. For example, `[0, 1, 3, 5]`\
                    means to collocate x in [0, 1] and y in [3, 5].
        @param nums: tuple[int]. Number of steps in each axis/dim.
        @param dtype: Data type of collocation points, defaults to `torch.float64`
        @param device: device.
        @param requires_grad: bool.
        """
        self.nd, r = divmod(len(box), 2)
        if r != 0:
            raise ValueError(f"Length of box must be even, but got {len(box)}.")
        if self.nd != len(nums):
            raise ValueError(f"Length of nums must match the area dimension.")

        self.starts = box[::2]
        self.stops = box[1::2]
        self.steps = nums
        from functools import reduce
        m = reduce(lambda x, y: x*y, nums, 1)
        super().__init__(m, dtype, device, requires_grad)

    def run(self):
        lins: List[Tensor] = []
        for i in range(self.nd):
            lins.append(torch.linspace(self.starts[i], self.stops[i], self.steps[i],
                                       dtype=self.dtype, device=self.device,
                                       requires_grad=self.requires_grad))
        return torch.stack(torch.meshgrid(*lins, indexing='ij'), dim=-1).reshape(-1, self.nd)
