
from typing import List, Sequence

import torch
from torch import device, Tensor, float64

from .sampler import Sampler, _MeshSampler


class Collocator(Sampler):
    """
    Generate collocation points uniformly in n-d rectangle.
    """
    def __init__(self, box: Sequence[float], steps: Sequence[int],
                 dtype=float64,
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
        if self.nd != len(steps):
            raise ValueError(f"Length of nums must match the area dimension.")

        self.starts = box[::2]
        self.stops = box[1::2]
        self.steps = steps
        from functools import reduce
        m = reduce(lambda x, y: x*y, steps, 1)
        super().__init__(m, dtype, device, requires_grad)

    def run(self):
        lins: List[Tensor] = []
        for i in range(self.nd):
            lins.append(torch.linspace(self.starts[i], self.stops[i], self.steps[i],
                                       dtype=self.dtype, device=self.device,
                                       requires_grad=self.requires_grad))
        return torch.stack(torch.meshgrid(*lins, indexing='ij'), dim=-1).reshape(-1, self.nd)


class CircleCollocator(Sampler):
    """
    Generate collocation points uniformly on a 2-d circle.
    """
    def __init__(self, cx: float=0, cy: float=0, r: float=1, nums=10,
                 dtype=float64, device=None, requires_grad=False) -> None:
        super().__init__(nums, dtype, device, requires_grad)
        self.cx, self.cy, self.r = cx, cy, r

    def run(self):
        angles = torch.linspace(0, 2*torch.pi, self.m+1)
        x = self.cx + self.r * torch.cos(angles)
        y = self.cy + self.r * torch.sin(angles)
        points = torch.stack((x, y), dim=1).to(float64)
        return points[:-1, ...]
