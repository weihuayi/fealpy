
from typing import List, Sequence, Any

import torch
from torch import device, Tensor, float64, sqrt, dtype

from fealpy.ml.sampler import Sampler


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


class LineCollocator(Sampler):
    """
    Generate collocation points uniformly on a line segment.
    """
    def __init__(self, nums: int, vertices: Any, dtype: dtype=torch.float64,
                 device: device=None, requires_grad: bool=False) -> None:
        """

        @brief Initializes an LineCollocator instance.

        @param nums: The number of samples to generate.
        @param vertices: An object that can be converted to a `numpy.ndarray`,\
                       representing the vertices of the line segment.\
                       For example, if sampling on a line segment with vertices [0, 1] and [4, 5] ,\
                       use `vertices=[[0, 1], [4, 5]]`.
        @param dtype: Data type of samples. Defaults to `torch.float64`.
        @param requires_grad: A boolean indicating whether the samples should\
                              require gradient computation. Defaults to `False`.\
                              See `torch.autograd.grad`

        @throws ValueError: If `vertices` has an unexpected shape.
        """
        super().__init__(nums, dtype=dtype, device=device, requires_grad=requires_grad)
        if isinstance(vertices, Tensor):
            vertices_arr = vertices.detach().clone().to(device=device)
        else:
            vertices_arr = torch.tensor(vertices, dtype=dtype, device=device)
        if vertices_arr.ndim == 2:
            self.vertex_1 = vertices_arr[0:1, :]
            self.vertex_2 = vertices_arr[1:2, :]
        else:
            raise ValueError(f"Unexpected `vertices` shape {vertices_arr.shape}.")
    
    def standard_form_line(self, point_1: Tensor, point_2: Tensor):
        A = point_2[:, 1] - point_1[:, 1]
        B = point_1[:, 0] - point_2[:, 0]
        C = point_2[:, 0] * point_1[:, 1] - point_1[:, 0] * point_2[:, 1]
        val = torch.tensor([A, B, C])
        if B != 0:
            val = val / B
        return val
    
    def length(self):
        val = int(sqrt((self.vertex_2[:, 0:1] - self.vertex_1[:, 0:1]) ** 2 + (self.vertex_2[:, 1:2] - self.vertex_1[:, 1:2]) ** 2))
        return val

    def run(self):
        line = self.standard_form_line(self.vertex_1, self.vertex_2)
        new_points = torch.zeros((self.m - 2, 2))
        if self.m > 0:
            d = self.length() / (self.m -1)
            for i in range(1, self.m):
                if line[1] != 0:
                    alpha = torch.arctan(-line[0])
                    dx = d * torch.cos(alpha)
                    dy = d * torch.sin(alpha)
                    if self.vertex_2[:, 0] - self.vertex_1[:, 0] > 0:
                        new_points[i-1:, :] = torch.tensor([self.vertex_1[:, 0] + i * dx, 
                                                          self.vertex_1[:, 1] + i * dy], dtype=torch.float64)
                    else:
                        new_points[i-1:, :] = torch.tensor([self.vertex_1[:, 0] - i * dx, 
                                                          self.vertex_1[:, 1] - i * dy], dtype=torch.float64)
                else:
                    if self.vertex_2[:, 1] - self.vertex_1[:, 1] > 0:
                        new_points[i-1:, :] = torch.tensor([self.vertex_1[:, 0],                                       
                                                          self.vertex_1[:, 1] + i * d], dtype=torch.float64)
                    else:
                        new_points[i-1:, :] = torch.tensor([self.vertex_1[:, 0],                                       
                                                          self.vertex_1[:, 1] - i * d], dtype=torch.float64)
        val = torch.cat([self.vertex_1, new_points, self.vertex_2], dim = 0)
        return val


class QuadrangleCollocator(Sampler):
    def __init__(self, nums: int, ranges: Any, dtype: dtype=torch.float64,
                 device: device=None, requires_grad: bool=False) -> None:
        """
        @brief Initializes an QuadrangleCollocator instance.

        @param nums: The number of samples to generate.
        @param ranges: An object that can be converted to a `numpy.ndarray`,\
                       representing the ranges in each sampling axis.\
                       For example, if sampling on the boundary of a regionx which x in [0, 1] and y in [4, 5],\
                       use `ranges=[[0, 1], [4, 5]]`, or `ranges=[0, 1, 4, 5]`.
        @param dtype: Data type of samples. Defaults to `torch.float64`.
        @param requires_grad: A boolean indicating whether the samples should\
                              require gradient computation. Defaults to `False`.\
                              See `torch.autograd.grad`

        @throws ValueError: If `ranges` has an unexpected shape.
        """
        super().__init__(nums, dtype=dtype, device=device, requires_grad=requires_grad)
        self.ranges = ranges
        if isinstance(ranges, Tensor):
            ranges_arr = ranges.detach().clone().to(device=device)
        else:
            ranges_arr = torch.tensor(ranges, dtype=dtype, device=device)
        
        if ranges_arr.ndim == 2:
            self.nd = ranges_arr.shape[0]
            self.lows = ranges_arr[:, 0].reshape(self.nd, )
            self.highs = ranges_arr[:, 1].reshape(self.nd, )
        elif ranges_arr.ndim == 1:
            self.nd, mod = divmod(ranges_arr.shape[0], 2)
            if mod != 0:
                raise ValueError("If `ranges` is 1-dimensional, its length is"
                                 f"expected to be even, but got {mod}.")
            self.lows = ranges_arr[::2].reshape(self.nd, )
            self.highs = ranges_arr[1::2].reshape(self.nd, )
        else:
            raise ValueError(f"Unexpected `ranges` shape {ranges_arr.shape}.")
        self.points = torch.cat([
                                ranges_arr[:, 0:1].reshape(1, -1),
                                torch.cat([ranges_arr[0:1, 0:1], ranges_arr[1:2, 1:2]], dim = 1),
                                ranges_arr[:, 1:2].reshape(1, -1),
                                torch.cat([ranges_arr[0:1, 1:2], ranges_arr[1:2, 0:1]], dim = 1),
                                ranges_arr[:, 0:1].reshape(1, -1)
                                ], dim = 0)

    def run(self):
        points = LineCollocator(self.m // (self.points.shape[0] - 1) + 1, self.points[0:2]).run()
        for i in range(1, self.points.shape[0] - 1):
            new_points = LineCollocator(self.m // (self.points.shape[0] - 1) + 1, self.points[i:i + 2]).run()
            points = torch.cat([points, new_points], dim = 0 )
        unique_points = torch.unique(points, dim = 0)
        return unique_points
