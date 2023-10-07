
from typing import List, Any, Sequence

import numpy as np
import torch
from torch import device, Tensor, dtype, float64

from .sampler import Sampler

class Collocator(Sampler):
    """
    Generate collocation points uniformly in n-d rectangle.
    """
    def __init__(self, box: Sequence[float], steps: Sequence[int],
                 dtype=float64,
                 device: device=None, requires_grad: bool=False, **kwargs) -> None:
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
        self.m = reduce(lambda x, y: x*y, steps, 1)
        super().__init__(dtype=dtype, device=device, requires_grad=requires_grad,
                         **kwargs)

    def run(self, *args):
        lins: List[Tensor] = []
        for i in range(self.nd):
            lins.append(torch.linspace(self.starts[i], self.stops[i], self.steps[i],
                                       dtype=self.dtype, device=self.device,
                                       requires_grad=self.requires_grad))
        if self.enable_weight:
            self._weight[:] = 1/self.m
            self._weight = self._weight.broadcast_to(self.m, 1)
        return torch.stack(torch.meshgrid(*lins, indexing='ij'), dim=-1).reshape(-1, self.nd)


class CircleCollocator(Sampler):
    """
    Generate collocation points uniformly on a 2-d circle.
    """
    def __init__(self, cx: float=0, cy: float=0, r: float=1,
                 dtype=float64, device=None, requires_grad=False, **kwargs) -> None:
        super().__init__(dtype=dtype, device=device, requires_grad=requires_grad,
                         **kwargs)
        self.cx, self.cy, self.r = cx, cy, r

    def run(self, nums: int=10):
        angles = torch.linspace(0, 2*torch.pi, nums+1)
        x = self.cx + self.r * torch.cos(angles)
        y = self.cy + self.r * torch.sin(angles)
        points = torch.stack((x, y), dim=1).to(float64)
        if self.enable_weight:
            self._weight[:] = 1/nums
            self._weight = self._weight.broadcast_to(nums, 1)
        return points[:-1, ...]


class LineCollocator(Sampler):
    """
    Generate collocation points uniformly on a line segment.
    """
    def __init__(self, vertices: Any, dtype: dtype=torch.float64,
                 device: device=None, requires_grad: bool=False, **kwargs) -> None:
        """
        @brief Initializes an LineCollocator instance.
        @param nums: The number of samples to generate.
        @param vertices: An object that can be converted to a `numpy.ndarray`,\
               representing the vertices of the line segment.\
               For example, if sampling on a line segment with vertices\
               [0, 1] and [4, 5], use `vertices=[[0, 1], [4, 5]]`.
        @param dtype: Data type of samples. Defaults to `torch.float64`.
        @param requires_grad: A boolean indicating whether the samples should\
               require gradient computation. Defaults to `False`.\
               See `torch.autograd.grad`
        @throws ValueError: If `vertices` has an unexpected shape.
        """
        super().__init__(dtype=dtype, device=device, requires_grad=requires_grad,
                         **kwargs)
        if isinstance(vertices, Tensor):
            vertices_arr = vertices.detach().clone().to(device=device)
        else:
            vertices_arr = torch.tensor(vertices, dtype=dtype, device=device)
        if vertices_arr.ndim == 2:
            self.vertex_1 = vertices_arr[0:1, :]
            self.vertex_2 = vertices_arr[1:2, :]
        else:
            raise ValueError(f"Unexpected `vertices` shape {vertices_arr.shape}.")

    def run(self, nums: int):
        points_dim = self.vertex_1.shape[-1]
        new_points =torch.zeros((nums, self.vertex_1.shape[-1]), dtype = torch.float64)
        for i in range(0, points_dim):
            new_points[:, i:i+1] = torch.linspace(self.vertex_1[:, i].item(), self.vertex_2[:, i].item(), nums, dtype=torch.float64).reshape(-1, 1)
        return new_points


class QuadrangleCollocator(Sampler):
    def __init__(self, ranges: Any, dtype: dtype=torch.float64,
                 device: device=None, requires_grad: bool=False, **kwargs) -> None:
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
        super().__init__(dtype=dtype, device=device, requires_grad=requires_grad,
                         **kwargs)
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

    def run(self, nums: int) -> Tensor:
        points = LineCollocator(self.points[0:2]).run(nums // (self.points.shape[0] - 1) + 1)
        for i in range(1, self.points.shape[0] - 1):
            new_points = LineCollocator(self.points[i:i + 2]).run(nums // (self.points.shape[0] - 1) + 1)
            points = torch.cat([points, new_points], dim = 0 )
        unique_points = torch.unique(points, dim = 0)
        return unique_points


class PolygonCollocator(Sampler):
    def __init__(self, vertices: Any, dtype: dtype=torch.float64,
                 device: device=None, requires_grad: bool=False, **kwargs) -> None:
        """
        @brief Initializes an PolygonCollocator instance.
        @param nums: The number of samples on each edge of the polygon to generate.
        @param vertices: An object that can be converted to a `numpy.ndarray`,\
               representing the vertices of the line segment.\
               For example, if sampling on a polygon in three-dimensional space with vertices\
               [1, 0, 0], [2, 0, 0], [1, 1, 1], [2, 1, 1] use `vertices=[[1, 0, 0], [2, 0, 0], [1, 1, 1], [2, 1, 1]]`.
        @param dtype: Data type of samples. Defaults to `torch.float64`.
        @param requires_grad: A boolean indicating whether the samples should\
               require gradient computation. Defaults to `False`.\
               See `torch.autograd.grad`
        @throws ValueError: If `vertices` has an unexpected shape.
        """
        super().__init__(dtype=dtype, device=device, requires_grad=requires_grad,
                         **kwargs)
        if isinstance(vertices, Tensor):
            self.vertices = vertices.detach().clone().to(device=device)
        else:
            self.vertices = torch.tensor(vertices, dtype=dtype, device=device)
        if self.vertices.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected `vertices` shape {self.vertices.shape}.")
        
    def run(self, *nums: int) -> Tensor:
        vertices_arr = torch.cat([self.vertices, self.vertices[0:1, :]], dim = 0)
        points = LineCollocator(vertices_arr[0:2]).run(nums[0])
        for i in range(1, vertices_arr.shape[0] - 1):
            new_points = LineCollocator(vertices_arr[i:i + 2]).run(nums[i])
            points = torch.cat([points, new_points], dim = 0 )
        unique_points = torch.unique(points, dim = 0)
        return unique_points
    

class SphereCollocator(Sampler):
    def __init__(self, center, radius, method='fibonacci', dtype:dtype=torch.float64,
                 device: device=None, requires_grad: bool=False, **kwargs) -> None:
        """
        @brief Initializes an SphereCollocator instance.
        @param nums: The number of samples to generate.
        @param center: An object that can be converted to a `numpy.ndarray`,\
               representing the ranges in each sampling axis.\
               For example, if sampling on a spherical with center [0, 0, 0]\
               use `center=[0, 0, 0]`.
        @param radius: Radius of ball.
        @param dtype: Data type of samples. Defaults to `torch.float64`.
        @param requires_grad: A boolean indicating whether the samples should\
                              require gradient computation. Defaults to `False`.\
                              See `torch.autograd.grad`
        """
        super().__init__(dtype=dtype, device=device, requires_grad=requires_grad, **kwargs)  
        self.center = torch.tensor(center, dtype=torch.float64)
        self.radius = radius
        if method in {'fibonacci', }:
            self.method = method
        else:
            raise ValueError("Unsupported method. Method 'fibonacci' is the default sampling method ")

    def run(self, nums: int ):
            phi = torch.tensor(np.pi * (3. - np.sqrt(5.)))
            indices = torch.arange(nums, dtype=torch.float64) + 0.5
            y = 1 - (indices / (nums - 1)) * 2  
            r = torch.sqrt(1 - y*y) * self.radius  
            theta = phi * indices 
            x = self.center[0] + torch.cos(theta) * r
            z = self.center[2] + torch.sin(theta) * r
            return torch.stack((x, y + self.center[1], z), dim=1)
