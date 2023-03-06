from typing import (
    List,
    SupportsIndex
)
import torch
from torch.autograd import Variable
import numpy as np

from .nntyping import TensorOrArray


__all__ = [
    "ISampler",
    "BoxEdgeSampler",
    "TriangleMeshSampler"
]

class Sampler():
    m: int = 0
    nd: int = 0
    def __init__(self, m: SupportsIndex=0, requires_grad: bool=False) -> None:
        self.m = int(m)
        self.requires_grad = bool(requires_grad)

    def __add__(self, other):
        if isinstance(other, Sampler):
            return CompoundSampler(self, other)
        else:
            return NotImplemented

    def run(self) -> torch.Tensor:
        raise NotImplementedError


class CompoundSampler(Sampler):
    def __init__(self, *samplers: Sampler) -> None:
        self.samplers: List[Sampler] = []
        for sampler in samplers:
            self.add(sampler)

    def __add__(self, other):
        if isinstance(other, Sampler):
            ret = CompoundSampler(*self.samplers)
            ret.add(other)
            return ret
        else:
            return NotImplemented

    @property
    def m(self):
        return sum(x.m for x in self.samplers)

    def add(self, sampler: Sampler):
        if isinstance(sampler, CompoundSampler):
            for sub in sampler.samplers:
                self.add(sub)
        else:
            if self.nd <= 0:
                self.nd = sampler.nd
            elif sampler.nd != self.nd:
                raise ValueError('Cannot add samplers with different dimension.')
            self.samplers.append(sampler)

    def run(self) -> torch.Tensor:
        results = []
        for sampler in self.samplers:
            results.append(sampler.run())
        return torch.cat(results, dim=0)


class ISampler(Sampler):
    """Generate samples independently in each axis."""
    def __init__(self, m: SupportsIndex, ranges: TensorOrArray,
                 requires_grad: bool=False) -> None:
        """
        Generate samples independently in each axis.

        Parameters
        ---
        m: int.
            Number of samples.
        ranges: Object that can be converted to `numpy.ndarray`.
            Ranges in each sampling axes.
        requires_grad: bool. Defaults to `False`.
            See `torch.autograd.grad`.
        """
        super().__init__(m=m, requires_grad=requires_grad)
        self.ranges = np.array(ranges)
        if len(self.ranges.shape) == 2:
            self.nd = self.ranges.shape[0]
        else:
            raise ValueError

    def run(self) -> torch.Tensor:
        ret = np.zeros((self.m, self.nd))
        for i in range(self.nd):
            low = self.ranges[i, 0]
            high = self.ranges[i, 1]
            if abs(high - low) < 1e-16:
                ret[..., i] = low
            else:
                ret[..., i] = torch.rand(self.m)*(high - low) + low
        return Variable(torch.from_numpy(ret).float(), requires_grad=self.requires_grad)


class BoxEdgeSampler(CompoundSampler):
    """Generate samples on the edges of a multidimensional rectangle."""
    def __init__(self, m_edge: SupportsIndex, p1: TensorOrArray, p2: TensorOrArray, requires_grad: bool = False) -> None:
        """
        Generate samples on the edges of a multidimensional rectangle.

        Parameters
        ---
        m: int.
            Number of samples in each edge.
        p1, p2: Object that can be converted to `torch.Tensor`.
            Points at both ends of the diagonal.
        requires_grad: bool. Defaults to `False`.
            See `torch.autograd.grad`.
        """
        super().__init__()
        p1, p2 = torch.Tensor(p1), torch.Tensor(p2)
        if len(p1.shape) != 1:
            raise ValueError
        if p1.shape != p2.shape:
            raise ValueError
        data = torch.vstack([p1, p2]).T
        range1, range2 = data.clone(), data.clone()
        for d in range(p1.shape[0]):
            range1[d, :] = data[d, 0]
            range2[d, :] = data[d, 1]
            self.add(ISampler(m=m_edge, ranges=range1, requires_grad=requires_grad))
            self.add(ISampler(m=m_edge, ranges=range2, requires_grad=requires_grad))


class TriangleMeshSampler(Sampler):
    """Sampler in a triangle mesh."""
    def __init__(self, m_cell: SupportsIndex, mesh, requires_grad: bool=False) -> None:
        """
        Generate samples in every cells of a triangle mesh.

        Parameters
        ---
        m_cell: int.
            Number of samples in each cell.
        mesh: TriangleMesh.
        requires_grad: bool. Defaults to `False`.
            See `torch.autograd.grad`.
        """
        self.m_cell = int(m_cell)
        m = self.m_cell * mesh.entity('cell').shape[0]
        super().__init__(m=m, requires_grad=requires_grad)
        self.mesh = mesh

    def run(self) -> torch.Tensor:
        bcs = np.zeros((self.m_cell, 3))
        bcs[:, 1:3] = np.random.rand(self.m_cell, 2)
        bcs[:, 0] = 1 - bcs[:, 1] - bcs[:, 2]
        reflect_state = bcs[:, 0] < 0
        bcs[reflect_state, 1:3] = 1 - bcs[reflect_state, 1:3]
        bcs[reflect_state, 0] = - bcs[reflect_state, 0]
        ret: np.ndarray = self.mesh.cell_bc_to_point(bcs).reshape((-1, 2))
        return torch.tensor(ret, dtype=torch.float32, requires_grad=self.requires_grad)
