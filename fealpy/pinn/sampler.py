from typing import (
    List,
    SupportsIndex,
    Any
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

    def __and__(self, other):
        if isinstance(other, Sampler):
            return JoinedSampler(self, other)
        else:
            return NotImplemented

    def __or__(self, other):
        if isinstance(other, Sampler):
            return HybridSampler(self, other)
        else:
            return NotImplemented

    def run(self) -> torch.Tensor:
        raise NotImplementedError


class JoinedSampler(Sampler):
    """Generate samples joined from different samplers in dim-0."""
    def __init__(self, *samplers: Sampler) -> None:
        self.samplers: List[Sampler] = []
        for sampler in samplers:
            self.add(sampler)

    @property
    def m(self):
        return sum(x.m for x in self.samplers)

    def add(self, sampler: Sampler):
        if isinstance(sampler, JoinedSampler):
            for sub in sampler.samplers:
                self.add(sub)
        else:
            if self.nd <= 0:
                self.nd = sampler.nd
            elif sampler.nd != self.nd:
                raise ValueError('Cannot join samplers generating samples with different number of features.')
            self.samplers.append(sampler)

    def run(self) -> torch.Tensor:
        return torch.cat([s.run() for s in self.samplers], dim=0)


class HybridSampler(Sampler):
    """Generate samples with features from different samplers in dim-1."""
    def __init__(self, *samplers: Sampler) -> None:
        self.samplers: List[Sampler] = []
        for sampler in samplers:
            self.add(sampler)

    @property
    def nd(self):
        return sum(x.nd for x in self.samplers)

    def add(self, sampler: Sampler):
        if isinstance(sampler, HybridSampler):
            for sub in sampler.samplers:
                self.add(sub)
        else:
            if self.m <= 0:
                self.m = sampler.m
            elif sampler.m != self.m:
                raise ValueError('Cannot hybrid samplers generating different number of samples.')
            self.samplers.append(sampler)

    def run(self) -> torch.Tensor:
        return torch.cat([s.run() for s in self.samplers], dim=1)


class ConstantSampler(Sampler):
    def __init__(self, value: torch.Tensor, requires_grad: bool = False) -> None:
        assert value.ndim == 2
        super().__init__(0, requires_grad)
        self.value = value
        self.m, self.nd = value.shape

    def run(self) -> torch.Tensor:
        return self.value.clone()


class ISampler(Sampler):
    """Generate samples independently in each axis."""
    def __init__(self, m: SupportsIndex, ranges: Any,
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


class BoxEdgeSampler(JoinedSampler):
    """Generate samples on the edges of a multidimensional rectangle."""
    def __init__(self, m_edge: SupportsIndex, p1: TensorOrArray, p2: TensorOrArray, requires_grad: bool=False) -> None:
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


class _MeshSampler(Sampler):
    def __init__(self, m_cell: SupportsIndex, mesh, requires_grad: bool=False) -> None:
        """
        Generate samples in every cells of a mesh.

        Parameters
        ---
        m_cell: int.
            Number of samples in each cell.
        mesh: Mesh.
        requires_grad: bool. Defaults to `False`.
            See `torch.autograd.grad`.
        """
        self.m_cell = int(m_cell)
        m = self.m_cell * mesh.entity('cell').shape[0]
        super().__init__(m=m, requires_grad=requires_grad)
        self.mesh = mesh
        self.nd = mesh.top_dimension()


def _inverse(p, *idx):
        mask = np.array(idx, dtype=np.int_)
        p[:, mask] = 1 - p[:, mask]
        return p


class TriangleMeshSampler(_MeshSampler):
    """Sampler in a triangle mesh."""
    def run(self) -> torch.Tensor:
        bcs = np.zeros((self.m_cell, 3))
        bcs[:, 1:3] = np.random.rand(self.m_cell, 2)
        bcs[:, 0] = 1 - bcs[:, 1] - bcs[:, 2]
        reflect_state = bcs[:, 0] < 0
        bcs[reflect_state, 1:3] = 1 - bcs[reflect_state, 1:3]
        bcs[reflect_state, 0] = - bcs[reflect_state, 0]
        ret: np.ndarray = self.mesh.cell_bc_to_point(bcs).reshape((-1, 2))
        return torch.tensor(ret, dtype=torch.float32, requires_grad=self.requires_grad)


class TetrahedronMeshSampler(_MeshSampler):
    """Sampler in a tetrahedron mesh."""
    def run(self) -> torch.Tensor:
        bcs = np.zeros((self.m_cell, 4))
        u_0 = np.random.rand(self.m_cell, 3)
        bcs[:, 1:4] = u_0
        ui = [
            _inverse(u_0.copy(), 0, 1),
            _inverse(u_0.copy(), 1, 2),
            _inverse(u_0.copy(), 2, 0)
        ]

        for i in range(3):
            mask = np.sum(ui[i], -1) < 1
            bcs[mask, 1:4] = ui[i][mask, :]

        bcs[:, 0] = 1 - np.sum(bcs[:, 1:4], -1)
        mask = bcs[:, 0] < 0
        bcs[mask, 1:4] += 0.5*bcs[mask, 0:1]
        bcs[mask, 0] = 1 - np.sum(bcs[mask, 1:4], -1)
        ret: np.ndarray = self.mesh.cell_bc_to_point(bcs).reshape((-1, 3))
        return torch.tensor(ret, dtype=torch.float32, requires_grad=self.requires_grad)
