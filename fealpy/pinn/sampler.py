from typing import (
    List,
    Any,
    Generic,
    TypeVar,
    Generator
)
import torch
from torch.autograd import Variable
import numpy as np

from ..mesh.TriangleMesh import TriangleMesh
from ..mesh.TetrahedronMesh import TetrahedronMesh
from .nntyping import MeshLike


class Sampler():
    """
    The base class for all types of samplers.
    """
    m: int = 0
    nd: int = 0
    def __init__(self, m: int=0, requires_grad: bool=False) -> None:
        """
        @brief Initializes a Sampler instance.

        @param m: The number of samples to generate.
        @param requires_grad: A boolean indicating whether the samples should\
                              require gradient computation. Defaults to `False`.
        """
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
        """
        @brief Generates samples.

        @return: A tensor with shape (m, nd) containing the generated samples.
        """
        raise NotImplementedError

    def load(self, epoch: int=1) -> Generator[torch.Tensor, None, None]:
        """
        @brief Return a generator to call `sampler.run()`.

        @param epoch: Iteration number, defaults to 1.

        @return: Generator.
        """
        for _ in range(epoch):
            yield self.run()


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
        """
        @brief Run joined samplers.

        @return: Samples concatenated in dim-0.
        """
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
        """
        @brief Run hybrid samplers.

        @return: Samples concatenated in dim-1.
        """
        return torch.cat([s.run() for s in self.samplers], dim=1)


class ConstantSampler(Sampler):
    """
    A sampler generating constants.
    """
    def __init__(self, value: torch.Tensor, requires_grad: bool = False) -> None:
        assert value.ndim == 2
        super().__init__(0, requires_grad)
        self.value = value
        self.m, self.nd = value.shape

    def run(self) -> torch.Tensor:
        return self.value.clone()


class ISampler(Sampler):
    """
    A sampler that generates samples independently in each axis.
    """
    def __init__(self, m: int, ranges: Any,
                 requires_grad: bool=False) -> None:
        """
        @brief Initializes an ISampler instance.

        @param m: The number of samples to generate.
        @param ranges: An object that can be converted to a `numpy.ndarray`,\
                       representing the ranges in each sampling axis.
        @param requires_grad: A boolean indicating whether the samples should\
                              require gradient computation. Defaults to `False`.\
                              See `torch.autograd.grad`

        @throws ValueError: If `ranges` has an unexpected shape.
        """
        super().__init__(m=m, requires_grad=requires_grad)
        ranges_arr = np.array(ranges, np.float32)
        if len(ranges_arr.shape) == 2:
            self.nd = ranges_arr.shape[0]
        else:
            raise ValueError(f"Unexpected `ranges` shape {ranges_arr.shape}.")
        self.lows = ranges_arr[:, 0].reshape(1, self.nd)
        self.highs = ranges_arr[:, 1].reshape(1, self.nd)
        self.deltas = self.highs - self.lows

    def run(self) -> torch.Tensor:
        """
        @brief Generates independent samples in each axis.

        @return: A tensor with shape (m, nd) containing the generated samples.
        """
        ret = np.random.rand(self.m, self.nd) * self.deltas + self.lows
        return Variable(torch.from_numpy(ret).float(), requires_grad=self.requires_grad)


class BoxBoundarySampler(JoinedSampler):
    """Generate samples on the boundaries of a multidimensional rectangle."""
    def __init__(self, m_edge: int, p1: List[float], p2: List[float], requires_grad: bool=False) -> None:
        """
        @brief Generate samples on the boundaries of a multidimensional rectangle.

        @param m: int. Number of samples in each boundary.
        @param p1, p2: Object that can be converted to `torch.Tensor`.\
                       Points at both ends of the diagonal.
        @param requires_grad: bool. Defaults to `False`. See `torch.autograd.grad`.
        """
        super().__init__()
        t1, t2 = torch.tensor(p1), torch.tensor(p2)
        if len(t1.shape) != 1:
            raise ValueError
        if t1.shape != t2.shape:
            raise ValueError
        data = torch.vstack([t1, t2]).T
        range1, range2 = data.clone(), data.clone()
        for d in range(t1.shape[0]):
            range1[d, :] = data[d, 0]
            range2[d, :] = data[d, 1]
            self.add(ISampler(m=m_edge, ranges=range1, requires_grad=requires_grad))
            self.add(ISampler(m=m_edge, ranges=range2, requires_grad=requires_grad))


_MT = TypeVar("_MT", bound=MeshLike)

class _MeshSampler(Sampler, Generic[_MT]):
    def __init__(self, m_cell: int, mesh:_MT, requires_grad: bool=False) -> None:
        """
        @brief Generate samples in every cells of a mesh.

        @param m_cell: int. Number of samples in each cell.
        @param mesh: Mesh.
        @param requires_grad: bool. Defaults to `False`. See `torch.autograd.grad`.
        """
        self.m_cell = int(m_cell)
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        m = self.m_cell * cell.shape[0]
        super().__init__(m=m, requires_grad=requires_grad)
        self.mesh = mesh
        TD = mesh.top_dimension()
        self.nd = node.shape[1]

        bcs_example = np.zeros((m_cell, TD+1))
        self._path_info = np.einsum_path('...j, ijk->...ik', bcs_example, node[cell])[0]

    def cell_bc_to_point(self, bcs) -> np.ndarray:
        """
        The optimized version of method `mesh.cell_bc_to_point()`
        to support faster sampling.
        """
        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        return np.einsum('...j, ijk->...ik', bcs, node[cell], optimize=self._path_info)


def random_weights(m: int, n: int):
    """
    @brief Generate m random samples, where each sample has n features (n >= 2),\
    such that the sum of each feature is 1.0.

    @param m: The number of samples to generate.
    @param n: The number of features in each sample.

    @return: An ndarray with shape (m, n), where each row represents a random sample.

    @throws ValueError: If n < 2.
    """
    m, n = int(m), int(n)
    if n < 2:
        raise ValueError(f'Integer `n` should be larger than 1 but got {n}.')
    u = np.zeros((m, n+1))
    u[:, n] = 1.0
    u[:, 1:n] = np.sort(np.random.rand(m, n-1), axis=1)
    return u[:, 1:n+1] - u[:, 0:n]


class TriangleMeshSampler(_MeshSampler[TriangleMesh]):
    """Sampler in a triangle mesh."""
    def run(self) -> torch.Tensor:
        bcs = random_weights(self.m_cell, 3)
        ret = self.cell_bc_to_point(bcs).reshape((-1, 2))
        return torch.tensor(ret, dtype=torch.float32, requires_grad=self.requires_grad)


class TetrahedronMeshSampler(_MeshSampler[TetrahedronMesh]):
    """Sampler in a tetrahedron mesh."""
    def run(self) -> torch.Tensor:
        bcs = random_weights(self.m_cell, 4)
        ret = self.cell_bc_to_point(bcs).reshape((-1, 3))
        return torch.tensor(ret, dtype=torch.float32, requires_grad=self.requires_grad)
