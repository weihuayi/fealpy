from typing import (
    List,
    Any,
    Generator,
    Callable
)
import torch
from torch import Tensor, dtype
import numpy as np
from numpy.typing import NDArray

from ..nntyping import MeshLike


class Sampler():
    """
    The base class for all types of samplers.
    """
    m: int = 0
    nd: int = 0
    def __init__(self, m: int=0, dtype: dtype=torch.float32, requires_grad: bool=False) -> None:
        """
        @brief Initializes a Sampler instance.

        @param m: The number of samples to generate.
        @param dtype: Data type of samples. Defaults to `torch.float32`.
        @param requires_grad: A boolean indicating whether the samples should\
                              require gradient computation. Defaults to `False`.
        """
        self.m = int(m)
        self.dtype = dtype
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

    def run(self) -> Tensor:
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

    def run(self) -> Tensor:
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

    def run(self) -> Tensor:
        """
        @brief Run hybrid samplers.

        @return: Samples concatenated in dim-1.
        """
        return torch.cat([s.run() for s in self.samplers], dim=1)


class ConstantSampler(Sampler):
    """
    A sampler generating constants.
    """
    def __init__(self, value: Tensor, requires_grad: bool=False) -> None:
        """
        @brief Build a sampler generats constants.

        @param value: A constant tensor.
        @param requires_grad: bool.
        """
        assert value.ndim == 2
        super().__init__(m=0, requires_grad=requires_grad)
        self.value = value
        self.m, self.nd = value.shape

    def run(self) -> Tensor:
        ret = self.value.clone()
        ret.requires_grad = self.requires_grad
        return ret


class ISampler(Sampler):
    """
    A sampler that generates samples independently in each axis.
    """
    def __init__(self, m: int, ranges: Any, dtype: dtype=torch.float64,
                 requires_grad: bool=False) -> None:
        """
        @brief Initializes an ISampler instance.

        @param m: The number of samples to generate.
        @param ranges: An object that can be converted to a `numpy.ndarray`,\
                       representing the ranges in each sampling axis.
        @param dtype: Data type of samples. Defaults to `torch.float64`.
        @param requires_grad: A boolean indicating whether the samples should\
                              require gradient computation. Defaults to `False`.\
                              See `torch.autograd.grad`

        @throws ValueError: If `ranges` has an unexpected shape.
        """
        super().__init__(m=m, dtype=dtype, requires_grad=requires_grad)
        ranges_arr = np.array(ranges, dtype=np.float64)
        if len(ranges_arr.shape) == 2:
            self.nd = ranges_arr.shape[0]
        else:
            raise ValueError(f"Unexpected `ranges` shape {ranges_arr.shape}.")
        self.lows = ranges_arr[:, 0].reshape(1, self.nd)
        self.highs = ranges_arr[:, 1].reshape(1, self.nd)
        self.deltas = self.highs - self.lows

    def run(self) -> Tensor:
        """
        @brief Generates independent samples in each axis.

        @return: A tensor with shape (m, nd) containing the generated samples.
        """
        ret = torch.rand((self.m, self.nd), dtype=self.dtype) * self.deltas + self.lows
        ret.requires_grad = self.requires_grad
        return ret


class BoxBoundarySampler(JoinedSampler):
    """Generate samples on the boundaries of a multidimensional rectangle."""
    def __init__(self, m_edge: int, p1: List[float], p2: List[float],
                 dtype: dtype=torch.float64, requires_grad: bool=False) -> None:
        """
        @brief Generate samples on the boundaries of a multidimensional rectangle.

        @param m: int. Number of samples in each boundary.
        @param p1, p2: Object that can be converted to `torch.Tensor`.\
                       Points at both ends of the diagonal.
        @param dtype: Data type of samples. Defaults to `torch.float64`.
        @param requires_grad: bool. Defaults to `False`. See `torch.autograd.grad`.
        """
        super().__init__()
        t1, t2 = torch.tensor(p1), torch.tensor(p2)
        if len(t1.shape) != 1:
            raise ValueError
        if t1.shape != t2.shape:
            raise ValueError
        data = torch.vstack([t1, t2]).T

        for d in range(t1.shape[0]):
            range1, range2 = data.clone(), data.clone()
            range1[d, :] = data[d, 0]
            range2[d, :] = data[d, 1]
            self.add(ISampler(m=m_edge, ranges=range1, dtype=dtype, requires_grad=requires_grad))
            self.add(ISampler(m=m_edge, ranges=range2, dtype=dtype, requires_grad=requires_grad))


##################################################
### Mesh samplers
##################################################

class _MeshSampler(Sampler):
    def __init__(self, m_cell: int, mesh: MeshLike, dtype: dtype=torch.float64,
                 requires_grad: bool=False) -> None:
        """
        @brief Generate samples in every cells of a mesh.

        @param m_cell: int. Number of samples in each cell.
        @param mesh: Mesh.
        @param dtype: Data type of samples. Defaults to `torch.float64`.
        @param requires_grad: bool. Defaults to `False`. See `torch.autograd.grad`.
        """
        self.m_cell = int(m_cell)
        self.node = mesh.entity('node')
        self.nd = self.node.shape[-1]
        self.node = self.node.reshape(-1, self.nd)
        self.cell = mesh.entity('cell')
        self.NVC = self.cell.shape[-1]

        m = self.m_cell * self.cell.shape[0]
        super().__init__(m=m, dtype=dtype, requires_grad=requires_grad)
        self.mesh = mesh
        self.bcs = np.zeros((m_cell, self.NVC))
        """The latest bcs generated by the mesh sampler."""
        self._path_info = np.einsum_path('...j, ijk->...ik', self.bcs, self.node[self.cell])[0]
        self._init_weight()

    def _init_weight(self) -> None:
        raw = self.mesh.entity_measure(etype='cell')/self.m_cell
        if isinstance(raw, (float, int)):
            arr = torch.tensor([raw, ], dtype=self.dtype).broadcast_to(self.cell.shape[0], 1)
        elif isinstance(raw, np.ndarray):
            arr = torch.from_numpy(raw)[:, None]
        else:
            raise TypeError(f"Invalid return from cell_area method.")
        self.weight = arr.repeat(1, self.m_cell).reshape(-1, 1)

    def cell_bc_to_point(self, bcs: NDArray) -> NDArray:
        """
        The optimized version of method `mesh.cell_bc_to_point()`
        to support faster sampling.
        """
        node = self.node
        cell = self.cell
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


class TMeshSampler(_MeshSampler):
    def run(self) -> Tensor:
        self.bcs = random_weights(self.m_cell, self.NVC)
        ret = self.cell_bc_to_point(self.bcs).reshape((-1, self.nd))
        return torch.tensor(ret, dtype=self.dtype, requires_grad=self.requires_grad)

TriangleMeshSampler = TMeshSampler
"""Sampler in a triangle mesh. This name is deprecated, please use 'TMeshSampler' instead.
And 'get_mesh_sampler()' is recommended as it is able to get the target sampler directly."""
TetrahedronMeshSampler = TMeshSampler
"""Sampler in a tetrahedron mesh. This name is deprecated, please use 'TMeshSampler' instead.
And 'get_mesh_sampler()' is recommended as it is able to get the target sampler directly."""


class QuadrangleMeshSampler(_MeshSampler):
    """Sampler in a quadrangle mesh."""
    def run(self) -> Tensor:
        bc_0 = random_weights(self.m_cell, 2)
        bc_1 = random_weights(self.m_cell, 2)
        self.bcs[..., 0] = bc_0[..., 0] * bc_1[..., 0]
        self.bcs[..., 1] = bc_0[..., 1] * bc_1[..., 0]
        self.bcs[..., 2] = bc_0[..., 1] * bc_1[..., 1]
        self.bcs[..., 3] = bc_0[..., 0] * bc_1[..., 1]
        ret = self.cell_bc_to_point(self.bcs).reshape((-1, 2))
        return torch.tensor(ret, dtype=self.dtype, requires_grad=self.requires_grad)


class UniformMesh2dSampler(_MeshSampler):
    """Sampler in a 2-d uniform mesh."""
    def run(self) -> Tensor:
        bc_0 = random_weights(self.m_cell, 2)
        bc_1 = random_weights(self.m_cell, 2)
        self.bcs[..., 0] = bc_0[..., 0] * bc_1[..., 0]
        self.bcs[..., 1] = bc_0[..., 0] * bc_1[..., 1]
        self.bcs[..., 2] = bc_0[..., 1] * bc_1[..., 0]
        self.bcs[..., 3] = bc_0[..., 1] * bc_1[..., 1]
        ret = self.cell_bc_to_point(self.bcs).reshape((-1, 2))
        return torch.tensor(ret, dtype=self.dtype, requires_grad=self.requires_grad)


DIRECTOR = {
    'TriangleMesh': TMeshSampler,
    'TetrahedronMesh': TMeshSampler,
    'QuadrangleMesh': QuadrangleMeshSampler,
    'UniformMesh2d': UniformMesh2dSampler
}

def get_mesh_sampler(m_cell: int, mesh: MeshLike, dtype: dtype=torch.float64,
                     requires_grad: bool=False) -> _MeshSampler:
    """
    @brief Get a sampler that generates samples in every cells of a mesh.

    @param m_cell: int. Number of samples in each cell.
    @param mesh: Mesh.
    @param dtype: Data type of samples. Defaults to `torch.float32`.
    @param requires_grad: bool. Defaults to `False`. See `torch.autograd.grad`.

    @return: A mesh sampler works on the given mesh type.

    @note: Now triangle mesh, tetrahedron mesh and quadrangle mesh are supported. Samplers for more mesh types will be\
    implemented in the future.
    """
    mesh_name = mesh.__class__.__name__
    ms_class = DIRECTOR.get(mesh_name, None)
    if ms_class is None:
        raise NotImplementedError(f"Sampler for {mesh_name} has not been implemented.")

    return ms_class(m_cell=m_cell, mesh=mesh, dtype=dtype, requires_grad=requires_grad)
