from warnings import warn
from typing import (
    Tuple, List, Dict, Any, Generator, Type, Optional, Literal, Sequence
)
from math import log2
import torch
from torch import Tensor, float64, device

from . import functional as F

SampleMode = Literal['random', 'linspace']


def _as_tensor(__sequence: Sequence, dtype=float64, device: device=None):
    seq = __sequence
    if isinstance(seq, Tensor):
        return seq.detach().clone().to(device=device).to(dtype=dtype)
    else:
        return torch.tensor(seq, dtype=dtype, device=device)


class Sampler():
    """
    The base class for all types of samplers.
    """
    nd: int = 0
    _weight: Tensor
    def __init__(self, enable_weight=False,
                 dtype=float64, device: device=None,
                 requires_grad: bool=False, **kwargs) -> None:
        """
        @brief Initializes a Sampler instance.

        @param dtype: Data type of samples. Defaults to `torch.float64`.
        @param device: device.
        @param requires_grad: A boolean indicating whether the samples should\
               require gradient computation. Defaults to `False`.
        """
        self.enable_weight = enable_weight
        self.dtype = dtype
        self.device = device
        self.requires_grad = bool(requires_grad)
        self._weight = torch.tensor(torch.nan, dtype=dtype, device=device)

    def run(self, n: int) -> Tensor:
        """
        @brief Generates samples.

        @return: A tensor with shape (n, GD) containing the generated samples.
        """
        raise NotImplementedError

    def weight(self) -> Tensor:
        """
        @brief Get weights of the latest sample points. The weight of a sample is\
               equal to the reciprocal of the sampling density.

        @return: A tensor with shape (m, 1).
        """
        return self._weight

    def load(self, n: int, epoch: int=1) -> Generator[torch.Tensor, None, None]:
        """
        @brief Return a generator to call `sampler.run()`.

        @param epoch: Iteration number, defaults to 1.

        @return: Generator.
        """
        for _ in range(epoch):
            yield self.run(n)


class ConstantSampler(Sampler):
    """
    A sampler generating constants.
    """
    def __init__(self, value: Tensor, requires_grad: bool=False, **kwargs) -> None:
        """
        @brief Build a sampler generats constants.

        @param value: A constant tensor.
        @param requires_grad: bool.
        """
        assert value.ndim == 2
        super().__init__(dtype=value.dtype, device=value.device,
                         requires_grad=requires_grad, **kwargs)
        self.value = value
        self.nd = value.shape[-1]
        if self.enable_weight:
            self._weight[:] = torch.tensor(0.0, dtype=self.dtype, device=value.device)

    def run(self, n: int) -> Tensor:
        ret = self.value.repeat(n)
        ret.requires_grad = self.requires_grad
        return ret


class ISampler(Sampler):
    """
    A sampler that generates samples independently in each axis.
    """
    def __init__(self, ranges: Any, mode: SampleMode='random', dtype=float64,
                 device: device=None, requires_grad: bool=False, **kwargs) -> None:
        """
        @brief Initializes an ISampler instance.

        @param ranges: An object that can be converted to a `numpy.ndarray`,\
               representing the ranges in each sampling axis.\
               For example, if sampling x in [0, 1] and y in [4, 5],\
               use `ranges=[[0, 1], [4, 5]]`, or `ranges=[0, 1, 4, 5]`.
        @param mode: 'random' or 'linspace'. Defaults to 'random'.
        @param dtype: Data type of samples. Defaults to `torch.float64`.
        @param requires_grad: A boolean indicating whether the samples should\
               require gradient computation. Defaults to `False`.\
               See `torch.autograd.grad`

        @throws ValueError: If `ranges` has an unexpected shape.
        """
        super().__init__(dtype=dtype, device=device, requires_grad=requires_grad,
                         **kwargs)
        ranges_arr = _as_tensor(ranges, dtype=dtype, device=device)

        if ranges_arr.ndim == 1:
            _, mod = divmod(ranges_arr.shape[0], 2)
            if mod != 0:
                raise ValueError("If `ranges` is 1-dimensional, its length is"
                                 f"expected to be even, but got {mod}.")
            ranges_arr = ranges_arr.reshape(-1, 2)
        assert ranges_arr.ndim == 2
        self.nd = ranges_arr.shape[0]
        self.nodes = ranges_arr # (GD, 2)
        self.mode = mode

    def run(self, *m: int) -> Tensor:
        """
        @brief Generates independent samples in each axis.

        @param *m: int. In 'random' mode, only one single int `m` is required, saying\
               the number of samples. In 'linspace' mode, number of `m` must match\
               the dimension, saying number of steps in each dimension.

        @return: A tensor with shape (#samples, GD) containing the generated samples.
        """
        if self.mode == 'random':
            ruler = torch.stack(
                [F.random_weights(m[0], 2, dtype=self.dtype, device=self.device)
                 for _ in range(self.nd)],
                dim=0
            ) # (GD, m, 2)
            ret = torch.einsum('db, dmb -> md', self.nodes, ruler)
        elif self.mode == 'linspace':
            assert len(m) == self.nd, "Length of `m` must match the dimension."
            ps = [torch.einsum(
                'b, mb -> m',
                self.nodes[i, :],
                F.linspace_weights(m[i], 2, dtype=self.dtype, device=self.device)
            ) for i in range(self.nd)]
            ret = torch.stack(torch.meshgrid(*ps, indexing='ij'), dim=-1).reshape(-1, self.nd)
        else:
            raise ValueError(f"Invalid sampling mode '{self.mode}'.")
        if self.enable_weight:
            self._weight[:] = 1/ret.shape[0]
            self._weight = self._weight.broadcast_to(ret.shape[0])
        ret.requires_grad_(self.requires_grad)
        return ret


class BoxBoundarySampler(Sampler):
    """Generate samples on the boundaries of a multidimensional rectangle."""
    def __init__(self, p1: Sequence[float], p2: Sequence[float], mode: SampleMode='random',
                 dtype=float64, device: device=None,
                 requires_grad: bool=False, **kwargs) -> None:
        """
        @brief Generate samples on the boundaries of a multidimensional rectangle.

        @param p1, p2: Object that can be converted to `torch.Tensor`.\
               Points at both ends of the diagonal.
        @param mode: 'random' or 'linspace'. Defaults to 'random'.
        @param dtype: Data type of samples. Defaults to `torch.float64`.
        @param requires_grad: bool. Defaults to `False`. See `torch.autograd.grad`.
        """
        super().__init__(dtype=dtype, device=device, requires_grad=requires_grad,
                         **kwargs)
        t1 = _as_tensor(p1, dtype=dtype, device=device)
        t2 = _as_tensor(p2, dtype=dtype, device=device)
        if len(t1.shape) != 1:
            raise ValueError
        if t1.shape != t2.shape:
            raise ValueError("p1 and p2 should be in a same shape.")
        self.nd = int(t1.shape[0])
        self.mode = mode

        # NOTE: the data is like
        # [[x_min, x_max]
        #  [y_min, y_max]
        #  [z_min, z_max]]
        data = torch.vstack([t1, t2]).T

        self.subs: List[ISampler] = []

        for d in range(t1.shape[0]):
            range1, range2 = data.clone(), data.clone()
            range1[d, 1] = data[d, 0]
            range2[d, 0] = data[d, 1]
            self.subs.append(ISampler(ranges=range1, mode=mode, dtype=dtype,
                              device=device, requires_grad=requires_grad))
            self.subs.append(ISampler(ranges=range2, mode=mode, dtype=dtype,
                              device=device, requires_grad=requires_grad))

    def run(self, *mb: int, bd_type=False) -> Tensor:
        """
        @brief Generate samples on the boundaries of a multidimensional rectangle.

        @param *mb: int. Number of `mb` must match the dimension. In 'random' mode,\
               these are numbers of sample points on the boundary perpendicular\
               to each dimension. In 'linspace' mode, these are numbers of steps\
               in each dimension.
        @param bd_type: bool. Separate samples in each boundary if `True`, and\
               the output shape will be (#boundaries, #samples, #dims).\
               Defaults to `False`.

        @return: Tensor.
        """
        assert len(mb) * 2 == len(self.subs)
        results: List[Tensor] = []

        if self.mode == 'linspace':
            for idx, m in enumerate(mb):
                mb_proj = list(mb)
                mb_proj[idx] = 1
                results.append(self.subs[idx*2].run(*mb_proj))
                results.append(self.subs[idx*2+1].run(*mb_proj))

        elif self.mode == 'random':
            for idx, m in enumerate(mb):
                results.append(self.subs[idx*2].run(m))
                results.append(self.subs[idx*2+1].run(m))

        else:
            raise ValueError(f"Invalid sampling mode '{self.mode}'.")

        if bd_type:
            return torch.stack(results, dim=0)
        return torch.cat(results, dim=0)


##################################################
### Mesh samplers
##################################################

from ..nntyping import S
EType = Literal['cell', 'face', 'edge', 'node']

class MeshSampler(Sampler):
    """
    Sample in the specified entity of a mesh.
    """

    DIRECTOR: Dict[Tuple[Optional[str], Optional[str]], Type['MeshSampler']] = {}

    def __new__(cls, mesh, etype: EType, index=S,
                mode: Literal['random', 'linspace']='random',
                dtype=float64, device: device=None,
                requires_grad: bool=False):
        mesh_name = mesh.__class__.__name__
        ms_class = cls._get_sampler_class(mesh_name, etype)
        return object.__new__(ms_class)

    @classmethod
    def _assigned(cls, mesh_name: Optional[str], etype: Optional[str]='cell'):
        if (mesh_name, etype) in cls.DIRECTOR.keys():
            if mesh_name is None:
                mesh_name = "all types of mesh"
            if etype is None:
                etype = "entitie"
            raise KeyError(f"{etype}s in {mesh_name} has already assigned to "
                           "another mesh sampler.")
        cls.DIRECTOR[(mesh_name, etype)] = cls

    @classmethod
    def _get_sampler_class(cls, mesh_name: str, etype: EType):
        if etype not in {'cell', 'face', 'edge', 'node'}:
            raise ValueError(f"Invalid etity type name '{etype}'.")
        ms_class = cls.DIRECTOR.get((mesh_name, etype), None)
        if ms_class is None:
            ms_class = cls.DIRECTOR.get((mesh_name, None), None)
            if ms_class is None:
                ms_class = cls.DIRECTOR.get((None, etype), None)
                if ms_class is None:
                    raise NotImplementedError(f"Sampler for {mesh_name}'s {etype} "
                                              "has not been implemented.")
        return ms_class

    def __init__(self, mesh, etype: EType, index=S,
                 mode: Literal['random', 'linspace']='random',
                 dtype=float64, device: device=None,
                 requires_grad: bool=False, **kwargs) -> None:
        """
        @brief Generate samples in the specified entities of a mesh.

        @param mesh: Mesh.
        @param etype: 'cell', 'face' or 'edge'. Type of entity to sample from.
        @param index: Index of entities to sample from.
        @param mode: 'random' or 'linspace'.
        @param dtype: Data type of samples. Defaults to `torch.float64`.
        @param requires_grad: bool. Defaults to `False`. See `torch.autograd.grad`.
        """
        self.etype = etype
        self.node = torch.tensor(mesh.entity('node'), dtype=dtype, device=device)
        self.nd = self.node.shape[-1]
        self.node = self.node.reshape(-1, self.nd)
        try:
            if etype == 'node':
                self.cell = torch.arange(self.node.shape[0])[index].unsqueeze(-1)
            else:
                self.cell = torch.tensor(mesh.entity(etype, index=index), device=device)
        except TypeError:
            warn(f"{mesh.__class__.__name__}.entity() does not support the 'index' "
                 "parameter. The entity is sliced after returned.")
            self.cell = torch.tensor(mesh.entity(etype)[index, :], device=device)
        self.NVC: int = self.cell.shape[-1]
        self.mode = mode

        super().__init__(dtype=dtype, device=device, requires_grad=requires_grad,
                         **kwargs)

    # def _set_weight(self, mp: int) -> None:
    #     raw = self.mesh.entity_measure(etype=self.etype)
    #     raw /= mp * np.sum(raw, axis=0)
    #     if isinstance(raw, (float, int)):
    #         arr = torch.tensor([raw, ], dtype=self.dtype).broadcast_to(self.cell.shape[0], 1)
    #     elif isinstance(raw, np.ndarray):
    #         arr = torch.from_numpy(raw)[:, None]
    #     else:
    #         raise TypeError(f"Unsupported return from entity_measure method.")
    #     self._weight = arr.repeat(1, mp).reshape(-1, 1).to(device=self.device)

    def get_bcs(self, mp: int, n: int):
        """
        @brief Generate bcs according to the current mode.

        `mp` is the number of samples in 'random' mode, and is the order of\
        multiple indices in 'linspace' mode.
        """
        if self.mode == 'random':
            return F.random_weights(mp, n, dtype=self.dtype, device=self.device)
        elif self.mode == 'linspace':
            return F.linspace_weights(mp, n, dtype=self.dtype, device=self.device)
        else:
            raise ValueError(f"Invalid mode {self.mode}.")

    def cell_bc_to_point(self, bcs: Tensor) -> Tensor:
        """
        The optimized version of method `mesh.cell_bc_to_point()`
        to support faster sampling.
        """
        node = self.node
        cell = self.cell
        return torch.einsum('...j, ijk->...ik', bcs, node[cell])


class _PolytopeSampler(MeshSampler):
    """Sampler in all homogeneous polytope entities, such as triangle cells\
        and tetrahedron cells."""
    def run(self, mp: int) -> Tensor:
        self.bcs = self.get_bcs(mp, self.NVC)
        return self.cell_bc_to_point(self.bcs).reshape((-1, self.nd))

_PolytopeSampler._assigned(None, 'edge')
_PolytopeSampler._assigned('IntervalMesh', 'cell')
_PolytopeSampler._assigned('TriangleMesh', None)
_PolytopeSampler._assigned('TetrahedronMesh', None)
_PolytopeSampler._assigned('QuadrangleMesh', 'face')
_PolytopeSampler._assigned('PolygonMesh', 'face')


class _QuadSampler(MeshSampler):
    """Sampler in a quadrangle entity."""
    def run(self, mp: int) -> Tensor:
        bc_0 = self.get_bcs(mp, 2)
        bc_1 = self.get_bcs(mp, 2)
        if self.mode == 'linspace':
            self.bcs = F.multiply(bc_0, bc_1, mode='cross', order=[0, 2, 3, 1])
        else:
            self.bcs = F.multiply(bc_0, bc_1, mode='dot', order=[0, 2, 3, 1])
        return self.cell_bc_to_point(self.bcs).reshape((-1, self.nd))

_QuadSampler._assigned('QuadrangleMesh', 'cell')
_QuadSampler._assigned('HexahedronMesh', 'face')


class _UniformSampler(MeshSampler):
    """Sampler in a n-d uniform mesh."""
    def run(self, mp: int, *, entity_type=False) -> Tensor:
        ND = int(log2(self.NVC))
        bc_list = [self.get_bcs(mp, 2) for _ in range(ND)]
        if self.mode == 'linspace':
            self.bcs = F.multiply(*bc_list, mode='cross')
        else:
            self.bcs = F.multiply(*bc_list, mode='dot')
        ret = self.cell_bc_to_point(self.bcs)
        if entity_type:
            return ret
        return ret.reshape((-1, self.nd))

_UniformSampler._assigned('UniformMesh1d', None)
_UniformSampler._assigned('UniformMesh2d', None)
_UniformSampler._assigned('UniformMesh3d', None)
