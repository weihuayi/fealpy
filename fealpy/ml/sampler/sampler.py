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
    """Convert a sequence to a tensor with specified dtype and device.
    
    Parameters:
        __sequence: Sequence
            Input sequence to convert (can be numpy array, list, or existing tensor)
        dtype: torch.dtype, optional
            Target data type (default: float64)
        device: torch.device, optional
            Target device (default: None for CPU)
            
    Returns:
        Tensor: Converted tensor with specified properties
    """
    seq = __sequence
    if isinstance(seq, Tensor):
        return seq.detach().clone().to(device=device).to(dtype=dtype)
    else:
        return torch.tensor(seq, dtype=dtype, device=device)


class Sampler():
    """Base class for all sampling implementations.
    
    Provides common interface and functionality for generating sample points in various
    domains and distributions. Subclasses should implement the actual sampling logic.
    
    Attributes:
        nd: int
            Number of dimensions in sampled points
        _weight: Tensor
            Storage for sample weights (reciprocal of sampling density)
        enable_weight: bool
            Whether to compute sample weights
        dtype: torch.dtype
            Data type for samples
        device: torch.device
            Device for sample storage
        requires_grad: bool
            Whether samples require gradient computation
    """
    nd: int = 0
    _weight: Tensor
    def __init__(self, enable_weight=False,
                 dtype=float64, device: device=None,
                 requires_grad: bool=False, **kwargs) -> None:
        self.enable_weight = enable_weight
        self.dtype = dtype
        self.device = device
        self.requires_grad = bool(requires_grad)
        self._weight = torch.tensor(torch.nan, dtype=dtype, device=device)

    def run(self, n: int) -> Tensor:
        """Generate samples (must be implemented by subclasses).
        
        Parameters:
            n: int
                Number of samples to generate
                
        Returns:
            Tensor: Generated samples with shape (n, nd)
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

    def weight(self) -> Tensor:
        """Get weights of latest generated samples.
        
        Returns:
            Tensor: Sample weights with shape (m, 1)
        """
        return self._weight

    def load(self, n: int, epoch: int=1) -> Generator[torch.Tensor, None, None]:
        """Create a generator that yields samples over multiple epochs.
        
        Parameters:
            n: int
                Samples per epoch
            epoch: int, optional
                Number of epochs (default: 1)
                
        Returns:
            Generator: Yields sample tensors for each epoch
        """
        for _ in range(epoch):
            yield self.run(n)


class ConstantSampler(Sampler):
    """Sampler that generates constant values.
    
    Attributes:
        value: Tensor
            Constant value to repeat
    """
    def __init__(self, value: Tensor, requires_grad: bool=False, **kwargs) -> None:
        assert value.ndim == 2
        super().__init__(dtype=value.dtype, device=value.device,
                         requires_grad=requires_grad, **kwargs)
        self.value = value
        self.nd = value.shape[-1]
        if self.enable_weight:
            self._weight[:] = torch.tensor(0.0, dtype=self.dtype, device=value.device)

    def run(self, n: int) -> Tensor:
        """Generate constant samples by repeating input value.
        
        Parameters:
            n: int
                Number of repetitions
                
        Returns:
            Tensor: Repeated values with shape (n, nd)
        """
        ret = self.value.repeat(n)
        ret.requires_grad = self.requires_grad
        return ret


class ISampler(Sampler):
    """Independent axis sampler for hyperrectangular domains.
    
    Samples each dimension independently according to specified ranges and mode.
    
    Attributes:
        nodes: Tensor
            Sampling ranges for each dimension (shape [GD, 2])
        mode: SampleMode
            Sampling mode ('random' or 'linspace')
    """
    def __init__(self, ranges: Any, mode: SampleMode='random', dtype=float64,
                 device: device=None, requires_grad: bool=False, **kwargs) -> None:
        
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
        """Generate independent samples in each dimension.
        
        Parameters:
            *m: int
                In 'random' mode: single integer for number of samples
                In 'linspace' mode: one integer per dimension for steps
                
        Returns:
            Tensor: Generated samples with shape (#samples, GD)
            
        Raises:
            ValueError: For invalid sampling mode or dimension mismatch
        """
        if self.mode == 'random':
            ruler = torch.stack(
                [F.random_weights(m[0], 2, dtype=self.dtype, device=self.device)
                 for _ in range(self.nd)],
                dim=0
            ) # (GD, m, 2)
            ret = torch.einsum('db, dmb -> md', self.nodes, ruler)
        elif self.mode == 'linspace':
            if len(m) == 1:
                m *= self.nd
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
    """
    A sampler class that generates samples on the boundaries of a multidimensional rectangle.

    This class provides functionality to sample points on all boundaries of a hyperrectangle
    defined by two diagonal points, supporting both random and uniform sampling modes.

    Parameters:
        *args: Either a single sequence of domain (e.g., [x_min, x_max, y_min, y_max, ...])
               or two separate sequences representing diagonal points (p1=[x_min, y_min, ...], p2=[x_max, y_max, ...])
        p1: Sequence of floats representing the first diagonal point coordinates.
        p2: Sequence of floats representing the opposite diagonal point coordinates.
        mode: Sampling mode ('random' or 'linspace'). Defaults to 'random'.
        dtype: Data type for the samples('None'('cpu') or 'cuda'). Defaults to torch.float64.
        device: Device to store the samples. Defaults to None.
        requires_grad: Whether the samples require gradient computation. Defaults to False.
    
    Example 1:
        >>> domain = (-1, 1, -1, 1, -1, 1)
        >>> p1, p2 = (-1, -1, -1) , (1, 1, 1)
        >>> bc_sampler = BoxBoundarySampler(p1, p2, mode='random', 
                                dtype=bm.float64, device=None, 
                                requires_grad=True) 
        >>> bc_point = bc_sampler.run(2, 2, 2, bd_type=False)
    Example 2:
        >>> domain = (-1, 1, -1, 1, -1, 1)
        >>> bc_sampler = BoxBoundarySampler(domain, mode='random', 
                                dtype=bm.float64, device=None, 
                                requires_grad=True) 
        >>> bc_point = bc_sampler.run(2, 2, 2, bd_type=False)
    """
    def __init__(self, *args: Sequence[float], mode: SampleMode='random',
                 dtype=float64, device: device=None,
                 requires_grad: bool=False, **kwargs) -> None:
    
        super().__init__(dtype=dtype, device=device, requires_grad=requires_grad,
                         **kwargs)
        
        if len(args) == 1:
            p1 = args[0][0::2] 
            p2 = args[0][1::2] 
        elif len(args) == 2:
            p1, p2 = args
        else:
            raise ValueError( f"Expected 1 argument (domain) or 2 arguments (p1, p2) for *args, but got {len(args)}")
        t1 = _as_tensor(p1, dtype=dtype, device=device)
        t2 = _as_tensor(p2, dtype=dtype, device=device)
        if len(t1.shape) != 1:
            raise ValueError
        if t1.shape != t2.shape:
            raise ValueError("p1 and p2 should be in a same shape.")
        self.nd = int(t1.shape[0])
        self.mode = mode
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
            Generate samples on the boundaries of the multidimensional rectangle.

            Parameters:
                *mb: Variable number of integers specifying sample counts per dimension,these integers must be equal.
                    In 'random' mode: number of samples per boundary.
                    In 'linspace' mode: number of steps per dimension.
                bd_type: If True, returns samples separated by boundary type with shape
                        (boundaries, samples, dims). If False, returns concatenated
                        samples with shape (total_samples, dims). Defaults to False.

            Returns:
                Tensor: The generated samples, either concatenated or separated by boundary.
        """
        if len(mb) == 1:
            mb *= self.nd
        # assert len(mb) * 2 == len(self.subs)
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
    """Abstract base class for mesh-based sampling.
    
    Provides infrastructure for sampling points within mesh entities (cells, faces, edges, nodes)
    using a factory pattern to select appropriate implementations based on mesh type.
    
    Attributes:
        DIRECTOR: Dict[Tuple[Optional[str], Optional[str]], Type['MeshSampler']]
            Registry mapping (mesh_type, entity_type) to sampler implementations
        etype: EType
            Entity type being sampled
        node: Tensor
            Mesh nodes coordinates
        cell: Tensor
            Mesh entity connectivity
        NVC: int
            Number of vertices per entity
        mode: Literal['random', 'linspace']
            Sampling mode
    """

    DIRECTOR: Dict[Tuple[Optional[str], Optional[str]], Type['MeshSampler']] = {}

    def __new__(cls, mesh, etype: EType, index=S,
                mode: Literal['random', 'linspace']='random',
                dtype=float64, device: device=None,
                requires_grad: bool=False):
        """Factory method to create appropriate sampler instance.
        
        Parameters:
            mesh: Any
                Mesh object to sample from
            etype: EType
                Entity type to sample ('cell', 'face', 'edge', or 'node')
            index: Any, optional
                Entity indices to sample (default: all)
            mode: Literal['random', 'linspace'], optional
                Sampling mode (default: 'random')
            dtype: torch.dtype, optional
                Sample data type (default: float64)
            device: torch.device, optional
                Sample storage device (default: None for CPU)
            requires_grad: bool, optional
                Whether samples require gradients (default: False)
                
        Returns:
            MeshSampler: Appropriate sampler instance for given mesh and entity type
        """
        mesh_name = mesh.__class__.__name__
        ms_class = cls._get_sampler_class(mesh_name, etype)
        return object.__new__(ms_class)

    @classmethod
    def _assigned(cls, mesh_name: Optional[str], etype: Optional[str]='cell'):
        """Register sampler implementation for specific mesh and entity types.
        
        Parameters:
            mesh_name: Optional[str]
                Mesh class name or None for all meshes
            etype: Optional[str]
                Entity type or None for all entities
                
        Raises:
            KeyError: If mapping already exists
        """
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
        """Look up appropriate sampler class for given mesh and entity type.
        
        Parameters:
            mesh_name: str
                Mesh class name
            etype: EType
                Entity type to sample
                
        Returns:
            Type[MeshSampler]: Sampler implementation class
            
        Raises:
            ValueError: For invalid entity type
            NotImplementedError: If no suitable implementation found
        """
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
        """Initialize mesh sampler base functionality.
        
        Parameters:
            mesh: Any
                Mesh object to sample from
            etype: EType
                Entity type to sample
            index: Any, optional
                Entity indices to sample (default: all)
            mode: Literal['random', 'linspace'], optional
                Sampling mode (default: 'random')
            dtype: torch.dtype, optional
                Sample data type (default: float64)
            device: torch.device, optional
                Sample storage device (default: None for CPU)
            requires_grad: bool, optional
                Whether samples require gradients (default: False)
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
        """Generate barycentric coordinates according to current sampling mode.
        
        Parameters:
            mp: int
                In 'random' mode: number of samples
                In 'linspace' mode: order of indices
            n: int
                Number of barycentric coordinates to generate
                
        Returns:
            Tensor: Generated coordinates
            
        Raises:
            ValueError: For invalid sampling mode
        """
        if self.mode == 'random':
            return F.random_weights(mp, n, dtype=self.dtype, device=self.device)
        elif self.mode == 'linspace':
            return F.linspace_weights(mp, n, dtype=self.dtype, device=self.device)
        else:
            raise ValueError(f"Invalid mode {self.mode}.")

    def cell_bc_to_point(self, bcs: Tensor) -> Tensor:
        """Convert barycentric coordinates to physical points.
        
        Optimized version of mesh.cell_bc_to_point() for faster sampling.
        
        Parameters:
            bcs: Tensor
                Barycentric coordinates
                
        Returns:
            Tensor: Physical point coordinates
        """
        node = self.node
        cell = self.cell
        return torch.einsum('...j, ijk->...ik', bcs, node[cell])


class _PolytopeSampler(MeshSampler):
    """Sampler in all homogeneous polytope entities, such as triangle cells\
        and tetrahedron cells."""
    def run(self, mp: int) -> Tensor:
        """Generate samples in polytope entities.
        
        Parameters:
            mp: int
                Number of samples per entity
                
        Returns:
            Tensor: Generated samples with shape (total_samples, nd)
        """
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
        """Generate samples in quadrangle entities.
        
        Parameters:
            mp: int
                Number of samples per entity
                
        Returns:
            Tensor: Generated samples with shape (total_samples, nd)
        """
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
        """Generate samples in uniform mesh entities.
        
        Parameters:
            mp: int
                Number of samples per entity
            entity_type: bool, optional
                If True, returns samples organized by entity (default: False)
                
        Returns:
            Tensor: Generated samples with shape:
                   - (total_samples, nd) if entity_type=False
                   - (entities, samples, nd) if entity_type=True
        """
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
