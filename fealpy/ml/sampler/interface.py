
from warnings import warn
import torch
from typing import Literal
from torch import device, float64

from ..nntyping import S
from .sampler import _UniformSampler

class InterfaceSampler(_UniformSampler):
    """
    """
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, uniform_mesh, part_loc: Literal['node', 'cell'],
                 mode: Literal['random', 'linspace']='random',
                 dtype=float64, device: device=None,
                 requires_grad: bool=False, **kwargs) -> None:
        """
        @brief Generate samples on the interfaces of partitions generating by\
               a uniform mesh.

        @param uniform_mesh: UniformMesh1/2/3d.
        @param part_loc: 'node' or 'cell'. Saying where the center of partitions locates.
        @param mode: 'random' or 'linspace'.
        @param dtype: Data type of samples. Defaults to `torch.float64`.
        @param requires_grad: bool. Defaults to `False`. See `torch.autograd.grad`.
        """
        mesh = uniform_mesh
        self.part_loc = part_loc
        if part_loc == 'node':
            mesh = self.cell_to_node(mesh)
        if part_loc == 'node':
            inner_face = ~mesh.ds.boundary_face_flag()
        else:
            inner_face = S
        self.mesh = mesh

        super().__init__(mesh=mesh, etype='face', index=inner_face, mode=mode,
                         dtype=dtype, device=device, requires_grad=requires_grad)

    @staticmethod
    def cell_to_node(mesh):
        extent = list(mesh.extent)
        h = list(mesh.h)
        origin = list(mesh.origin)
        for i in range(len(h)):
            extent[2*i + 1] += 1
            origin[i] -= h[i]/2
        return mesh.__class__(extent, h, origin, itype=mesh.itype, ftype=mesh.ftype)

    def sub_to_partition(self):
        """
        @brief Return the relationship between sub-boundaries and partitions.

        @return: `Tensor` with shape (#Subs, 2).
        """
        _bd_flag = self.mesh.ds.boundary_face_flag()
        data = self.mesh.ds.face_to_cell()[~_bd_flag, 0:2]
        return torch.from_numpy(data).to(device=self.device)
