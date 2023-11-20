
import torch
from typing import Literal, overload, Tuple
from torch import device, float64, Tensor

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
        GD = mesh.geo_dimension()
        self.part_loc = part_loc
        if part_loc == 'node':
            mesh = self.cell_to_node(mesh)
        inner_face = ~mesh.ds.boundary_face_flag()
        self.mesh = mesh

        # NOTE: this is because `face` API in the 1-d uniform mesh is different
        # from that in the 2-d and 3-d uniform mesh.
        # `cell`, `face`, `edge` should be the structure of the entity. So `face`
        # of 1-d uniform mesh should be array([[0], [1], ..., [NF]]), meaning that
        # each face is made of only one node. But actually the `face` of current
        # 1-d uniform mesh is just `node`, the positions of every nodes.
        if GD > 1:
            super().__init__(mesh=mesh, etype='face', index=inner_face, mode=mode,
                            dtype=dtype, device=device, requires_grad=requires_grad)
        # TODO: remove this else after 1-d uniform mesh having the correct `face`.
        else:
            super().__init__(mesh=mesh, etype='node', index=inner_face, mode=mode,
                            dtype=dtype, device=device, requires_grad=requires_grad)

    @staticmethod
    def cell_to_node(mesh):
        GD = mesh.geo_dimension()
        if GD > 1:
            extent = list(mesh.extent)
            h = list(mesh.h)
            origin = list(mesh.origin)
            for i in range(len(h)):
                extent[2*i + 1] += 1
                origin[i] -= h[i]/2
        else:
            extent = list(mesh.extent) # NOTE: this copies the list.
            h = mesh.h                 # NOTE: `h` is float in UniformMesh1d.
            origin = mesh.origin       # NOTE: float.
            extent[1] += 1
            origin -= h/2
        return mesh.__class__(extent, h, origin, itype=mesh.itype, ftype=mesh.ftype)

    @overload
    def sub_to_part(self) -> Tensor: ...
    @overload
    def sub_to_part(self, return_normal: Literal[False]) -> Tensor: ...
    @overload
    def sub_to_part(self, return_normal: Literal[True]) -> Tuple[Tensor, Tensor]: ...
    def sub_to_part(self, return_normal=False):
        """
        @brief Return the relationship between sub-boundaries and partitions.

        @param: return_normal: bool. Return unit normal of sub-boundries if `True`.\
                Defaults to `False`.

        @return: `Tensor` with shape (#Subs, 2). If `return_normal`, return a\
                 tuple of two tensors, with shape (#subs, 2) and (#subs, GD).
        """
        _bd_flag = self.mesh.ds.boundary_face_flag()
        data = self.mesh.ds.face_to_cell()[~_bd_flag, 0:2]
        sub2part = torch.from_numpy(data).to(device=self.device)
        if return_normal:
            sub2normal = self.part_direction(sub2part)
            return sub2part, sub2normal
        return sub2part

    # NOTE: design to use `sub_to_part` directly.
    def part_direction(self, sub_to_partition: Tensor) -> Tensor:
        """
        @brief Return the unit vector between two partition centers.

        @return: `Tensor` with shape (#Subs, GD).
        """
        GD = self.nd
        cell_ctr = torch.from_numpy(self.mesh.cell_barycenter()).reshape(-1, GD) # (NC, GD)
        raw = cell_ctr[sub_to_partition] # (NS, 2, GD)
        raw = raw[:, 1, :] - raw[:, 0, :] # (ND, GD)
        return raw / torch.norm(raw, dim=-1, keepdim=True)
