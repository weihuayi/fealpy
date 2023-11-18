"""
Partitions of Units
"""
from typing import (
    Union, List, Callable, Any, Generic, TypeVar, Tuple, Literal, Optional,
)

import numpy as np
import torch
from torch.nn import Module, Parameter
from torch import Tensor, sin, cos, einsum

from ..nntyping import S, deprecated
from .module import TensorMapping
from .linear import Standardize

PI = torch.pi


class _PoU_Fn(Module):
    """
    Base class for PoU functions. These functions apply on each dimension of inputs,
    and in the PoU module, the results will be multiplied in the last dim.

    @note: works on singleton mode.
    """
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def forward(self, x: Tensor):
        raise NotImplementedError

    def flag(self, x: Tensor):
        """
        @brief Support set flag. Return `True` if inside the support set, element-wise.
        """
        raise NotImplementedError

    def dn(self, x: Tensor, order: int) -> Tensor:
        """
        @brief order-specified derivative
        """
        if order == 0:
            return self.forward(x)
        elif order >= 1:
            fn = getattr(self, 'd'+str(order), None)
            if fn is None:
                raise NotImplementedError(f"{order}-order derivative has not been implemented.")
            else:
                return fn(x)
        else:
            raise ValueError(f"order can not be negative, but got {order}.")

    def d1(self, x: Tensor):
        """
        @brief 1-order derivative
        """
        raise NotImplementedError

    def d2(self, x: Tensor):
        """
        @brief 2-order derivative
        """
        raise NotImplementedError

    def d3(self, x: Tensor):
        """
        @brief 3-order derivative
        """
        raise NotImplementedError

    def d4(self, x: Tensor):
        """
        @brief 4-order derivative
        """
        raise NotImplementedError


class PoU(TensorMapping):
    """
    Base class of non-negative functions for Partitions of Unity (PoU),\
    usually denoted as $\psi$. This base class using the indicator function of\
    each partition.
    """
    def __init__(self, keepdim=True) -> None:
        super().__init__()
        self.keepdim = keepdim

    def forward(self, x: Tensor): # (..., d) -> (..., 1)
        flag = (x >= -1-1e-16) * (x < 1-1e-16)
        flag = torch.prod(flag, dim=-1, keepdim=self.keepdim)
        return flag.to(dtype=x.dtype)
        # Here we cast the data type after the operation for lower memory usage.

    def flag(self, x: Tensor):
        """
        @brief Return a bool tensor with shape (N, ) showing if samples are in\
               the supporting area.
        """
        flag = (x >= -1-1e-16) * (x < 1-1e-16)
        return torch.prod(flag, dim=-1, dtype=torch.bool)

    def gradient(self, x: Tensor):
        """
        @brief Return gradient vector of the function, with shape (N, GD).
        """
        N, GD = x.shape[0], x.shape[-1]
        return torch.tensor(0, dtype=x.dtype, device=x.device).broadcast_to(N, GD)

    def hessian(self, x: Tensor):
        """
        @brief Return hessian matrix of the function, with shape (N, GD, GD).
        """
        N, GD = x.shape[0], x.shape[-1]
        return torch.tensor(0, dtype=x.dtype, device=x.device).broadcast_to(N, GD, GD)

    def derivative(self, x: Tensor, *idx: int):
        """
        @brief Return derivatives.
        """
        order = len(idx)
        if order == 0:
            return self.forward(x)
        else:
            N, _ = x.shape[0], x.shape[-1]
            return torch.tensor(0, dtype=x.dtype, device=x.device).broadcast_to(N, 1)

PoUA = PoU
POU_DEFAULT = PoU()

### Sin-style PoU function & module

class _PoU_Sin_Fn(_PoU_Fn):
    def forward(self, x: Tensor):
        f1 = (-1.25 <= x) * (x < -0.75)
        f2 = (-0.75 <= x) * (x < 0.75)
        f3 = (0.75 <= x) * (x < 1.25)
        l1 = 0.5 * (1 + sin(2*PI*x)) * f1
        l2 = f2.to(dtype=x.dtype)
        l3 = 0.5 * (1 - sin(2*PI*x)) * f3
        return l1 + l2 + l3

    def flag(self, x: Tensor):
        return (x >= -1.25) * (x <= 1.25)

    def d1(self, x: Tensor):
        f1 = (-1.25 <= x) * (x < -0.75)
        f3 = (0.75 <= x) * (x < 1.25)
        return PI * cos(2*PI*x) * f1 - PI * cos(2*PI*x) * f3

    def d2(self, x: Tensor):
        f1 = (-1.25 <= x) * (x < -0.75)
        f3 = (0.75 <= x) * (x < 1.25)
        return -2*PI**2 * sin(2*PI*x) * f1 + 2*PI**2 * sin(2*PI*x) * f3

    def d3(self, x: Tensor):
        f1 = (-1.25 <= x) * (x < -0.75)
        f3 = (0.75 <= x) * (x < 1.25)
        return -4*PI**3 * cos(2*PI*x) * f1 + 4*PI**3 * cos(2*PI*x) * f3

    def d4(self, x: Tensor):
        f1 = (-1.25 <= x) * (x < -0.75)
        f3 = (0.75 <= x) * (x < 1.25)
        return 8*PI**4 * sin(2*PI*x) * f1 - 8*PI**4 * sin(2*PI*x) * f3


class PoUSin(PoU):
    """
    @brief Sin-style partition of unity.

    For inputs with shape (..., GD), the output is like (..., ) or (..., 1),\
    and values of each element is between 0 and 1.
    """
    func = _PoU_Sin_Fn()
    def forward(self, x: Tensor): # (..., d) -> (..., 1)
        return torch.prod(self.func(x), dim=-1, keepdim=self.keepdim)

    def flag(self, x: Tensor):
        # As bool type is used in indexing, keepdim is set to False.
        return torch.prod(self.func.flag(x), dim=-1, dtype=torch.bool)

    def gradient(self, x: Tensor):
        pg = self.func.d1(x)
        p = self.func(x)
        GD = x.shape[-1]
        if x.ndim == 1:
            grad = torch.ones((GD, ), dtype=x.dtype)
        else:
            N = x.shape[0]
            grad = torch.ones((N, GD, ), dtype=x.dtype)
        for i in range(GD):
            element = torch.zeros_like(grad)
            element[:] = p[..., i][:, None]
            element[..., i] = pg[..., i]
            grad *= element
        return grad

    def hessian(self, x: Tensor):
        ph = self.func.d2(x)
        pg = self.func.d1(x)
        p = self.func(x)
        GD = x.shape[-1]
        if x.ndim == 1:
            hes = torch.ones((GD, GD), dtype=x.dtype)
        else:
            N = x.shape[0]
            hes = torch.ones((N, GD, GD), dtype=x.dtype)
        for i in range(GD):
            element = torch.zeros_like(hes)
            element[:] = p[..., i][:, None, None]
            element[..., i, :] = pg[..., i][:, None]
            element[..., :, i] = pg[..., i][:, None]
            element[..., i, i] = ph[..., i]
            hes *= element
        return hes

    def derivative(self, x: Tensor, *idx: int):
        N = x.shape[0]
        os = [0, ] * x.shape[-1]
        for i in idx:
            os[i] += 1

        ret = torch.ones((N, 1), dtype=x.dtype, device=x.device)
        for i in range(x.shape[-1]):
            ret *= self.func.dn(x[:, i:i+1], order=os[i])
        return ret


##################################################
### PoU in Spaces
##################################################

from .function_space import FunctionSpace

_FS = TypeVar('_FS', bound=FunctionSpace)

# NOTE: Partition of Unity has 3 main functions: LocalMapping, PoUFunctions and
# Assembling. We put LocalMapping and PoUFunction together in this PoULocalSpace
# class, because:
# (1) Some operator like laplace must be calculated before assembling, so
#     LocalMapping must be done in small space, instead of PoUSpace which
#     conducts Assembling.
# (2) It is obvious that the gradient of small space is the gradient of product
#     of PoU function and the original space. The same to hessian. This may contain
#     so many codes that very bad to put in Assembling.
class PoULocalSpace(FunctionSpace, Generic[_FS]):
    def __init__(self, pou_fn: PoU, space: _FS, idx: int,
                 center: Tensor, radius: Tensor) -> None:
        super().__init__()
        self.pou_fn = pou_fn
        self.space = space
        self.idx = idx
        assert center.ndim == 1
        assert radius.ndim == 1
        self.center = Parameter(center, requires_grad=False)
        self.radius = Parameter(radius, requires_grad=False)

    def global_to_local(self, p: Tensor):
        if p.ndim == 1:
            return (p - self.center)/self.radius
        else:
            return (p - self.center[None, :])/self.radius[None, :]

    def flag(self, p: Tensor):
        p = self.global_to_local(p)
        return self.pou_fn.flag(p)

    def number_of_basis(self) -> int:
        return self.space.number_of_basis()

    def basis(self, p: Tensor, *, index=S) -> Tensor:
        p = self.global_to_local(p)
        return self.space.basis(p, index=index) * self.pou_fn(p)

    def grad_basis(self, p: Tensor, *, index=S) -> Tensor:
        space = self.space
        p = self.global_to_local(p)
        ret = einsum("...d, ...f -> ...fd", self.pou_fn.gradient(p), space.basis(p, index=index))
        ret += self.pou_fn(p)[..., None] * space.grad_basis(p, index=index)
        return einsum('...d, d -> ...d', ret, 1/self.radius)

    def hessian_basis(self, p: Tensor, *, index=S) -> Tensor:
        space = self.space
        p = self.global_to_local(p)
        ret = einsum("...xy, ...f -> ...fxy", self.pou_fn.hessian(p), space.basis(p, index=index))
        cross = einsum("...x, ...fy -> ...fxy", self.pou_fn.gradient(p), space.grad_basis(p, index=index))
        ret += cross + torch.transpose(cross, -1, -2)
        ret += self.pou_fn(p)[..., None, None] * space.hessian_basis(p, index=index)
        return einsum('...xy, x, y -> ...xy', ret, 1/self.radius, 1/self.radius)

    def laplace_basis(self, p: Tensor, *, index=S) -> Tensor:
        space = self.space
        p = self.global_to_local(p)
        rs = 1/self.radius**2
        ret = einsum("...dd, ...f, d -> ...f", self.pou_fn.hessian(p),
                     space.basis(p, index=index), rs)
        ret += 2 * einsum("...d, ...fd, d -> ...f", self.pou_fn.gradient(p),
                          space.grad_basis(p, index=index), rs)
        ret += einsum("...fdd, d -> ...f",
                      self.pou_fn(p)[..., None, None] * space.hessian_basis(p, index=index),
                      rs)
        return ret

    def derivative_basis(self, p: Tensor, *idx: int, index=S) -> Tensor:
        nf = self.number_of_basis()
        order = len(idx)
        space = self.space
        p = self.global_to_local(p)
        rs = torch.prod(1/self.radius[idx,], dim=-1)
        if p.ndim == 1:
            ret = torch.zeros((nf, ), dtype=self.dtype, device=self.device)
        else:
            N = p.shape[0]
            ret = torch.zeros((N, nf), dtype=self.dtype, device=self.device)

        if order == 0:
            ret[:] = self.basis(p, index=index)
        elif order == 1:
            ret += self.pou_fn.derivative(p, idx[0]) * space.basis(p, index=index)
            ret += self.pou_fn(p) * space.derivative_basis(p, idx[0], index=index)
        elif order == 2:
            ret += self.pou_fn.derivative(p, idx[0], idx[1]) * space.basis(p, index=index)
            ret += self.pou_fn.derivative(p, idx[0]) * space.derivative_basis(p, idx[1], index=index)
            ret += self.pou_fn.derivative(p, idx[1]) * space.derivative_basis(p, idx[0], index=index)
            ret += self.pou_fn(p) * space.derivative_basis(p, idx[0], idx[1], index=index)
        elif order == 3:
            ret += self.pou_fn.derivative(p, idx[0], idx[1], idx[2]) * space.basis(p, index=index)
            ret += self.pou_fn.derivative(p, idx[0], idx[1]) * space.derivative_basis(p, idx[2], index=index)
            ret += self.pou_fn.derivative(p, idx[1], idx[2]) * space.derivative_basis(p, idx[0], index=index)
            ret += self.pou_fn.derivative(p, idx[2], idx[0]) * space.derivative_basis(p, idx[1], index=index)
            ret += self.pou_fn.derivative(p, idx[0]) * space.derivative_basis(p, idx[2], idx[1], index=index)
            ret += self.pou_fn.derivative(p, idx[1]) * space.derivative_basis(p, idx[0], idx[2], index=index)
            ret += self.pou_fn.derivative(p, idx[2]) * space.derivative_basis(p, idx[1], idx[0], index=index)
            ret += self.pou_fn(p) * space.derivative_basis(p, idx[0], idx[1], idx[2], index=index)

        elif order == 4:
            pass
        # TODO: finish this
        else:
            raise NotImplementedError("Derivatives higher than order 4 have bot been implemented.")
        return ret * rs


SpaceFactory = Callable[[int], _FS]
def assemble(dimension: int=0):
    """
    @brief Assemble data from partitions.

    The function that this decorator acts on must have at least two inputs:\
    the partition index number and the input tensor. And two outputs: the sample\
    index/flag and data.

    The functions acted upon by this decorator automatically collect the output of the\
    original function within the partition and assemble it into a total matrix.
    """
    def assemble_(func: Callable[..., Tensor]):
        def wrapper(self, p: Tensor, *args, **kwargs) -> Tensor:
            N = p.shape[0:-1]
            M = self.number_of_basis()
            GD = p.shape[-1]
            ret = torch.zeros(N + (M, ) + (GD, )*dimension,
                              dtype=self.dtype, device=self.device)
            basis_cursor = 0

            # NOTE: This works well for p with shape (N, GD) and (GD, ).
            for idx, part in enumerate(self.partitions):
                NF = part.number_of_basis()
                flag, data = func(self, idx, p, *args, **kwargs)
                ret[flag, basis_cursor:basis_cursor+NF, ...] += data
                basis_cursor += NF
            return ret
        return wrapper
    return assemble_


class PoUSpace(FunctionSpace, Generic[_FS]):
    """
    @brief Partitions of Unity
    """
    def __init__(self, space_factory: SpaceFactory[_FS], centers: Tensor,
                 radius: Union[Tensor, Any],
                 pou: Union[PoU, List[PoU], None]=None,
                 print_status=False) -> None:
        """
        @brief Construct a function space by sub-spaces, using Partitions of Unity.

        @param space_factory: (int -> FunctionSpace). A function returning\
               local spaces by their indices.
        @param centers: position Tensor with shape (#parts, GD).
        @param radius: Tensor, float, or sequence of floats.
        @param pou: PoU or list of PoU, optional. Continuous real valued functions.\
               Use the indicator functions of partitions by default (then continuous\
               conditions between partitions are required).
        """
        super().__init__(dtype=centers.dtype, device=centers.device)

        # NOTE: pou(s) are not necessary to add to sub-modules.
        if isinstance(pou, list):
            self.pou = pou
        else:
            pou_ = pou if pou else POU_DEFAULT
            self.pou = [pou_, ] * centers.shape[0]

        self.partitions: List[PoULocalSpace[_FS]] = []
        self.in_dim = -1
        self.out_dim = -1

        if not isinstance(radius, Tensor):
            rdata = torch.tensor(radius, dtype=centers.dtype, device=centers.device)
        else:
            rdata = radius.to(device=centers.device)
        radius = rdata.expand(centers.shape)

        for i in range(centers.shape[0]):
            part = PoULocalSpace(pou_fn=self.pou[i], space=space_factory(i), idx=i,
                                 center=centers[i, :], radius=radius[i, :])
            if self.in_dim == -1:
                self.in_dim = part.space.in_dim
                self.out_dim = part.space.out_dim
            else:
                if self.in_dim != part.space.in_dim:
                    raise RuntimeError("Can not group together local spaces with"
                                       "different input dimension.")
                if self.out_dim != part.space.out_dim:
                    raise RuntimeError("Can not group together local spaces with"
                                       "differnet output dimension.")

            self.partitions.append(part)
            self.add_module(f"part_{i}", part)

        if print_status:
            print(self.status_string)

    @property
    def status_string(self):
        return f"""PoU Space
#Partitions: {self.number_of_partitions()},
#Basis: {self.number_of_basis()}"""

    def number_of_partitions(self):
        return len(self.partitions)

    def number_of_basis(self):
        return sum(p.number_of_basis() for p in self.partitions)

    def partition_basis_slice(self, idx: int):
        """
        @brief Returns the index slice of the local basis function of the\
               specified partition in the global model.

        @param idx: int. The index of the partition.
        """
        assert idx >= 0
        assert idx < self.number_of_partitions()

        start = sum((p.number_of_basis() for p in self.partitions[:idx]), 0)
        stop = start + self.partitions[idx].number_of_basis()
        return slice(start, stop, None)

    @assemble(0)
    def basis(self, idx: int, p: Tensor, *, index=S) -> Tuple[Tensor, Tensor]:
        part = self.partitions[idx]
        flag = part.flag(p)
        return flag, self.partitions[idx].basis(p[flag, ...], index=index)

    @assemble(0)
    def convect_basis(self, idx: int, p: Tensor, *, coef: Tensor, index=S) -> Tuple[Tensor, Tensor]:
        part = self.partitions[idx]
        flag = part.flag(p)
        if coef.ndim == 1:
            coef = coef.broadcast_to(p.shape[0], coef.shape[-1])
        return flag, self.partitions[idx].convect_basis(p[flag, ...], coef=coef[flag, ...], index=index)

    @assemble(0)
    def laplace_basis(self, idx: int, p: Tensor, *, index=S) -> Tuple[Tensor, Tensor]:
        part = self.partitions[idx]
        flag = part.flag(p)
        return flag, self.partitions[idx].laplace_basis(p[flag, ...], index=index)

    @assemble(0)
    def derivative_basis(self, idx: int, p: Tensor, *dim: int, index=S) -> Tuple[Tensor, Tensor]:
        part = self.partitions[idx]
        flag = part.flag(p)
        return flag, self.partitions[idx].derivative_basis(p[flag, ...], *dim, index=index)

    @assemble(1)
    def grad_basis(self, idx: int, p: Tensor, *, index=S) -> Tuple[Tensor, Tensor]:
        part = self.partitions[idx]
        flag = part.flag(p)
        return flag, self.partitions[idx].grad_basis(p[flag, ...], index=index)

    @classmethod
    def from_uniform_mesh(cls, space_factory: SpaceFactory[_FS], uniform_mesh,
                          part_loc: Literal['node', 'cell'], pou: Optional[PoU]=None,
                          print_status=False, device=None):
        """
        @brief Construct PoU space from a uniform mesh.

        @param space_factory: A function generating spaces from indices.
        @param uniform_mesh: UniformMesh1d, 2d or 3d.
        @param part_loc: 'node' or 'cell'. Saying where the center of partition locates.
        @param pou: PoU, optional. Continuous real valued functions. Use the\
               indicator functions of partitions by default (then continuous\
               conditions between partitions are required).
        """
        if part_loc == 'node':
            ctrs = torch.from_numpy(uniform_mesh.entity('node'))
        elif part_loc == 'cell':
            ctrs = torch.from_numpy(uniform_mesh.entity_barycenter('cell'))
        else:
            raise ValueError(f"Invalid location center '{part_loc}'")
        length = torch.tensor(uniform_mesh.h, dtype=ctrs.dtype, device=device)
        return cls(space_factory, ctrs, length/2, pou, print_status)


class UniformPoUSpace(PoUSpace[_FS]):
    """
    Deprecated. Use `PoUSpace.from_uniform_mesh` instead.\n
    PoU space based on uniform mesh.
    """
    def __init__(self, space_factory: SpaceFactory[_FS], uniform_mesh,
                 part_loc: Literal['node', 'cell'], pou: PoUA,
                 print_status=False, device=None) -> None:
        """
        @brief Deprecated. Use `PoUSpace.from_uniform_mesh` instead.\
               Construct PoU space from a uniform mesh. Centers of partitions\
               are **nodes** of the mesh.

        @param space_factory: A function generating spaces from indices.
        @param uniform_mesh: UniformMesh1d, 2d or 3d.
        @param pou: PoU. The PoU function.
        @param part_loc: 'node' or 'cell'.
        """
        if part_loc == 'node':
            ctrs = torch.from_numpy(uniform_mesh.entity('node'))
        elif part_loc == 'cell':
            ctrs = torch.from_numpy(uniform_mesh.entity_barycenter('cell'))
        else:
            raise ValueError(f"Invalid location center '{part_loc}'")
        length = torch.tensor(uniform_mesh.h, dtype=ctrs.dtype, device=device)
        super().__init__(space_factory, ctrs, length/2, pou, print_status)
        self.mesh = uniform_mesh
        self.location = part_loc

    @deprecated("2.1.0", "sampler.InterfaceSampler")
    def collocate_sub_edge(self, n: int, edge_type=True):
        """
        @brief Collocate interior points in every sub-boundaries of partitions.\
               This is deprecated, use `sampler.InterfaceSampler` instead.

        @param n: int. The number of collocation points in each sub-boundary.
        @param edge_type: bool.

        @return: `Tensor` with shape (N, #Subs, #Dims) if `edge_type`,\
                 else (N*#Subs, #Dims).

        @note: Only works for 1-d boundary(edge).
        """
        GD = self.mesh.geo_dimension()
        location = self.location
        mesh = self.mesh
        node = mesh.entity('node')
        if location == 'node':
            edge = mesh.entity('edge')
            ctrsf = torch.from_numpy(mesh.entity_barycenter('edge'))
            R = np.array([[0.0, -mesh.h[1]/mesh.h[0]],
                          [mesh.h[0]/mesh.h[1], 0.0]], dtype=mesh.ftype)
            raw = np.einsum('nid, de -> nie', node[edge], R) #(NE, 2, GD)
        elif location == 'cell':
            _bd_flag = mesh.ds.boundary_edge_flag()
            edge = mesh.entity('edge')[~_bd_flag, :]
            ctrsf = torch.from_numpy(mesh.entity_barycenter('edge')[~_bd_flag, :])
            raw = node[edge] #(NE, 2, GD)
        else:
            raise RuntimeError("Invalid location center.")

        vec = torch.from_numpy(raw[:, 1, :] - raw[:, 0, :]) #(NE, GD)
        std_edge = Standardize(ctrsf, vec/2)
        loc = torch.stack((torch.linspace(-1, 1, n+2)[1:-1], )*2, dim=-1)
        points = std_edge.inverse(loc) #(N, NE, GD)
        if edge_type:
            return points
        else:
            return points.reshape(-1, GD)

    @deprecated("2.1.0", "sampler.InterfaceSampler")
    def sub_to_partition(self):
        """
        @brief Return the relationship between sub-boundaries and partitions.

        @return: `Tensor` with shape (#Subs, 2).
        """
        location = self.location
        if location == 'node':
            data = self.mesh.ds.face_to_node()
        elif location == 'cell':
            _bd_flag = self.mesh.ds.boundary_face_flag()
            data = self.mesh.ds.face_to_cell()[~_bd_flag, 0:2]
        else:
            raise RuntimeError("Invalid location center.")
        return torch.from_numpy(data).to(device=self.device)

    # NOTE: We get basis value from the inner space(without PoU function).
    # This continue matrix is mainly designed for PoUA.
    @deprecated("2.1.0", "operators.Continuous0")
    def continue_matrix_0(self, points: Tensor, return_sparse=False):
        """
        @brief Construct the matrix of 0th order continuity condition between\
               partitions(WITHOUT PoU function).

        @param points: Tensor. Collocation points in sub-boundaries, must be with\
               shape (N, #Subs, #Dims).

        @return: `Tensor` with shape (N*#Subs, #Basis).
        """
        assert points.ndim == 3 #(NVS, #Subs, #Dims)
        NVS = points.shape[0]
        NS = points.shape[1]
        N_basis = self.number_of_basis()
        sub2part = self.sub_to_partition()
        if return_sparse:
            from scipy.sparse import csr_matrix
            data = csr_matrix((NVS*NS, N_basis), dtype=points.cpu().detach().numpy().dtype)
        else:
            data = torch.zeros((NVS*NS, N_basis), dtype=self.dtype, device=self.device)
        for idx in range(NS):
            p = points[:, idx, :] #(NVS, #Dims)

            left_idx: int = sub2part[idx, 0].item()
            left_part = self.partitions[left_idx]
            basis_slice_l = self.partition_basis_slice(left_idx)
            left_data = left_part.space.basis(left_part.global_to_local(p))
            data[idx*NVS:(idx+1)*NVS, basis_slice_l] = left_data

            right_idx: int = sub2part[idx, 1].item()
            right_part = self.partitions[right_idx]
            basis_slice_r = self.partition_basis_slice(right_idx)
            right_data = right_part.space.basis(right_part.global_to_local(p))
            data[idx*NVS:(idx+1)*NVS, basis_slice_r] = -right_data
        return data

    @deprecated("2.1.0", "operators.Continuous1")
    def continue_matrix_1(self, points: Tensor, return_sparse=False):
        """
        @brief Construct the matrix of 1st order continuity condition between\
               partitions(WITHOUT PoU function).

        @param points: Tensor. Collocation points in sub-boundaries, must be with\
               shape (N, #Subs, #Dims).

        @return: `Tensor` with shape (2*N*#Subs, #Basis).
        """
        assert points.ndim == 3 #(NVS, #Subs, #Dims)
        NVS = points.shape[0]
        NS = points.shape[1]
        N_basis = self.number_of_basis()
        sub2part = self.sub_to_partition()
        if return_sparse:
            from scipy.sparse import csr_matrix
            data = csr_matrix((NVS*NS*2, N_basis), dtype=points.cpu().detach().numpy().dtype)
        else:
            data = torch.zeros((NVS*NS*2, N_basis), dtype=self.dtype, device=self.device)
        for idx in range(NS):
            p = points[:, idx, :] #(NVS, #Dims)

            left_idx: int = sub2part[idx, 0].item()
            left_part = self.partitions[left_idx]
            basis_slice_l = self.partition_basis_slice(left_idx)
            NPBL = left_part.number_of_basis()
            x = left_part.global_to_local(p)
            grad = torch.einsum('...fd, d -> ...fd', left_part.space.grad_basis(x),
                                1/left_part.radius)
            left_data = torch.swapdims(grad, 1, 2).reshape(-1, NPBL)
            data[idx*NVS*2:(idx+1)*NVS*2, basis_slice_l] = left_data

            right_idx: int = sub2part[idx, 1].item()
            right_part = self.partitions[right_idx]
            basis_slice_r = self.partition_basis_slice(right_idx)
            NPBR = right_part.number_of_basis()
            x = right_part.global_to_local(p)
            grad = torch.einsum('...fd, d -> ...fd', right_part.space.grad_basis(x),
                                1/right_part.radius)
            right_data = torch.swapdims(grad, 1, 2).reshape(-1, NPBR)
            data[idx*NVS*2:(idx+1)*NVS*2, basis_slice_r] = -right_data
        return data
