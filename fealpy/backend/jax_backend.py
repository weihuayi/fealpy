from typing import Optional, Union, Tuple, Any
from itertools import combinations_with_replacement
from functools import reduce, partial
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import config, jit

    config.update("jax_enable_x64", True)

except ImportError:
    raise ImportError("Name 'jax' cannot be imported. "
                      'Make sure JAX is installed before using '
                      'the JAX backend in FEALPy. '
                      'See https://github.com/google/jax for installation.')

from .base import BackendProxy, ATTRIBUTE_MAPPING, FUNCTION_MAPPING, TRANSFORMS_MAPPING

Array = jax.Array
_device = jax.Device

def _remove_device(func):
    def wrapper(*args, **kwargs):
        if 'device' in kwargs:
            kwargs.pop('device')
        return func(*args, **kwargs)
    return wrapper

class JAXBackend(BackendProxy, backend_name='jax'):
    DATA_CLASS = Array
    linalg = jnp.linalg
    random = jax.random

    @staticmethod
    def context(tensor):
        return {"dtype": tensor.dtype}

    @staticmethod
    def set_default_device(device: Union[str, _device]) -> None:
        jax.default_device = device

    @staticmethod
    def device_type(array: Array, /): return array.device.platform.lower()

    @staticmethod
    def device_index(array: Array, /): return array.device.id

    @staticmethod
    def get_device(tensor_like: Array, /): return tensor_like.device

    # TODO 
    @staticmethod
    def device_put(tensor_like: Array, /, device=None) -> Array:
        return tensor_like
        # return jax.device_put(tensor_like, device)

    @staticmethod
    def to_numpy(jax_array: Array, /) -> Any:
        return np.array(jax_array) 

    @staticmethod
    def from_numpy(numpy_array: np.ndarray, /) -> Any:
        """
        TODO:
            1. add support to `device` agument
        """
        return jax.device_put(numpy_array)

    @staticmethod
    def tolist(tensor: Array, /): return tensor.tolist()

    ### Tensor creation methods ###
    # NOTE: all copied

    ### Reduction methods ###
    # NOTE: all copied

    ### Unary methods ###
    # NOTE: all copied

    ### Binary methods ###
    # NOTE: all copied

    # NOTE 临时删除 device
    arange = staticmethod(_remove_device(jnp.arange))
    zeros = staticmethod(_remove_device(jnp.zeros))
    linspace = staticmethod(_remove_device(jnp.linspace))
    empty = staticmethod(_remove_device(jnp.empty))
    array = staticmethod(_remove_device(jnp.array))
    tensor = staticmethod(_remove_device(jnp.array))
    ones = staticmethod(_remove_device(jnp.ones))

    @staticmethod
    def set_at(x: Array, indices, val, /):
        return x.at[indices].set(val)

    @staticmethod
    def add_at(x: Array, indices, val, /):
        return x.at[indices].add(val)

    @staticmethod
    def index_add(x: Array, index, src, /, *, axis=0, alpha=1):
        indexing = [slice(None)] * x.ndim
        indexing[axis] = index
        return x.at[tuple(indexing)].add(alpha*src)

    @staticmethod
    def scatter(x, indices, val, /, *, axis=0):
        raise NotImplementedError

    @staticmethod
    def scatter_add(x, indices, val, /, *, axis=0):
        raise NotImplementedError

    @staticmethod
    def unique(a, return_index=False, return_inverse=False, return_counts=False, axis=0, **kwargs):
        """
        unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None) -> Tuple[Tensor, Tensor, Tensor]
        """
        b, index, inverse, counts = jnp.unique(a, return_index=True, 
                return_inverse=True,
                return_counts=True,
                axis=axis, **kwargs)
        any_return = return_index or return_inverse or return_counts
        if any_return:
            result = (b, )
        else:
            result = b

        if return_index:
            result += (index, )

        if return_inverse:
            inverse = inverse.reshape(-1)
            result += (inverse, )

        if return_counts:
            result += (counts, )

        return result

    @staticmethod
    def unique_all(x, axis=None, **kwargs):
        """
        unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None) -> Tuple[Tensor, Tensor, Tensor]
        """
        return jnp.unique(x, return_index=True, 
                return_inverse=True,
                return_counts=True,
                axis=axis, **kwargs)

    @staticmethod
    def unique_all_(x, axis=None, **kwargs):
        """
        unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None) -> Tuple[Tensor, Tensor, Tensor]
        """
        b, indices0, inverse, counts = jnp.unique(x, return_index=True, 
                return_inverse=True,
                return_counts=True,
                axis=axis, **kwargs)
        indices1 = jnp.zeros_like(indices0)
        idx = jnp.arange(inverse.shape[0], dtype=indices0.dtype)
        indices1 = indices1.at[inverse].set(idx)
        return b, indices0, indices1, inverse, counts

    @staticmethod
    def query_point(x, y, h, box_size, mask_self=True, periodic=[True, True, True]):
        from .jax.jax_md import space
        from .jax import partition
        from .jax.jax_md.partition import Sparse

        if not isinstance(periodic, list) or len(periodic) != 3 or not all(isinstance(p, bool) for p in periodic):
            raise TypeError("periodic type is：[bool, bool, bool]")
        displacement, shift = space.periodic(side=box_size)
       
        neighbor_fn = partition.neighbor_list(
            displacement,
            box_size,
            r_cutoff = jnp.array(h, dtype=jnp.float64),
            backend ="jaxmd_vmap",
            capacity_multiplier = 1,
            mask_self = not mask_self,
            format = Sparse,
            num_particles_max = x.shape[0],
            num_partitions = x.shape[0],
            pbc = periodic,
            )
        neighbor_list = neighbor_fn.allocate(x, num_particles=x.shape[0])
        neighbors, node_self = neighbor_list.idx

        return node_self, neighbors

    ### FEALPy functionals ###
    @staticmethod
    def multi_index_matrix(p: int, dim: int, *, dtype=None) -> Array:
        r"""Create a multi-index matrix."""
        dtype = dtype or jnp.int_
        sep = jnp.flip(jnp.array(
            tuple(combinations_with_replacement(range(p+1), dim)),
            dtype=jnp.int_
        ), axis=0)
        z = jnp.zeros((sep.shape[0], 1))
        p = jnp.broadcast_to(p, shape=(sep.shape[0], 1))
        raw = jnp.concatenate((z, sep, p), axis=1)
        return jnp.array(raw[:, 1:] - raw[:, :-1]).astype(dtype)

    @staticmethod
    @jit
    def edge_length(edge: Array, node: Array, *, out=None) -> Array:
        return jnp.linalg.norm(node[edge[:, 0]] - node[edge[:, 1]], axis=-1)

    @staticmethod
    @partial(jit, static_argnames=['unit'])
    def edge_normal(edge: Array, node: Array, unit=False, *, out=None) -> Array:
        points = node[edge, :]
        if points.shape[-1] != 2:
            raise ValueError("Only 2D meshes are supported.")
        edges = points[..., 1, :] - points[..., 0, :]
        if unit:
            edges /= jnp.linalg.norm(edges, axis=-1, keepdims=True)
        return jnp.stack([edges[..., 1], -edges[...,0]], axis=-1, out=out)    

    @staticmethod
    @partial(jit, static_argnames=['unit'])
    def edge_tangent(edge: Array, node: Array, unit=False, *, out=None) -> Array:
        edges = node[edge[:, 1], :] - node[edge[:, 0], :]
        if unit:
            l = jnp.linalg.norm(edges, axis=-1, keepdims=True)
            edges /= l
        return jnp.stack([edges[..., 0], edges[...,1]], axis=-1, out=out)    

    @staticmethod
    @jit
    def tensorprod(*tensors: Array) -> Array:
        num = len(tensors)
        NVC = reduce(lambda x, y: x * y.shape[-1], tensors, 1)
        desp1 = 'mnopq'
        desp2 = 'abcde'
        string = ", ".join([desp1[i]+desp2[i] for i in range(num)])
        string += " -> " + desp1[:num] + desp2[:num]
        return jnp.einsum(string, *tensors).reshape(-1, NVC)

    @classmethod
    def bc_to_points(cls, bcs: Union[Array, Tuple[Array, ...]], node: Array, entity: Array) -> Array:
        points = node[entity, :]

        if not isinstance(bcs, Array):
            bcs = cls.tensorprod(*bcs)
        return jnp.einsum('ijk, ...j -> i...k', points, bcs)

    @staticmethod
    @jit
    def barycenter(entity: Array, node: Array, loc: Optional[Array]=None) -> Array:
        return jnp.mean(node[entity, :], axis=1) # TODO: polygon mesh case

    @staticmethod
    @jit
    def simplex_measure(entity: Array, node: Array) -> Array:
        points = node[entity, :]
        TD = points.shape[-2] - 1
        if TD != points.shape[-1]:
            raise RuntimeError("The geometric dimension of points must be NVC-1"
                            "to form a simplex.")
        edges = points[..., 1:, :] - points[..., :-1, :]
        return jnp.linalg.det(edges)/jax.scipy.special.factorial(TD)

    @staticmethod
    @partial(jit, static_argnums=1)
    def wrapper_simplex_shape_function_kernel(bc: Array, p: int, mi:Array) -> Array:
        TD = bc.shape[-1] - 1
        c = jnp.arange(1, p+1, dtype=jnp.int_)
        P = 1.0/jnp.cumprod(c)
        t = jnp.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = jnp.ones(shape, dtype=bc.dtype)
        A = A.at[..., 1:, :].set(p*bc[..., None, :] - t.reshape(-1, 1))
        A = jnp.cumprod(A, axis=-2)
        A = A.at[..., 1:, :].set(A[..., 1:, :]*P.reshape(-1, 1))
        idx = jnp.arange(TD+1)
        phi = jnp.prod(A[..., mi, idx], axis=-1)
        return phi

    @classmethod
    def _simplex_shape_function_kernel(cls, bc: Array, p: int, mi: Optional[Array]=None) -> Array:
        if p == 1:
            return bc
        TD = bc.shape[-1] - 1
        if mi is None:
            mi = cls.multi_index_matrix(p, TD)
        return cls.wrapper_simplex_shape_function_kernel(bc, p, mi)

    '''
    @classmethod
    def _simplex_shape_function_kernel(cls, bc: Array, p: int, mi: Optional[Array]=None) -> Array:
        if p == 1:
            return bc
        TD = bc.shape[-1] - 1
        if mi is None:
            mi = cls.multi_index_matrix(p, TD)
        c = jnp.arange(1, p+1, dtype=jnp.int_)
        P = 1.0/jnp.cumprod(c)
        t = jnp.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = jnp.ones(shape, dtype=bc.dtype)
        A = A.at[..., 1:, :].set(p*bc[..., None, :] - t.reshape(-1, 1))
        A = jnp.cumprod(A, axis=-2)
        A = A.at[..., 1:, :].set(A[..., 1:, :]*P.reshape(-1, 1))
        idx = jnp.arange(TD+1)
        phi = jnp.prod(A[..., mi, idx], axis=-1)
        return phi
    '''
    @classmethod
    def simplex_shape_function(cls, bcs: Array, p: int, mi=None) -> Array:
        fn = jax.vmap(
            partial(cls._simplex_shape_function_kernel, p=p, mi=mi)
        )
        return fn(bcs)

    @classmethod
    def simplex_grad_shape_function(cls, bcs: Array, p: int, mi=None) -> Array:
        fn = jax.vmap(jax.jacfwd(
            partial(cls._simplex_shape_function_kernel, p=p, mi=mi)
        ))
        return fn(bcs)

    @classmethod
    def simplex_hess_shape_function(cls, bcs: Array, p: int, mi=None) -> Array:
        fn = jax.vmap(jax.jacrev(jax.jacfwd(
            partial(cls._simplex_shape_function_kernel, p=p, mi=mi)
        )))
        return fn(bcs)

    @staticmethod
    def tensor_measure(entity: Array, node: Array) -> Array:
        # TODO
        raise NotImplementedError

    # Interval Mesh
    # =============
    @staticmethod
    @jit
    def interval_grad_lambda(line: Array, node: Array) -> Array:
        points = node[line, :]
        v = points[..., 1, :] - points[..., 0, :] # (NC, GD)
        h2 = jnp.sum(v**2, axis=-1, keepdims=True)
        v /= h2
        return jnp.stack([-v, v], axis=-2)

    # Triangle Mesh
    # =============

    @staticmethod
    @jit
    def triangle_area_3d(tri: Array, node: Array) -> Array:
        points = node[tri, :]
        edge1 = points[..., 1, :] - points[..., 0, :]
        edge2 = points[..., 2, :] - points[..., 0, :]
        points = node[tri, :]
        cross_product = jnp.cross(edge1, edge2, axis=-1)
        area = 0.5 * jnp.linalg.norm(cross_product, axis=-1)
        return area

    @staticmethod
    def triangle_grad_lambda_2d(cell: Array, node: Array) -> Array:
        """grad_lambda function for the triangle mesh in 2D.

        Parameters:
            cell (Tensor[NC, 3]): Indices of vertices of triangles.\n
            node (Tensor[NN, 2]): Node coordinates.

        Returns:
            Tensor[..., 3, 2]:
        """
        e0 = node[cell[:, 2]] - node[cell[:, 1]]
        e1 = node[cell[:, 0]] - node[cell[:, 2]]
        e2 = node[cell[:, 1]] - node[cell[:, 0]]
        nv = jnp.linalg.det(jnp.stack([e0, e1], axis=-2)) # (...)
        e0 = jnp.flip(e0, axis=-1)
        e1 = jnp.flip(e1, axis=-1)
        e2 = jnp.flip(e2, axis=-1)
        result = jnp.stack([e0, e1, e2], axis=-2)
        result = result.at[..., 0].mul(-1)
        return result/(nv[..., None, None])

    @staticmethod
    def triangle_grad_lambda_3d(cell: Array, node: Array) -> Array:
        """
        Parameters:
            cell (Tensor[NC, 3]): Indices of vertices of triangles.\n
            node (Tensor[NN, 3]): Node coordinates.

        Returns:
            Tensor[..., 3, 3]:
        """
        e0 = node[cell[:, 2]] - node[cell[:, 1]]
        e1 = node[cell[:, 0]] - node[cell[:, 2]]
        e2 = node[cell[:, 1]] - node[cell[:, 0]]
        nv = jnp.cross(e0, e1, axis=-1) # (..., 3)
        length = jnp.linalg.norm(nv, axis=-1, keepdims=True) # (..., 1)
        n = nv/length
        return jnp.stack([
            jnp.cross(n, e0, axis=-1),
            jnp.cross(n, e1, axis=-1),
            jnp.cross(n, e2, axis=-1)
        ], axis=-2)/length[...,jnp.newaxis] # (..., 3, 3)

    # Quadrangle Mesh
    # =============
    @staticmethod
    def quadrangle_grad_lambda_2d(quad: Array, node: Array) -> Array:
        pass

    # Tetrahedron Mesh
    # =============
    @staticmethod
    def tetrahedron_grad_lambda_3d(tet: Array, node: Array, localFace: Array) -> Array:
        NC = tet.shape[0]
        Dlambda = jnp.zeros((NC, 4, 3), dtype=node.dtype)
        
        points = node[tet, :]
        TD = points.shape[-2] - 1
        if TD != points.shape[-1]:
            raise RuntimeError("The geometric dimension of points must be NVC-1"
                            "to form a simplex.")
        edges = points[..., 1:, :] - points[..., :-1, :]
        volume = jnp.linalg.det(edges)/jax.scipy.special.factorial(TD)
        
        for i in range(4):
            j, k, m = localFace[i]
            vjk = node[tet[:, k],:] - node[tet[:, j],:]
            vjm = node[tet[:, m],:] - node[tet[:, j],:]
            Dlambda = Dlambda.at[:, i, :].set(jnp.cross(vjm, vjk) / (6*volume.reshape(-1, 1)))
        return Dlambda

JAXBackend.attach_attributes(ATTRIBUTE_MAPPING, jnp)
function_mapping = FUNCTION_MAPPING.copy()
function_mapping.update(
    tensor='array',
    concat='concatenate'
    )
JAXBackend.attach_methods(function_mapping, jnp)
JAXBackend.attach_methods({'compile': 'jit'}, jax)
JAXBackend.attach_methods(TRANSFORMS_MAPPING, jax)
