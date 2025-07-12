from typing import Optional
from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, CoefLike
from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..decorator.variantmethod import variantmethod
from fealpy.fem.integrator import LinearInt, OpInt, FaceInt, enable_cache
from .vector_decomposition import VectorDecomposition
from .gradient_reconstruct import GradientReconstruct

class ScalarCrossDiffusionIntegrator(LinearInt, OpInt, FaceInt):
    def __init__(self, uh, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        super().__init__()
        self.uh = uh
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched
        self.assembly.set(method)

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        index = self.index
        mesh = getattr(space, 'mesh', None)
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarMassIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")
        Tf = VectorDecomposition(mesh).old_tangential_vector_calculation()         # (NE, 2)
        grad_f = GradientReconstruct(mesh).old_reconstruct(self.uh)  # (NE, 2)
        cell_to_edge = mesh.cell_to_edge(index=index)
        return Tf, grad_f,cell_to_edge
        

    @variantmethod 
    def assembly(self, space: _FS) -> TensorLike:
        Tf, grad_f,cell_to_edge= self.fetch(space)
        # grad_f = grad_f[cell_to_edge]
        Cross_diffusion = bm.einsum('ijk,ijk->i', Tf, grad_f)[..., None]
        # print(f"Cross_diffusion: {Cross_diffusion}")
        return Cross_diffusion[:,0]


        
        