from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike
from fealpy.utils import process_coef_func
from fealpy.decorator.variantmethod import variantmethod

from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace.space import FunctionSpace as _FS

from fealpy.fem.integrator import LinearInt, OpInt, FaceInt, enable_cache

from .vector_decomposition import VectorDecomposition

class DivIntegrator(LinearInt, OpInt, FaceInt):
    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        super().__init__()
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched
        self.assembly.set(method)
        
    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.edge_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        index = self.index
        mesh = getattr(space, 'mesh', None)
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarMassIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")
        n = mesh.face_unit_normal(index=index)
        facemeasure = mesh.entity_measure('face', index=index)
        Sf = facemeasure[:, None] * n  # (NE, 2)
        q = self.q
        qf = mesh.quadrature_formula(q, 'face') 
        bcs, ws = qf.get_quadrature_points_and_weights()
        return Sf, index, bcs,  
    
    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        Sf, index, bcs = self.fetch(space)
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        base_block = bm.array([[0.5, 0.5], [-0.5, -0.5]])
        blocks = Sf[:, :, bm.newaxis, bm.newaxis] * base_block
        result = bm.zeros((Sf.shape[0], 2*Sf.shape[1], 2*Sf.shape[1]))
        for i in range(Sf.shape[1]):
            result[:, 2*i:2*(i+1), 2*i:2*(i+1)] = blocks[:, i]
        # print(result)

        return result