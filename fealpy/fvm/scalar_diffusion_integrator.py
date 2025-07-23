from typing import Optional
from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, CoefLike
from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..decorator.variantmethod import variantmethod
from fealpy.fem.integrator import LinearInt, OpInt, FaceInt, enable_cache
from .vector_decomposition import VectorDecomposition

class ScalarDiffusionIntegrator(LinearInt, OpInt, FaceInt):
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
        mesh = getattr(space, 'mesh', None)
        return mesh.face_to_cell()[self.index][:,:2]

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
        e, d = VectorDecomposition(mesh).centroid_vector_calculation()
        q = self.q
        qf = mesh.quadrature_formula(q, 'face') 
        bcs, ws = qf.get_quadrature_points_and_weights()
        return Sf, e, d, index, bcs
    
    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        Sf, e, d, index, bcs = self.fetch(space)
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        Sf_dot_Sf = bm.einsum('ij,ij->i', Sf, Sf).reshape(-1, 1)  
        e_dot_Sf = bm.einsum('ij,ij->i', e, Sf).reshape(-1, 1)    
        e_norm = bm.linalg.norm(e, axis=-1, keepdims=True)  
        Ef_abs = (Sf_dot_Sf / e_dot_Sf) * e_norm  
        integrator = Ef_abs/ d  
        x = bm.stack([[1, -1], [-1, 1]])
        integrator = integrator.reshape(-1,1,1)*x
        return integrator