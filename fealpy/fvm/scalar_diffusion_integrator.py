from typing import Optional
from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, CoefLike
from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..decorator.variantmethod import variantmethod
from ..fem.integrator import LinearInt, OpInt, FaceInt, enable_cache
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
        e, d = VectorDecomposition(mesh).centroid_vector_calculation()
        q = self.q
        qf = mesh.quadrature_formula(q, 'face') 
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index)
        return Sf, e, d, index, bcs,  phi
    
    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        Sf, e, d, index, bcs,phi = self.fetch(space)
        D = phi.shape[-1]
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        Sf_dot_Sf = bm.einsum('ij,ij->i', Sf, Sf)              
        e_dot_Sf = bm.einsum('ij,ij->i', e, Sf)                
        e_norm = bm.einsum('ij,ij->i', e, e)**0.5               
        # Ef_abs = (|Sf|^2 / (e·Sf)) * |e|
        Ef_abs = bm.einsum('i,i->i', Sf_dot_Sf / e_dot_Sf, e_norm)
        direction_matrix = bm.array([[1.0, -1.0], [-1.0, 1.0]], dtype=space.ftype)
        eye_D = bm.eye(D, dtype=space.ftype, device=bm.get_device(space))
        base_matrix = bm.einsum('ij,pq->ipjq', eye_D, direction_matrix).reshape(2*D, 2*D)
        local_matrix = bm.einsum('i,ab->iab', Ef_abs / d, base_matrix)
        return local_matrix