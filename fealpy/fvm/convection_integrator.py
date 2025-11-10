from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike
from fealpy.decorator.variantmethod import variantmethod

from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace.space import FunctionSpace as _FS

from fealpy.fem.integrator import LinearInt, OpInt, FaceInt, enable_cache

class ConvectionIntegrator(LinearInt, OpInt, FaceInt):
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
        phi = space.basis(bcs, index=index)
        return Sf, index, bcs, phi
    
    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        Sf, index, bcs, phi = self.fetch(space)
        D = phi.shape[-1]
        # val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        eye_D = bm.eye(D, dtype=space.ftype, device=bm.get_device(space))
        direction_matrix = bm.array([[0.5, 0.5], [-0.5, -0.5]])
        base_matrix = bm.einsum('ij,pq->ipjq', eye_D, direction_matrix).reshape(2*D, 2*D)
        if coef is None:
            coef = bm.stack([bm.ones_like(Sf[:,0]), bm.zeros_like(Sf[:,0])], axis=1)
        integrator  = bm.einsum('ij,ij->i', Sf, coef)
        result = bm.einsum("i,jk->ijk", integrator, base_matrix)  # (NE, 2, 2)
        
        return result
    