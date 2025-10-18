from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike
from fealpy.decorator import variantmethod

from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace.space import FunctionSpace as _FS

from fealpy.fem.integrator import LinearInt, OpInt, FaceInt, enable_cache

from .vector_decomposition import VectorDecomposition
from .gradient_reconstruct import GradientReconstruct

class ScalarCrossDiffusionIntegrator(LinearInt, OpInt, FaceInt):
    def __init__(self, uh, grad_f, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        super().__init__()
        self.uh = uh
        self.grad_f = grad_f
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
        Tf = VectorDecomposition(mesh).tangential_vector_calculation() # (NE, 2)
        edge_to_cell = mesh.edge_to_cell(index=index)[:,:2]
        NC = mesh.number_of_cells()
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights() 
        phi = space.basis(bcs, index=index) 
        return Tf,edge_to_cell,NC,phi
        

    @variantmethod 
    def assembly(self, space: _FS) -> TensorLike:
        Tf,edge_to_cell,NC,phi= self.fetch(space)
        D = phi.shape[-1]
        if D ==1:
            Cross_diffusion = bm.einsum('ij,ij->i', Tf, self.grad_f)
            NE = Cross_diffusion.shape[0]
            flux = bm.zeros((NE, 2))
            is_boundary = edge_to_cell[:, 0] == edge_to_cell[:, 1]
            is_internal = ~is_boundary
            flux[is_internal, 0] =  Cross_diffusion[is_internal]
            flux[is_internal, 1] = -Cross_diffusion[is_internal]
            flux[is_boundary, 0] = Cross_diffusion[is_boundary]
            result = bm.zeros((NC,))
            bm.add_at(result, edge_to_cell[:, 0], flux[:, 0])
            bm.add_at(result, edge_to_cell[:, 1], flux[:, 1])
            return result
        elif D == 2:
            grad_f_u = self.grad_f[:,0,:]
            grad_f_v = self.grad_f[:,1,:]
            is_boundary = edge_to_cell[:, 0] == edge_to_cell[:, 1]
            is_internal = ~is_boundary
            Cross_diffusion_u = bm.einsum('ij,ij->i', Tf, grad_f_u)
            Cross_diffusion_v = bm.einsum('ij,ij->i', Tf, grad_f_v)
            NE = Cross_diffusion_u.shape[0]
            flux = bm.zeros((NE, 4))
            flux[is_internal, 0] =  Cross_diffusion_u[is_internal]
            flux[is_internal, 1] = -Cross_diffusion_u[is_internal]
            flux[is_boundary, 0] = Cross_diffusion_u[is_boundary]
            flux[is_internal, 2] =  Cross_diffusion_v[is_internal]
            flux[is_internal, 3] = -Cross_diffusion_v[is_internal]
            flux[is_boundary, 2] = Cross_diffusion_v[is_boundary]
            result = bm.zeros((NC,2))
            bm.add_at(result[:,0], edge_to_cell[:, 0], flux[:, 0])
            bm.add_at(result[:,0], edge_to_cell[:, 1], flux[:, 1])
            bm.add_at(result[:,1], edge_to_cell[:, 0], flux[:, 2])
            bm.add_at(result[:,1], edge_to_cell[:, 1], flux[:, 3])
            return result
        

        


        
        