from typing import Optional, Literal                                            
                                                                                
from ..backend import backend_manager as bm                                     
from ..typing import TensorLike, Index, _S, SourceLike                          
                                                                                
from ..mesh import HomogeneousMesh                                              
from ..functionspace.space import FunctionSpace as _FS                          
from ..utils import process_coef_func                                           
from ..functional import linear_integral                                        
from .integrator import LinearInt, SrcInt, CellInt, enable_cache, assemblymethod
                                                                                
                                                                                
class ScalarSourceIntegrator(LinearInt, SrcInt, CellInt):                       
    r"""The domain source integrator for function spaces based on homogeneous meshes."""
    def __init__(self, source: Optional[SourceLike]=None, q: int=None, *,          
                 region: Optional[TensorLike] = None,                           
                 batched: bool=False,                                           
                 method: Literal['isopara', None] = None) -> None:              
        super().__init__(method=method if method else 'assembly')               
        self.source = source                                                    
        self.q = q                                                              
        self.set_region(region)                                                 
        self.batched = batched                                                  
                                                                                
    @enable_cache                                                               
    def to_global_dof(self, space: _FS, /, indices=None) -> TensorLike:         
        if indices is None:                                                     
            return space.cell_to_dof()                                          
        return space.cell_to_dof(index=self.entity_selection(indices))          
#                                                                                
#    @enable_cache                                                               
#    def fetch(self, space: _FS, /, inidces=None):                               
#        q = self.q                                                              
#        index = self.entity_selection(inidces)                                  
#        mesh = getattr(space, 'mesh', None)                                     
#                                                                                
#        if not isinstance(mesh, HomogeneousMesh):                               
#            raise RuntimeError("The ScalarSourceIntegrator only support spaces on"
#                               f"homogeneous meshes, but {type(mesh).__name__} is"
#                               "not a subclass of HomoMesh.")                   
#                                                                                
#        cm = mesh.entity_measure('cell', index=index)                           
#        q = space.p+3 if self.q is None else self.q                             
#        qf = mesh.quadrature_formula(q, 'cell')                                 
#        bcs, ws = qf.get_quadrature_points_and_weights()                        
#        phi = space.basis(bcs, index=index)                                     
#        return bcs, ws, phi, cm, index                                   

    def assembly(self, space):
        f = self.source

        phi = space.smspace.basis                                               
        p = space.p                                                             
        q = p+3                                                                 
        def u(x, index):                                                        
            return bm.einsum('ij, ijm->ijm', f(x), phi(x, index)) # (NE, NQ, 2)              
        bb = space.mesh.integral(u, q=q, celltype=True)                         
        g = lambda x: x[0].T @ x[1]                                             
        PI0 = space.PI0                        
        bb = bm.concatenate(list(map(g, zip(PI0, bb))))                         
        return bb 

    @assemblymethod('vector')
    def vector_assembly(self, space):
        scalar_space = space.scalar_space
        f = self.source
        phi = scalar_space.smspace.basis                                               
        p = space.p                                                             
        q = p+3                                                                 
        def u(x, index):   # x:NE, NQ, 2; index:NE
            return bm.einsum('ijd, ijm->ijmd', f(x), phi(x, index)) # (NE, NQ, 2) #(NE, NQ, ldof)             
        bb = scalar_space.mesh.integral(u, q=p+3, celltype=True)                         
        g = lambda x: x[0].T @ x[1]                                             
        PI0 = scalar_space.PI0                        
        import ipdb
        ipdb.set_trace()
        bb = list(map(g, zip(PI0, bb)))                     
        return bb 






