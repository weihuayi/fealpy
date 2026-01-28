from typing import Optional
from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, SourceLike
from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import linear_integral
from fealpy.functionspace.functional import symmetry_span_array, symmetry_index
from .integrator import LinearInt, SrcInt, CellInt, enable_cache 

class ScalarMLaplaceSourceIntegrator(LinearInt, SrcInt, CellInt):
    def __init__(self, m:int=2, source: Optional[SourceLike]=None, q:int=None,*,
                 index:Index=_S, batched: bool=False)->None:
        super().__init__()                                                              
        self.source = source                                                            
        self.m = m
        self.q = q                                                                      
        self.index = index                                                              
        self.batched = batched

    @enable_cache                                                               
    def to_global_dof(self, space: _FS) -> TensorLike:                          
        return space.cell_to_dof()[self.index]     

    @enable_cache                                                               
    def fetch(self, space: _FS):                                                
        q = self.q                                                              
        m = self.m
        index = self.index                                                      
        mesh = getattr(space, 'mesh', None)                                     
                                                                                
        if not isinstance(mesh, HomogeneousMesh):                               
            raise RuntimeError("The ScalarMLaplaceSourceIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")                   
                                                                                
        cm = mesh.entity_measure('cell', index=index)                           
        q = space.p+3 if self.q is None else self.q                             
        qf = mesh.quadrature_formula(q, 'cell')                                 
        bcs, ws = qf.get_quadrature_points_and_weights()                        
        gmphi = space.grad_m_basis(bcs, m=m)                                     
        return bcs, ws, gmphi, cm, index   

    def assembly(self, space: _FS) -> TensorLike:                               
        gmf = self.source                                                         
        m = self.m
        ftype = space.ftype
        mesh = getattr(space, 'mesh', None)                                     
        GD = mesh.geo_dimension()
        bcs, ws, gmphi, cm, index = self.fetch(space) # (NQ, GDI+1), (NQ,), (NC, NQ, ldof，r), (NC,)     
        idx, num= symmetry_index(GD, m, dtype=ftype) # (r,) (r,)
        point = mesh.bc_to_point(bcs, index=index)
        gmfval = bm.array(gmf(point)) #(NC, NQ, r)
        gF = bm.einsum('cqr,cqlr,r,q,c->cl' ,gmfval,gmphi,num,ws,cm)
        return gF 
