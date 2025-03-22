from typing import Optional

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral, linear_integral, get_semilinear_coef
from .integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache,
    assemblymethod,
    CoefLike
)


class ScalarMassIntegrator(LinearInt, OpInt, CellInt):
    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        q = self.q
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarMassIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index)
        return bcs, ws, phi, cm, index

    def assembly(self, space: _FS) -> TensorLike:
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, cm, index = self.fetch(space)
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)

        return bilinear_integral(phi, phi, ws, cm, val, batched=self.batched)

    @assemblymethod('semilinear')
    def semilinear_assembly(self, space: _FS) -> TensorLike:
        uh = self.uh
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, cm, index = self.fetch(space)
        val_A = coef.grad_func(uh(bcs))  #(C, Q)
        val_F = -coef.func(uh(bcs))      #(C, Q)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        coef_A = get_semilinear_coef(val_A, coef)
        coef_F = get_semilinear_coef(val_F, coef)

        return bilinear_integral(phi, phi, ws, cm, coef_A, batched=self.batched), \
               linear_integral(phi, ws, cm, coef_F, batched=self.batched)

    @assemblymethod('isopara')
    def isopara_assembly(self, space: _FS, /, indices=None) -> TensorLike:
        """
        等参有限元质量矩阵组装
        """
        index = self.entity_selection(indices)
        mesh = getattr(space, 'mesh', None)

        rm = mesh.reference_cell_measure()
        cm = mesh.entity_measure('cell', index=index)

        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        J = mesh.jacobi_matrix(bcs, index=index)
        G = mesh.first_fundamental_form(J) 
        d = bm.sqrt(bm.linalg.det(G))
        phi = space.basis(bcs)
        M = bm.einsum('q, cqi, cqj, cq -> cij', ws*rm, phi, phi, d) #(NC, ldof, ldof)
        return M

    @assemblymethod('homogeneous')
    def homogeneous_assembly(self, space: _FS):
        """
        homogenous funciton space(scaled monomial space) assembly, applicable to arbitrary polygonal meshes.
        """
        def integral(f):
            """
            homogenous function integral, applicable to arbitrary polygonal meshes
            """
            mesh = space.mesh                                                        
            node = mesh.entity('node')                                              
            edge = mesh.entity('edge')                                              
            edge2cell = mesh.edge_to_cell()                                         
            edgebarycenter = mesh.entity_barycenter('edge')
            cellbarycenter = mesh.entity_barycenter('cell')
            #edgebarycenter = node[edge[:, 0]] - cellbarycenter[edge2cell[:, 0]] # (NE, 2)
                                                                                 
            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])                         
                                                                                 
            NC = mesh.number_of_cells()                                             
            qf = mesh.quadrature_formula(p+1, etype='edge', qtype='legendre') # NQ  
            bcs, ws = qf.quadpts, qf.weights # (NQ, 2)  (NQ,)                       
            ps = bm.einsum('ij, kjm->kim', bcs, node[edge]) # (NQ, 2) (NE, 2, 2)                        
            f1 = f(ps, index=edge2cell[:, 0]) # (NE, NQ, ldof)
            nm = mesh.edge_normal()
            b = node[edge[:, 0]] - cellbarycenter[edge2cell[:, 0]]
            H0 = bm.einsum('eqlk, q, ed, ed-> elk', f1, ws, b, nm) # (NC, 2, 2)
            f2 = f(ps, index=edge2cell[:, 1])
            b = node[edge[isInEdge, 0]] - cellbarycenter[edge2cell[isInEdge, 1]]
            H1 = bm.einsum('eqlk, q, ed, ed-> elk', f2[isInEdge], ws, b, -nm[isInEdge]) # (NC, 2, 2)
            H = bm.zeros((NC, f1.shape[-2], f1.shape[-1]), **mesh.fkwargs)
            bm.index_add(H, edge2cell[:, 0], H0)
            bm.index_add(H, edge2cell[isInEdge, 1], H1)
            multiIndex = space.multi_index_matrix(p=p)
            q = bm.sum(multiIndex, axis=1)
            H /= q + q.reshape(-1, 1) + 2
            return H

        p = space.p                                    
        def f(x, index):
            phi = space.basis(x, index=index, p=p)
            return bm.einsum('eqi, eqj->eqij', phi, phi)
        return integral(f)
