from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike

from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.utils import process_coef_func
from fealpy.functional import bilinear_integral, linear_integral, get_semilinear_coef
from fealpy.quadrature import GaussLobattoQuadrature
from .integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache,
    assemblymethod,
    CoefLike
)


class ScalarDiffusionIntegrator(LinearInt, OpInt, CellInt):
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
#
#    def assembly(self, space: _FS) -> TensorLike:
#        coef = self.coef
#        mesh = getattr(space, 'mesh', None)
#        bcs, ws, phi, cm, index = self.fetch(space)
#        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
#
#        return bilinear_integral(phi, phi, ws, cm, val, batched=self.batched)

    @enable_cache
    def h1_left(self, space: _FS):
        smspace = space.smspace
        p = space.p
        from fealpy.fem import ScalarDiffusionIntegrator
        Integrator = ScalarDiffusionIntegrator(q=p+1,method='homogeneous')
        S = Integrator.homogeneous_assembly(smspace)
        if p==1:
            B1 = smspace.edge_integral(smspace.basis)
        else:
            B1 = smspace.integral(smspace.basis)
            #B1 = space.mesh.integral(smspace.basis, celltype=True)
        S[:, 0, :] = B1
        return  S

    @enable_cache
    def h1_right(self, space:_FS):
        smspace = space.smspace
        p = space.p
        mesh = space.mesh 
        smldof = smspace.number_of_local_dofs()
        cell, celllocaion = mesh.entity('cell') 
        cell2dof, cell2doflocation = space.cell_to_dof()
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.edge_to_cell()
        cell2edge = mesh.cell_to_edge()
        cell2edge = bm.split(cell2edge, celllocaion[1:-1], axis=-1)
        edge_measure = mesh.entity_measure('edge')

        qf = GaussLobattoQuadrature(p+1)
        bcs, ws = qf.quadpts, qf.weights
        ps = bm.einsum('qi,eij->eqj',bcs, node[edge]) #(NQ, 2), (NE,2,2)
        gsmphi = smspace.grad_basis(ps, index=edge2cell[:,0]) # (NE, NQ, ldof, 2)
        nm = mesh.edge_normal()
        A1 = bm.einsum('q,eqli,ei->leq',ws,gsmphi,nm)
        isInedge = edge2cell[:, 0] != edge2cell[:, 1]
        gsmphi = smspace.grad_basis(ps, index=edge2cell[:,1]) # (NE, NQ, ldof, 2)
        A2 = bm.einsum('q,eqli,ei->leq',ws,gsmphi,-nm)

        BB = bm.zeros((smldof, cell2doflocation[-1]), **mesh.fkwargs)
        B = bm.split(BB, cell2doflocation[1:-1], axis=-1)
        Px, Py = smspace.partial_matrix()
        L = Px@Px + Py@Py
        for i in range(NC):
            B[i][:(p-1)*p//2, NV[i]*p:] = bm.eye( (p-1)*p//2, **mesh.fkwargs)
            #B[i] = bm.einsum('ij,ik->jk', L[i], B[i])
            B[i] = -L[i].T @ B[i]
            flag = edge2cell[:, 0]==i
            begin = edge2cell[flag, 2]*p
            idx = bm.array(begin)[:, None] + bm.arange(p+1)
            end = idx == p*NV[i]
            idx[end] = 0
            bm.index_add(B[i],idx.flat, A1[:, flag].reshape(smldof,-1), axis=1)
            flag = (edge2cell[:, 1]==i) & isInedge
            if bm.sum(flag)>0:
                begin = edge2cell[flag, 3]*p
                idx = bm.flip(bm.array(begin)[:, None] + bm.arange(p+1), axis=1)
                end = idx == p*NV[i]
                idx[end] = 0
                bm.index_add(B[i],idx.flat, A2[:, flag].reshape(smldof,-1), axis=1)
            if p==1:
                cedge = bm.zeros(NV[i]+1, **mesh.ikwargs)
                cedge[1:] = cell2edge[i]
                cedge[0] = cell2edge[i][-1]
                B[i][0,:] = (edge_measure[cedge[:-1]] + edge_measure[cedge[1:]])/2
            else:
                B[i][0, NV[i]*p] = 1
        return B

    @enable_cache
    def H1_project_matrix(self, space):
        g = lambda x: bm.linalg.inv(x[0])@x[1]
        PI1 = list(map(g, zip(self.h1_left(space), self.h1_right(space))))
        return PI1

    @enable_cache
    def dof_matrix(self, space): 
        p = space.p 
        mesh = space.mesh
        smspace = space.smspace
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.edge_to_cell()
        smldof = smspace.number_of_local_dofs() 
        cell, celllocation = mesh.entity('cell')
        cell2dof, cell2doflocation = space.cell_to_dof()
        D = bm.zeros((cell2doflocation[-1], smldof), **mesh.fkwargs)

        qf = GaussLobattoQuadrature(p+1)
        bcs, ws = qf.quadpts, qf.weights
        ps = bm.einsum('qi, eij->eqj', bcs, node[edge])
        phi0 = smspace.basis(ps[:,:-1,:], index=edge2cell[:,0]) #(NE, NQ, ldof)
        isInedge = edge2cell[:,0] != edge2cell[:,1]
        phi1 = smspace.basis(bm.flip(ps[isInedge, 1:], axis=1), index=edge2cell[isInedge,1])
        idx = (cell2doflocation[edge2cell[:, 0]] + edge2cell[:, 2]*p + bm.arange(p).reshape(-1, 1)).T 
        D[idx, :] = phi0
        idx = (cell2doflocation[edge2cell[isInedge, 1]] + edge2cell[isInedge, 3]*p + bm.arange(p).reshape(-1, 1)).T
        D[idx, :] = phi1
        
        from fealpy.fem import ScalarMassIntegrator
        Integrator = ScalarMassIntegrator(q=p+1, method='homogeneous')
        M = Integrator.homogeneous_assembly(smspace)
        ildof = (p-1)*p//2
        idx = cell2doflocation[1:][:, None] + bm.arange(-ildof, 0)
        D[idx, :] = M[:, :ildof, :]
        return bm.split(D, cell2doflocation[1:-1], axis=0)

    @enable_cache
    def L2_left(self, space):
        from fealpy.fem import ScalarMassIntegrator
        p = space.p
        Integrator = ScalarMassIntegrator(q=p+1, method='homogeneous')
        M = Integrator.homogeneous_assembly(space.smspace)
        return M

    @enable_cache
    def L2_project_matrix(self, space): 
        p = space.p
        ldof = space.number_of_local_dofs()
        NC = space.mesh.number_of_cells()
        NV = space.mesh.number_of_vertices_of_cells()        
        cell2dof, cell2doflocation = space.cell_to_dof()
        PI1 = self.H1_project_matrix(space) #重复使用
        D = self.dof_matrix(space) #重复使用
        smldof = space.smspace.number_of_local_dofs(p=p)
        Q = bm.zeros((smldof, cell2doflocation[-1]), **space.mesh.fkwargs)
        Q = bm.split(Q, cell2doflocation[1:-1], axis=-1)
        if p==1:
            return PI1
        else:
            from fealpy.fem import ScalarMassIntegrator
            Integrator = ScalarMassIntegrator(q=p+1, method='homogeneous') #重复使用
            M = Integrator.homogeneous_assembly(space.smspace)
            smldof2 = space.smspace.number_of_local_dofs(p=p-2)
            M2 = M[:, :smldof2, :smldof2]
            for i in range(NC):
                I = bm.zeros((smldof2, ldof[i]), **space.mesh.fkwargs)
                idx = NV[i]*p
                I[:, idx:] = bm.eye(smldof2, **space.mesh.fkwargs)

                Q2 = bm.linalg.inv(M2[i]) @ I
                Q2 = bm.concatenate([Q2, bm.zeros((smldof-smldof2, ldof[i]), **space.mesh.fkwargs)], axis=0)
                Q[i] = PI1[i] - Q2 @ D[i] @ PI1[i] + Q2
                #Q[i] = PI1 - Q2 @ D @ PI1 + Q2
            return Q

    @enable_cache
    def stabilization(self, space):
        D = self.dof_matrix(space)
        PI1 = self.H1_project_matrix(space)
        f1 = lambda x: (bm.eye(x[1].shape[1])-x[0]@x[1]).T @ (bm.eye(x[1].shape[1]) - x[0]@x[1])
        K = list(map(f1, zip(D, PI1)))
        return K


    @enable_cache
    def assembly(self, space):
        space.PI0 = self.L2_project_matrix(space)
        space.PI1 = self.H1_project_matrix(space)
        p = space.p
        from fealpy.fem import ScalarDiffusionIntegrator
        Integrator = ScalarDiffusionIntegrator(q=p+1,method='homogeneous') #重复使用
        S = Integrator.homogeneous_assembly(space.smspace)

        PI1 = self.H1_project_matrix(space)
        f = lambda x: x[0].T @ x[1] @ x[0]
        K = list(map(f, zip(PI1, S)))
        S = self.stabilization(space)
        KK = list(map(lambda x: x[0] + x[1], zip(K, S)))
        return KK

    
        







        



