from .lagrange_fe_space import LagrangeFESpace

from ..quadrature import GaussLobattoQuadrature, GaussLegendreQuadrature

from typing import Optional, TypeVar, Union, Generic, Callable
from ..typing import TensorLike, Index, _S, Threshold

from ..backend import backend_manager as bm
from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from .dofs import LinearMeshCFEDof, LinearMeshDFEDof
from .function import Function
from ..decorator import barycentric, cartesian

from .scaled_monomial_space_2d import ScaledMonomialSpace2d

_MT = TypeVar('_MT', bound=Mesh)

class NonConformingScalarVESpace2d(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh, p=1, device=None):
        """
        Virtual element space in 2D.
        p: the space order
        q: the index of integral formular
        bc: user can give a barycenter for every mesh cell
        """
        self.mesh = mesh
        self.device = mesh.device
        self.p = p
        self.smspace = ScaledMonomialSpace2d(mesh, p)
        self.cellmeasure = self.smspace.cellmeasure

        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype
        self.ikwargs = bm.context(mesh.cell[0]) if mesh.meshtype =='polygon' else bm.context(mesh.cell)
        self.fkwargs = bm.context(mesh.node)
        self.stype = 'csvem' # 空间类型

        self.cell2dof, self.cell2dofLocation = self.cell_to_dof() # 初始化的时候就构建出 cell2dof 数组
        self.SM = self.smspace.cell_mass_matrix() # (NC, sldof, sldof) 缩放单形式空间单元质量矩阵
        self.SS = self.smspace.cell_stiff_matrix() # 缩放单项式空间单元刚度矩阵

        self.PI1 = self.H1_project_matrix()
        self.dof_matrix = self.dof_matrix()
        self.PI0 = self.L2_project_matrix()
        self.stab = self.stabilization()
        

    def is_boundary_dof(self, threshold=None, method='interp'):
        """
        @brief 获取边界自由度
        """
        TD = self.mesh.top_dimension()
        if isinstance(threshold, TensorLike):
            index = threshold
        else:
            index = self.mesh.boundary_face_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter(TD-1, index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        edge2dof = self.edge_to_dof(index=index) # 只获取指定的面的自由度信息
        isBdDof = bm.zeros(gdof, dtype=bm.bool)
        isBdDof[edge2dof] = True
        return isBdDof

    def edge_to_dof(self, index=_S):
        """
        @brief 获取网格边与自由度的对应关系
        """
        if isinstance(index, slice) and index == slice(None):
            NE = self.mesh.number_of_edges()
            index = bm.arange(NE, **self.ikwargs)
        elif isinstance(index, TensorLike) and (index.dtype == bm.bool):
            index, = bm.nonzero(index)
            NE = len(index)
        elif isinstance(index, list) and (type(index[0]) is bm.bool):
            index, = bm.nonzero(index)
            NE = len(index)
        else:
            NE = len(index)

        NN = self.mesh.number_of_nodes()
        p = self.p

        idx = bm.arange(p, **self.ikwargs)
        edge2dof =  p*index[:, None] + idx
        return edge2dof

    face_to_dof = edge_to_dof

    def cell_to_dof(self, index=_S):

        p = self.p
        mesh = self.mesh
        cell, cellLocation = mesh.entity('cell')
        #cell = mesh._cell
        #cellLocation = mesh.cellLocation
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        cell2dofLocation = bm.zeros(NC+1, **self.ikwargs)
        cell2dofLocation[1:] = bm.cumsum(ldof, axis=0)


        if p == 1:
            return mesh.cell_to_edge(), cell2dofLocation
        else:
            #cell2dofLocation[1:] = bm.add.accumulate(ldof)
            cell2dof = bm.zeros(cell2dofLocation[-1], **self.ikwargs)

            edge2dof = self.edge_to_dof()
            edge2cell = mesh.edge_to_cell()

            idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + bm.arange(p, **self.ikwargs)
            cell2dof[idx] = edge2dof
 
            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
            idx = (cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p).reshape(-1, 1) + bm.arange(p)
            cell2dof[idx] = bm.flip(edge2dof[isInEdge, :],axis=1)

            NN = mesh.number_of_nodes()
            NV = mesh.number_of_vertices_of_cells()
            NE = mesh.number_of_edges()
            idof = (p-1)*p//2
            idx = (cell2dofLocation[:-1] + NV*p).reshape(-1, 1) + bm.arange(idof)
            cell2dof[idx] =  NE*p + bm.arange(NC*idof, **self.ikwargs).reshape(NC, idof)
            return cell2dof, cell2dofLocation

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        gdof = NE*p + NC*(p-1)*p//2
        return gdof

    def number_of_local_dofs(self, doftype='all'):
        p = self.p
        mesh = self.mesh
        if doftype == 'all':
            NCE = mesh.number_of_edges_of_cells()
            return NCE*p + (p-1)*p//2
        elif doftype in {'cell', 2}:
            return (p-1)*p//2
        elif doftype in {'edge', 'face', 1}:
            return p
        elif doftype in {'node', 0}:
            return 0 

    def interpolation_points(self, scale=0.3):
        p = self.p
        mesh = self.mesh
        gdof = self.number_of_global_dofs()
        node = mesh.entity('node')
        edge = mesh.entity('edge')

        GD = mesh.geo_dimension()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        qf = GaussLegendreQuadrature(p)
        bcs, ws = qf.get_quadrature_points_and_weights()
        ipoint = bm.zeros((gdof, GD), **self.fkwargs)
        if p==1:
            ipoint = bm.einsum( 'ij, ...jm->...im', bcs, node[edge, :]).reshape(-1, GD) # bcs:(NQ=p, 2) (NE, 2, GD)
            return ipoint

        ipoint[:NE*p, :] =  bm.einsum( 'ij, ...jm->...im', bcs, node[edge, :]).reshape(-1, GD)
        if p == 2:
            ipoint[NE*p:, :] = mesh.entity_barycenter('cell')
            return ipoint

        h = bm.sqrt(mesh.cell_area())[:, None]*scale
        bc = mesh.entity_barycenter('cell')
        t = bm.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, bm.sqrt(bm.array(3))/2]], **self.fkwargs)
        t -= bm.array([0.5, bm.sqrt(bm.array(3))/6.0], **self.fkwargs)

        tri = bm.zeros((NC, 3, GD), **self.fkwargs)
        tri[:, 0, :] = bc + t[0]*h
        tri[:, 1, :] = bc + t[1]*h
        tri[:, 2, :] = bc + t[2]*h

        TD = mesh.top_dimension()
        bcs = mesh.multi_index_matrix(p-2, TD, **self.fkwargs)/(p-2)
        ipoint[NE*p:, :] = bm.einsum('ij, ...jm->...im', bcs, tri).reshape(-1, GD)
        return ipoint

    def project_to_smspace(self, uh):
        """
        @brief Project a non conforming vem function uh into polynomial space.

        @param[in] uh
        @param[in] PI
        """
        p = self.p
        cell2dof = self.cell2dof
        cell2dofLocation = self.cell2dofLocation
        cd = bm.split(cell2dof, cell2dofLocation[1:-1], axis=0)

        g = lambda x: x[0]@uh[x[1]]
        S = self.smspace.function()
        S[:] = bm.concatenate(list(map(g, zip(self.PI1, cd))))
        return S

    def boundary_interpolate(self, gd, uh, threshold=None, method='interp'):
        """
        初始化解 uh  的第一类边界条件。
        """
        p = self.p
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        ipoints = self.interpolation_points()
        isDDof = self.is_boundary_dof(threshold=threshold)
        uh[isDDof] = gd(ipoints[isDDof])
        return uh, isDDof

    def interpolation(self, u, iptype=True):
        """
        @brief 把函数 u 插值到非协调空间当中
        """
        p = self.p
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        ipoint = self.interpolation_points()
        uI = self.function()
        uI[:NE*p] = u(ipoint)
        if p > 1:
            phi = self.smspace.basis

            def f(x, index):
                return np.einsum('ij, ij...->ij...', u(x), phi(x, index=index, p=p-2))

            bb = self.mesh.integral(f, celltype=True)/self.smspace.cellmeasure[..., bm.newaxis]
            uI[p*NE:] = bb.reshape(-1)
        return uI


    def H1_project_matrix(self):                                         
        smspace = self.smspace                                                 
        p = self.p                                                             
        #from fealpy.fem import ScalarDiffusionIntegrator                        
        #Integrator = ScalarDiffusionIntegrator(q=p+1,method='homogeneous')      
        #S = Integrator.homogeneous_assembly(smspace)                            

        ## 投影矩阵左边矩阵
        if p==1:                                                                
            B1 = smspace.edge_integral(smspace.basis) # (NC, 4)                          
        else:                                                                   
            B1 = smspace.integral(smspace.basis) # (NC, sldof)                               
            #B1 = space.mesh.integral(smspace.basis, celltype=True)             
        left_matrix = bm.copy(self.SS) # (NC, sldof, sldof)
        left_matrix[:, 0, :] = B1  
        
        ## 投影矩阵右边矩阵
        mesh = self.mesh                                                       
        smldof = smspace.number_of_local_dofs()                                 
        cell, celllocaion = mesh.entity('cell')                                 
        cell2dof, cell2dofLocation = self.cell_to_dof()                        
        NC = mesh.number_of_cells()                                             
        NV = mesh.number_of_vertices_of_cells()                                 
        node = mesh.entity('node')                                              
        edge = mesh.entity('edge')                                              
        edge2cell = mesh.edge_to_cell()                                         
        cell2edge = mesh.cell_to_edge()                                         
        cell2edge = bm.split(cell2edge, celllocaion[1:-1], axis=-1)             

        qf = GaussLegendreQuadrature(p)                                        
        bcs, ws = qf.quadpts, qf.weights                                        
        ps = bm.einsum('qi,eij->eqj',bcs, node[edge]) #(NQ, 2), (NE,2,2)        
        gsmphi = smspace.grad_basis(ps, index=edge2cell[:,0]) # (NE, NQ, ldof, 2)
        nm = mesh.edge_normal()                                                 
        A1 = bm.einsum('q,eqli,ei->leq',ws,gsmphi,nm)                           
        isInEdge = edge2cell[:, 0] != edge2cell[:, 1]                           
        gsmphi = smspace.grad_basis(bm.flip(ps[isInEdge], axis=1), index=edge2cell[isInEdge,1]) # (NE, NQ, ldof, 2)
        A2 = bm.einsum('q,eqli,ei->leq',ws,gsmphi,-nm[isInEdge]) # (ldof, NE, NQ)                         
                                                                                
        BB = bm.zeros((smldof, cell2dofLocation[-1]), **mesh.fkwargs)           
        B = list(bm.split(BB, cell2dofLocation[1:-1], axis=-1))                       
        Px, Py = smspace.partial_matrix()                                       
        L = Px@Px + Py@Py           
        for i in range(NC):                                                     
            B[i][:(p-1)*p//2, NV[i]*p:] = bm.eye( (p-1)*p//2, **mesh.fkwargs)   
            B[i][...] = -L[i].T @ B[i]                                               
        idx = (cell2dofLocation[edge2cell[:, 0]] + edge2cell[:, 2]*p).reshape(-1, 1) + bm.arange(p, **mesh.ikwargs)
        BB[:, idx] += A1
        idx = (cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p).reshape(-1, 1) + bm.arange(p, **mesh.ikwargs)
        BB[:, idx] += A2
        edge_measure = mesh.entity_measure('edge')
        if p==1:
            idx = cell2dofLocation[edge2cell[:, 0]] + edge2cell[:, 2]
            BB[0, idx] = edge_measure
            idx = cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]
            BB[0, idx] += edge_measure[isInEdge]
        else:
            idx = cell2dofLocation[:-1]+ NV*p
            BB[0, idx] = bm.ones(NC, **self.fkwargs)

        B = bm.split(BB, cell2dofLocation[1:-1], axis=-1)

        ## 投影矩阵
        g = lambda x: bm.linalg.inv(x[0])@x[1]                                  
        PI1 = list(map(g, zip(left_matrix, B)))      
        return PI1              

    def dof_matrix(self): 
        p = self.p                                                             
        mesh = self.mesh                                                       
        smspace = self.smspace                                                 
        node = mesh.entity('node')                                              
        edge = mesh.entity('edge')                                              
        edge2cell = mesh.edge_to_cell()                                         
        smldof = smspace.number_of_local_dofs()                                 
        cell, celllocation = mesh.entity('cell')                                
        cell2dof, cell2doflocation = self.cell_to_dof()                        
        D = bm.zeros((cell2doflocation[-1], smldof), **mesh.fkwargs)            
                                                                                
        qf = GaussLegendreQuadrature(p)                                        
        bcs, ws = qf.quadpts, qf.weights                                        
        ps = bm.einsum('qi, eij->eqj', bcs, node[edge])                         
        phi0 = smspace.basis(ps, index=edge2cell[:,0]) #(NE, NQ, ldof) 
        isInedge = edge2cell[:,0] != edge2cell[:,1]                             
        phi1 = smspace.basis(bm.flip(ps[isInedge, :], axis=1), index=edge2cell[isInedge,1])
        idx = (cell2doflocation[edge2cell[:, 0]] + edge2cell[:, 2]*p +
               bm.arange(p).reshape(-1, 1)).T #(NE, NQ)
        D[idx, :] = phi0                                                        
        idx = (cell2doflocation[edge2cell[isInedge, 1]] + edge2cell[isInedge, 3]*p + bm.arange(p).reshape(-1, 1)).T
        D[idx, :] = phi1                                                        
                                                                                
        ildof = (p-1)*p//2                                                      
        idx = cell2doflocation[1:][:, None] + bm.arange(-ildof, 0)              
        D[idx, :] = self.SM[:, :ildof, :]                                             
        return list(bm.split(D, cell2doflocation[1:-1], axis=0))        

    def L2_project_matrix(self):                                         
        p = self.p                                                             
        smspace = self.smspace
        mesh = self.mesh
        ldof = self.number_of_local_dofs()                                     
        NC = mesh.number_of_cells()                                       
        NV = mesh.number_of_vertices_of_cells()                           
        cell2dof, cell2doflocation = self.cell_to_dof()                        
        smldof = smspace.number_of_local_dofs(p=p)                        
        Q = bm.zeros((smldof, cell2doflocation[-1]), **mesh.fkwargs)      
        Q = list(bm.split(Q, cell2doflocation[1:-1], axis=-1))                       
        if p==1:                                                                
            return self.PI1                                                          
        else:                                                                   
            smldof2 = smspace.number_of_local_dofs(p=p-2)                 
            M2 = self.SM[:, :smldof2, :smldof2]                                       
            for i in range(NC):                                                 
                I = bm.zeros((smldof2, ldof[i]), **mesh.fkwargs)          
                idx = NV[i]*p                                                   
                I[:, idx:] = bm.eye(smldof2, **mesh.fkwargs)              
                                                                                
                Q2 = bm.linalg.inv(M2[i]) @ I                                   
                Q2 = bm.concatenate([Q2, bm.zeros((smldof-smldof2, ldof[i]), **mesh.fkwargs)], axis=0)
                Q[i] = self.PI1[i] - Q2 @ self.dof_matrix[i] @ self.PI1[i] + Q2                         
            return Q                                                            

    def stabilization(self):                                             
        D = self.dof_matrix                               
        PI1 = self.PI1                             
        f1 = lambda x: (bm.eye(x[1].shape[1])-x[0]@x[1]).T @ (bm.eye(x[1].shape[1]) - x[0]@x[1])
        K = list(map(f1, zip(D, PI1)))                                          
        return K    



