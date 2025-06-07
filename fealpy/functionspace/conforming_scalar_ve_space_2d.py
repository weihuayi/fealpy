from .lagrange_fe_space import LagrangeFESpace

from fealpy.quadrature import GaussLobattoQuadrature

from typing import Optional, TypeVar, Union, Generic, Callable
from ..typing import TensorLike, Index, _S, Threshold

from ..backend import TensorLike
from ..backend import backend_manager as bm
from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from .dofs import LinearMeshCFEDof, LinearMeshDFEDof
from .function import Function
from fealpy.decorator import barycentric, cartesian

from .scaled_monomial_space_2d import ScaledMonomialSpace2d

_MT = TypeVar('_MT', bound=Mesh)

class ConformingScalarVESpace2d(FunctionSpace, Generic[_MT]):
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
        idx = self.mesh.boundary_face_index()
        if threshold is not None:
            bc = self.mesh.entity_barycenter('edge')[idx]
            flag = threshold(bc)
            idx  = idx[flag]
        gdof = self.number_of_global_dofs()
        isBdDof = bm.zeros(gdof, dtype=bm.bool)
        edge2dof = self.edge_to_dof()
        isBdDof[edge2dof[idx]] = True
        return isBdDof

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()

        edge = mesh.entity('edge')
        edge2dof = bm.zeros((NE, p+1), **self.ikwargs)
        edge2dof[:, [0, p]] = edge
        if p > 1:
            edge2dof[:, 1:-1] = bm.arange(NN, NN + NE*(p-1)).reshape(NE, p-1)
        return edge2dof

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell, cellLocation = mesh.entity('cell')
        #cell = mesh._cell
        #cellLocation = mesh.cellLocation

        if p == 1:
            return cell, cellLocation
        else:
            NC = mesh.number_of_cells()

            ldof = self.number_of_local_dofs()
            cell2dofLocation = bm.zeros(NC+1, **self.ikwargs)
            cell2dofLocation[1:] = bm.cumsum(ldof, axis=0)
            #cell2dofLocation[1:] = bm.add.accumulate(ldof)
            cell2dof = bm.zeros(cell2dofLocation[-1], **self.ikwargs)

            edge2dof = self.edge_to_dof()
            edge2cell = mesh.edge_to_cell()

            idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + bm.arange(p)
            cell2dof[idx] = edge2dof[:, 0:p]
 
            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
            idx = (cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p).reshape(-1, 1) + bm.arange(p)
            #cell2dof[idx] = edge2dof[isInEdge, p:0:-1]
            cell2dof[idx] = bm.flip(edge2dof[isInEdge, 1:p+1],axis=1)

            NN = mesh.number_of_nodes()
            NV = mesh.number_of_vertices_of_cells()
            NE = mesh.number_of_edges()
            idof = (p-1)*p//2
            idx = (cell2dofLocation[:-1] + NV*p).reshape(-1, 1) + bm.arange(idof)
            cell2dof[idx] = NN + NE*(p-1) + bm.arange(NC*idof, **self.ikwargs).reshape(NC, idof)
            return cell2dof, cell2dofLocation

    def number_of_global_dofs(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        gdof = NN
        p = self.p
        if p > 1:
            gdof += NE*(p-1) + NC*(p-1)*p//2
        return gdof

    def number_of_local_dofs(self, doftype='cell'):
        mesh = self.mesh
        NV = mesh.number_of_vertices_of_cells()
        ldofs = NV
        p = self.p
        if p > 1:
            ldofs += NV*(p-1) + (p-1)*p//2
        return ldofs

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        node = mesh.entity('node')

        if p == 1:
            return node
        if p > 1:
            NN = mesh.number_of_nodes()
            GD = mesh.geo_dimension()
            NE = mesh.number_of_edges()
            NC = mesh.number_of_cells()

            ipoint = bm.zeros((NN+(p-1)*NE+p*(p-1)//2*NC, GD), **self.fkwargs)
            ipoint[:NN, :] = node
            edge = mesh.entity('edge')

            qf = GaussLobattoQuadrature(p + 1)

            bcs = qf.quadpts[1:-1, :]
            ipoint[NN:NN+(p-1)*NE, :] = bm.einsum('ij, ...jm->...im', bcs, node[edge, :]).reshape(-1, GD)
            return ipoint


    #def integral(self, uh):
    #    """
    #    计算虚单元函数的积分 \int_\Omega uh dx
    #    """
 
    #    p = self.p
    #    cell2dof, cell2dofLocation = self.cell2dof, self.cell2dofLocation
    #    if p == 1:
    #        cd = bm.hsplit(cell2dof, cell2dofLocation[1:-1])
    #        f = lambda x: sum(uh[x[0]]*x[0, :])
    #        val = sum(map(f, zip(cd, self.C)))
    #        return val
    #    else:
    #        NV = self.mesh.number_of_vertices_of_cells()
    #        idx =cell2dof[cell2dofLocation[0:-1]+NV*p]
    #        val = bm.sum(uh[idx]*self.area)
    #        return val

    def project_to_smspace(self, uh):
        """
        Project a conforming vem function uh into polynomial space.
        """
        dim = len(uh.shape)
        p = self.p
        cell2dof = self.cell2dof
        cell2dofLocation = self.cell2dofLocation
        cd = bm.split(cell2dof, cell2dofLocation[1:-1], axis=0)
        g = lambda x: x[0]@uh[x[1]]
        S = self.smspace.function()
        S[:] = bm.concatenate(list(map(g, zip(self.PI1, cd))))
        return S

    def grad_recovery(self, uh):

        p = self.p
        smldof = self.smspace.number_of_local_dofs()
        NC = self.mesh.number_of_cells()
        h = self.smspace.cellsize

        s = self.project_to_smspace(uh).reshape(-1, smldof)
        sx = bm.zeros((NC, smldof), dtype=self.ftype)
        sy = bm.zeros((NC, smldof), dtype=self.ftype)

        start = 1
        r = bm.arange(1, p+1)
        for i in range(p):
            sx[:, start-i-1:start] = r[i::-1]*s[:, start:start+i+1]
            sy[:, start-i-1:start] = r[0:i+1]*s[:, start+1:start+i+2]
            start += i+2

        sx /= h.reshape(-1, 1)
        sy /= h.reshape(-1, 1)

        cell2dof, cell2dofLocation = self.cell2dof, self.cell2dofLocation
        NC = len(cell2dofLocation) - 1
        cd = bm.hsplit(cell2dof, cell2dofLocation[1:-1])
        DD = bm.vsplit(self.D, cell2dofLocation[1:-1])

        f1 = lambda x: x[0]@x[1]
        sx = bm.concatenate(list(map(f1, zip(DD, sx))))
        sy = bm.concatenate(list(map(f1, zip(DD, sy))))


        ldof = self.number_of_local_dofs()
        w = bm.repeat(1/self.smspace.cellsize, ldof)
        sx *= w
        sy *= w

        uh = self.function(dim=2)
        ws = bm.zeros(uh.shape[0], dtype=self.ftype)
        bm.add.at(uh[:, 0], cell2dof, sx)
        bm.add.at(uh[:, 1], cell2dof, sy)
        bm.add.at(ws, cell2dof, w)
        uh /=ws.reshape(-1, 1)
        return uh

    def recovery_estimate(self, uh, pde, method='simple', residual=True,
            returnsup=False):
        """
        estimate the recover-type error

        Parameters
        ----------
        self : PoissonVEMModel object
        rtype : str
            'simple':
            'area'
            'inv_area'

        See Also
        --------

        Notes
        -----

        """
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        cell, cellLocation = mesh.entity('cell')
        barycenter = self.smspace.cellbarycenter

        h = self.smspace.cellsize
        area = self.cellmeasure
        ldof = self.smspace.number_of_local_dofs()

        # project the vem solution into linear polynomial space
        idx = bm.repeat(range(NC), NV)
        S = self.project_to_smspace(uh)

        grad = S.grad_value(barycenter)
        S0 = self.smspace.function()
        S1 = self.smspace.function()
        n2c = mesh.node_to_cell()

        if method == 'simple':
            d = n2c.sum(axis=1)
            ruh = bm.asarray((n2c@grad)/d.reshape(-1, 1))
        elif method == 'area':
            d = n2c@area
            ruh = bm.asarray((n2c@(grad*area.reshape(-1, 1)))/d.reshape(-1, 1))
        elif method == 'inv_area':
            d = n2c@(1/area)
            ruh = bm.asarray((n2c@(grad/area.reshape(-1,1)))/d.reshape(-1, 1))
        else:
            raise ValueError("I have note code method: {}!".format(rtype))

        for i in range(ldof):
            S0[i::ldof] = bm.bincount(
                    idx,
                    weights=self.B[i, :]*ruh[cell, 0],
                    minlength=NC)
            S1[i::ldof] = bm.bincount(
                    idx,
                    weights=self.B[i, :]*ruh[cell, 1],
                    minlength=NC)

        k = 1 # TODO: for general diffusion coef

        node = mesh.node
        gx = S0.value(node[cell], idx) - bm.repeat(grad[:, 0], NV)
        gy = S1.value(node[cell], idx) - bm.repeat(grad[:, 1], NV)
        eta = k*bm.bincount(idx, weights=gx**2+gy**2)/NV*area


        if residual is True:
            fh = self.integralalg.fun_integral(pde.source, True)/area
            g0 = S0.grad_value(barycenter)
            g1 = S1.grad_value(barycenter)
            eta += (fh + k*(g0[:, 0] + g1[:, 1]))**2*area**2

        return bm.sqrt(eta)

    def smooth_estimator(self, eta):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        NV = mesh.number_of_vertices_of_cells()

        nodeEta = bm.zeros(NN, **self.fkwargs)

        cell, cellLocation = mesh.entity('cell')
        NNC = cellLocation[1:] - cellLocation[:-1] #number_of_node_per_cell
        NCN = bm.zeros(NN, **self.ikwargs) #number_of_cell_around_node

        number = bm.ones(NC, **self.ikwargs)

        
        for i in range(3):
            nodeEta[:]=0
            NCN[:]=0
            k = 0
            while True:
                flag = NNC > k
                if bm.all(~flag):
                    break
                bm.add.at(nodeEta, cell[cellLocation[:-1][flag]+k], eta[flag])
                bm.add.at(NCN, cell[cellLocation[:-1][flag]+k], number[flag])
                k += 1
            nodeEta = nodeEta/NCN
            eta[:] = 0

            k = 0
            while True:
                flag = NNC > k
                if bm.all(~flag):
                    break
                eta[flag] = eta[flag] + nodeEta[cell[cellLocation[:-1][flag]+k]]
                k += 1
            eta = eta/NNC
        return eta

    def H1_project_matrix(self):                                         
        smspace = self.smspace                                                 
        p = self.p                                                             
        #from fealpy.fem import ScalarDiffusionIntegrator                        
        #Integrator = ScalarDiffusionIntegrator(q=p+1,method='homogeneous')      
        #S = Integrator.homogeneous_assembly(smspace)                            

        ## 投影矩阵左边矩阵
        if p==1:                                                                
            B1 = smspace.edge_integral(smspace.basis)                           
        else:                                                                   
            B1 = smspace.integral(smspace.basis)                                
            #B1 = space.mesh.integral(smspace.basis, celltype=True)             
        left_matrix = bm.copy(self.SS)
        left_matrix[:, 0, :] = B1  
        
        ## 投影矩阵右边矩阵
        mesh = self.mesh                                                       
        smldof = smspace.number_of_local_dofs()                                 
        cell, celllocaion = mesh.entity('cell')                                 
        cell2dof, cell2doflocation = self.cell_to_dof()                        
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
        B = list(bm.split(BB, cell2doflocation[1:-1], axis=-1))                       
        Px, Py = smspace.partial_matrix()                                       
        L = Px@Px + Py@Py           
        cellmeasure = mesh.entity_measure('cell') 
        for i in range(NC):                                                     
            B[i][:(p-1)*p//2, NV[i]*p:] = cellmeasure[i]*bm.eye( (p-1)*p//2, **mesh.fkwargs)   
            #B[i] = bm.einsum('ij,ik->jk', L[i], B[i])                          
            B[i] = -L[i].T @ B[i]                                               
            flag = edge2cell[:, 0]==i                                           
            begin = edge2cell[flag, 2]*p                                        
            idx = bm.array(begin)[:, None] + bm.arange(p+1)                     
            end = idx == p*NV[i]                                                
            idx[end] = 0                                                        
            bm.index_add(B[i],idx.flatten(), A1[:, flag].reshape(smldof,-1), axis=1) 
            flag = (edge2cell[:, 1]==i) & isInedge                              
            if bm.sum(flag)>0:                                                  
                begin = edge2cell[flag, 3]*p                                    
                idx = bm.flip(bm.array(begin)[:, None] + bm.arange(p+1), axis=1)
                end = idx == p*NV[i]                                            
                idx[end] = 0                                                    
                bm.index_add(B[i],idx.flatten(), A2[:, flag].reshape(smldof,-1), axis=1)

            if p==1:                                                            
                cedge = bm.zeros(NV[i]+1, **mesh.ikwargs)                       
                cedge[1:] = cell2edge[i]                                        
                cedge[0] = cell2edge[i][-1]                                     
                B[i][0,:] = (edge_measure[cedge[:-1]] + edge_measure[cedge[1:]])/2
            else:                                                               
                B[i][0, NV[i]*p] = 1*cellmeasure[i]   

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
                                                                                
        #from fealpy.fem import ScalarMassIntegrator                             
        #Integrator = ScalarMassIntegrator(q=p+1, method='homogeneous')          
        #M = Integrator.homogeneous_assembly(smspace)                            
        ildof = (p-1)*p//2                                                      
        idx = cell2doflocation[1:][:, None] + bm.arange(-ildof, 0)              
        cellmeasure = mesh.entity_measure('cell')
        D[idx, :] = self.SM[:, :ildof, :]/cellmeasure.reshape(-1, 1,1)                                             
        return list(bm.split(D, cell2doflocation[1:-1], axis=0))        
    
    def L2_project_matrix(self):                                         
        p = self.p                                                             
        smspace = self.smspace
        mesh = self.mesh
        ldof = self.number_of_local_dofs()                                     
        NC = mesh.number_of_cells()                                       
        NV = mesh.number_of_vertices_of_cells()                           
        cell2dof, cell2doflocation = self.cell_to_dof()                        
        #PI1 = self.H1_project_matrix(space) #重复使用                           
        #D = self.dof_matrix(space) #重复使用                                    
        smldof = smspace.number_of_local_dofs(p=p)                        
        Q = bm.zeros((smldof, cell2doflocation[-1]), **mesh.fkwargs)      
        Q = list(bm.split(Q, cell2doflocation[1:-1], axis=-1))                       
        cellmeasure = mesh.entity_measure('cell')
        if p==1:                                                                
            return self.PI1                                                          
        else:                                                                   
            #from fealpy.fem import ScalarMassIntegrator                         
            #Integrator = ScalarMassIntegrator(q=p+1, method='homogeneous') #重复使用
            #M = Integrator.homogeneous_assembly(space.smspace)                  
            smldof2 = smspace.number_of_local_dofs(p=p-2)                 
            M2 = self.SM[:, :smldof2, :smldof2]                                       
            for i in range(NC):                                                 
                I = bm.zeros((smldof2, ldof[i]), **mesh.fkwargs)          
                idx = NV[i]*p                                                   
                I[:, idx:] = bm.eye(smldof2, **mesh.fkwargs)*cellmeasure[i]              
                                                                                
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


    def project(self, F, space1):
        """
        S is a function in ScaledMonomialSpace2d, this function project  S to 
        MonomialSpace2d.
        """
        space0 = F.space
        def f(x, index):
            return bm.einsum(
                    '...im, ...in->...imn',
                    mspace.basis(x, index),
                    smspace.basis(x, index)
                    )
        C = self.integralalg.integral(f, celltype=True)
        H = space1.matrix_H()
        PI0 = inv(H)@C
        SS = mspace.function()
        SS[:] = bm.einsum('ikj, ij->ik', PI0, S[smspace.cell_to_dof()]).reshape(-1)
        return SS

    def boundary_interpolate(self, gd, uh, threshold=None, method='interp'):
        """
        初始化解 uh  的第一类边界条件。
        """
        p = self.p
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        end = NN + (p - 1)*NE
        ipoints = self.interpolation_points()
        isDDof = self.is_boundary_dof(threshold=threshold)
        uh[isDDof] = gd(ipoints[:end][isDDof[:end]])
        return uh, isDDof

    def interpolation(self, u, HB=None):
        """
        u: 可以是一个连续函数， 也可以是一个缩放单项式函数
        """
        if HB is None:
            mesh = self.mesh
            NN = mesh.number_of_nodes()
            NE = mesh.number_of_edges()
            p = self.p
            ipoint = self.interpolation_points()
            uI = self.function()
            uI[:NN+(p-1)*NE] = u(ipoint[:NN+(p-1)*NE])
            if p > 1:
                phi = self.smspace.basis
                def f(x, index):
                    return bm.einsum(
                            'ij, ij...->ij...',
                            u(x), phi(x, index=index, p=p-2))
                bb = self.mesh.integral(f, q=p+3, celltype=True)/self.smspace.cellmeasure[..., bm.newaxis]
                uI[NN+(p-1)*NE:] = bb.reshape(-1)
            return uI
        else:
            uh = self.smspace.interpolation(u, HB)

            cell2dof, cell2dofLocation = self.cell_to_dof()
            NC = len(cell2dofLocation) - 1
            cd = bm.hsplit(cell2dof, cell2dofLocation[1:-1])
            DD = bm.vsplit(self.D, cell2dofLocation[1:-1])

            smldof = self.smspace.number_of_local_dofs()
            f1 = lambda x: x[0]@x[1]
            uh = bm.concatenate(list(map(f1, zip(DD, uh.reshape(-1, smldof)))))

            ldof = self.number_of_local_dofs()
            w = bm.repeat(1/self.smspace.cellmeasure, ldof)
            uh *= w

            uI = self.function()
            ws = bm.zeros(uI.shape[0], dtype=self.ftype)
            bm.add.at(uI, cell2dof, uh)
            bm.add.at(ws, cell2dof, w)
            uI /=ws
            return uI





#    def stiff_matrix(self, cfun=None):
#        area = self.smspace.cellmeasure
#
#        def f(x):
#            x[0, :] = 0
#            return x
#
#        p = self.p
#        G = self.G
#        D = self.D
#        PI1 = self.PI1
#
#        cell2dof, cell2dofLocation = self.cell_to_dof()
#        NC = len(cell2dofLocation) - 1
#        cd = bm.hsplit(cell2dof, cell2dofLocation[1:-1])
#        DD = bm.vsplit(D, cell2dofLocation[1:-1])
#
#        if p == 1:
#            tG = bm.array([(0, 0, 0), (0, 1, 0), (0, 0, 1)])
#            if cfun is None:
#                def f1(x):
#                    M = bm.eye(x[1].shape[1])
#                    M -= x[0]@x[1]
#                    N = x[1].shape[1]
#                    A = bm.zeros((N, N))
#                    idx = bm.arange(N)
#                    A[idx, idx] = 2
#                    A[idx[:-1], idx[1:]] = -1
#                    A[idx[1:], idx[:-1]] = -1
#                    A[0, -1] = -1
#                    A[-1, 0] = -1
#                    return x[1].T@tG@x[1] + M.T@A@M
#                f1 = lambda x: x[1].T@tG@x[1] + (bm.eye(x[1].shape[1]) - x[0]@x[1]).T@(bm.eye(x[1].shape[1]) - x[0]@x[1])
#                K = list(map(f1, zip(DD, PI1)))
#            else:
#                cellbarycenter = self.smspace.cellbarycenter
#                k = cfun(cellbarycenter)
#                f1 = lambda x: (x[1].T@tG@x[1] + (bm.eye(x[1].shape[1]) - x[0]@x[1]).T@(bm.eye(x[1].shape[1]) - x[0]@x[1]))*x[2]
#                K = list(map(f1, zip(DD, PI1, k)))
#        else:
#            tG = list(map(f, G))
#            if cfun is None:
#                f1 = lambda x: x[1].T@x[2]@x[1] + (bm.eye(x[1].shape[1]) - x[0]@x[1]).T@(bm.eye(x[1].shape[1]) - x[0]@x[1])
#                K = list(map(f1, zip(DD, PI1, tG)))
#            else:
#                cellbarycenter = self.smspace.cellbarycenter
#                k = cfun(cellbarycenter)
#                f1 = lambda x: (x[1].T@x[2]@x[1] + (bm.eye(x[1].shape[1]) - x[0]@x[1]).T@(bm.eye(x[1].shape[1]) - x[0]@x[1]))*x[3]
#                K = list(map(f1, zip(DD, PI1, tG, k)))
#
#        f2 = lambda x: bm.repeat(x, x.shape[0])
#        f3 = lambda x: bm.tile(x, x.shape[0])
#        f4 = lambda x: x.flatten()
#
#        I = bm.concatenate(list(map(f2, cd)))
#        J = bm.concatenate(list(map(f3, cd)))
#        val = bm.concatenate(list(map(f4, K)))
#        gdof = self.number_of_global_dofs()
#        A = csr_matrix((val, (I, J)), shape=(gdof, gdof), **self.fkwargs)
#        return A
#
#    def mass_matrix(self, cfun=None):
#        area = self.smspace.cellmeasure
#        p = self.p
#
#        PI0 = self.PI0
#        D = self.D
#        H = self.H
#        C = self.C
#
#        cell2dof, cell2dofLocation = self.cell_to_dof()
#        NC = len(cell2dofLocation) - 1
#        cd = bm.hsplit(cell2dof, cell2dofLocation[1:-1])
#        DD = bm.vsplit(D, cell2dofLocation[1:-1])
#
#        f1 = lambda x: x[0]@x[1]
#        PIS = list(map(f1, zip(DD, PI0)))
#
#        f1 = lambda x: x[0].T@x[1]@x[0] + x[3]*(bm.eye(x[2].shape[1]) - x[2]).T@(bm.eye(x[2].shape[1]) - x[2])
#        K = list(map(f1, zip(PI0, H, PIS, area)))
#
#        f2 = lambda x: bm.repeat(x, x.shape[0])
#        f3 = lambda x: bm.tile(x, x.shape[0])
#        f4 = lambda x: x.flatten()
#
#        I = bm.concatenate(list(map(f2, cd)))
#        J = bm.concatenate(list(map(f3, cd)))
#        val = bm.concatenate(list(map(f4, K)))
#        gdof = self.number_of_global_dofs()
#        M = csr_matrix((val, (I, J)), shape=(gdof, gdof), **self.fkwargs)
#        return M

    def cross_mass_matrix(self, wh):
        p = self.p
        mesh = self.mesh

        area = self.smspace.cellmeasure
        PI0 = self.PI0

        phi = self.smspace.basis
        def u(x, index):
            val = phi(x, index=index)
            wval = wh(x, index=index)
            return bm.einsum('ij, ijm, ijn->ijmn', wval, val, val)
        H = self.integralalg.integral(u, celltype=True)

        cell2dof, cell2dofLocation = self.cell_to_dof()
        NC = len(cell2dofLocation) - 1
        cd = bm.hsplit(cell2dof, cell2dofLocation[1:-1])

        f1 = lambda x: x[0].T@x[1]@x[0]
        K = list(map(f1, zip(PI0, H)))

        f2 = lambda x: bm.repeat(x, x.shape[0])
        f3 = lambda x: bm.tile(x, x.shape[0])
        f4 = lambda x: x.flatten()

        I = bm.concatenate(list(map(f2, cd)))
        J = bm.concatenate(list(map(f3, cd)))
        val = bm.concatenate(list(map(f4, K)))
        gdof = self.number_of_global_dofs()
        M = csr_matrix((val, (I, J)), shape=(gdof, gdof), **self.fkwargs)
        return M

#    def source_vector(self, f):
#        PI0 = self.PI0
#        phi = self.smspace.basis
#        def u(x, index):
#            return bm.einsum('ij, ijm->ijm', f(x), phi(x, index=index))
#        bb = self.integralalg.integral(u, celltype=True)
#        g = lambda x: x[0].T@x[1]
#        bb = bm.concatenate(list(map(g, zip(PI0, bb))))
#        gdof = self.number_of_global_dofs()
#        b = bm.bincount(self.cell2dof, weights=bb, minlength=gdof)
#        return b

    def chen_stability_term(self):
        area = self.smspace.cellmeasure

        p = self.p
        G = self.G
        D = self.D
        PI1 = self.PI1

        cell2dof, cell2dofLocation = self.cell_to_dof()
        NC = len(cell2dofLocation) - 1
        cd = bm.hsplit(cell2dof, cell2dofLocation[1:-1])
        DD = bm.vsplit(D, cell2dofLocation[1:-1])

        tG = bm.array([(0, 0, 0), (0, 1, 0), (0, 0, 1)])
        def f1(x):
            M = bm.eye(x[1].shape[1])
            M -= x[0]@x[1]
            N = x[1].shape[1]
            A = bm.zeros((N, N))
            idx = bm.arange(N)
            A[idx, idx] = 2
            A[idx[:-1], idx[1:]] = -1
            A[idx[1:], idx[:-1]] = -1
            A[0, -1] = -1
            A[-1, 0] = -1
            return x[1].T@tG@x[1],  M.T@A@M
        K = list(map(f1, zip(DD, PI1)))
        f2 = lambda x: bm.repeat(x, x.shape[0])
        f3 = lambda x: bm.tile(x, x.shape[0])
        f4 = lambda x: x[0].flatten()
        f5 = lambda x: x[1].flatten()

        I = bm.concatenate(list(map(f2, cd)))
        J = bm.concatenate(list(map(f3, cd)))
        val0 = bm.concatenate(list(map(f4, K)))
        val1 = bm.concatenate(list(map(f5, K)))
        gdof = self.number_of_global_dofs()
        A = csr_matrix((val0, (I, J)), shape=(gdof, gdof), **self.fkwargs)
        S = csr_matrix((val1, (I, J)), shape=(gdof, gdof), **self.fkwargs)
        return A, S


    def edge_basis(self, bc):
        pass

    def basis(self, bc):
        pass

    def grad_basis(self, bc):
        pass

    def hessian_basis(self, bc):
        pass

    def dual_basis(self, u):
        pass

    def value(self, uh, bc):
        pass

    def grad_value(self, uh, bc):
        pass

    def hessian_value(self, uh, bc):
        pass

    def div_value(self, uh, bc):
        pass

    #def function(self, dim=None, array=None):
    #    f = Function(self, dim=dim, array=array)
    #    return f

    #def set_dirichlet_bc(self, gD, uh, threshold=None):
    #    """
    #    初始化解 uh  的第一类边界条件。
    #    """
    #    p = self.p
    #    NN = self.mesh.number_of_nodes()
    #    NE = self.mesh.number_of_edges()
    #    end = NN + (p - 1)*NE
    #    ipoints = self.interpolation_points()
    #    isDDof = self.boundary_dof(threshold=threshold)
    #    uh[isDDof] = gD(ipoints[isDDof[:end]])
    #    return isDDof


    #def array(self, dim=None, dtype=bm.float64):
    #    gdof = self.number_of_global_dofs()
    #    if dim is None:
    #        shape = gdof
    #    elif type(dim) is int:
    #        shape = (gdof, dim)
    #    elif type(dim) is tuple:
    #        shape = (gdof, ) + dim
    #    return bm.zeros(shape, dtype=dtype)

#    def matrix_D(self, H):
#        p = self.p
#        smldof = self.smspace.number_of_local_dofs()
#        mesh = self.mesh
#        NV = mesh.number_of_vertices_of_cells()
#        h = self.smspace.cellsize
#        node = mesh.entity('node')
#        edge = mesh.entity('edge')
#        edge2cell = mesh.edge_to_cell()
#        cell, cellLocation = mesh.entity('cell')
#        #cell = mesh._cell
#        #cellLocation = mesh.cellLocation
#        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
#
#        cell2dof, cell2dofLocation = self.cell_to_dof()
#        D = bm.ones((len(cell2dof), smldof), **self.fkwargs)
#
#        if p == 1:
#            bc = bm.repeat(self.smspace.cellbarycenter, NV, axis=0)
#            D[:, 1:] = (node[cell, :] - bc)/bm.repeat(h, NV).reshape(-1, 1)
#            return D
#
#        qf = GaussLobattoQuadrature(p+1)
#        bcs, ws = qf.quadpts, qf.weights
#        ps = bm.einsum('ij, kjm->ikm', bcs, node[edge])
#        phi0 = self.smspace.basis(ps[:-1], index=edge2cell[:, 0])
#        phi1 = self.smspace.basis(ps[p:0:-1, isInEdge, :], index=edge2cell[isInEdge, 1])
#        idx = cell2dofLocation[edge2cell[:, 0]] + edge2cell[:, 2]*p + bm.arange(p).reshape(-1, 1)
#        D[idx, :] = phi0
#        idx = cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p + bm.arange(p).reshape(-1, 1)
#        D[idx, :] = phi1
#        if p > 1:
#            area = self.smspace.cellmeasure
#            idof = (p-1)*p//2 # the number of dofs of scale polynomial space with degree p-2
#            idx = cell2dofLocation[1:].reshape(-1, 1) + bm.arange(-idof, 0)
#            D[idx, :] = H[:, :idof, :]/area.reshape(-1, 1, 1)
#        return D
#
#    def matrix_B(self):
#        p = self.p
#        smldof = self.smspace.number_of_local_dofs()
#        mesh = self.mesh
#        NV = mesh.number_of_vertices_of_cells()
#        h = self.smspace.cellsize
#        cell2dof, cell2dofLocation = self.cell_to_dof()
#        B = bm.zeros((smldof, cell2dof.shape[0]), **self.fkwargs)
#        if p == 1:
#            B[0, :] = 1/bm.repeat(NV, NV)
#            B[1:, :] = mesh.node_normal().T/bm.repeat(h, NV).reshape(1, -1)
#            return B
#        else:
#            idx = cell2dofLocation[0:-1] + NV*p
#            B[0, idx] = 1
#            idof = (p-1)*p//2
#            start = 3
#            r = bm.arange(1, p+1)
#            r = r[0:-1]*r[1:]
#            for i in range(2, p+1):
#                idx0 = bm.arange(start, start+i-1)
#                idx1 =  bm.arange(start-2*i+1, start-i)
#                idx1 = idx.reshape(-1, 1) + idx1
#                B[idx0, idx1] -= r[i-2::-1]
#                B[idx0+2, idx1] -= r[0:i-1]
#                start += i+1
#
#            node = mesh.entity('node')
#            edge = mesh.entity('edge')
#            edge2cell = mesh.edge_to_cell()
#            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
#
#            qf = GaussLobattoQuadrature(p + 1)
#            bcs, ws = qf.quadpts, qf.weights
#            ps = bm.einsum('ij, kjm->ikm', bcs, node[edge])
#            gphi0 = self.smspace.grad_basis(ps, index=edge2cell[:, 0])
#            gphi1 = self.smspace.grad_basis(ps[-1::-1, isInEdge, :], index=edge2cell[isInEdge, 1])
#            nm = mesh.edge_normal()
#
#            # m: the scaled basis number,
#            # j: the edge number,
#            # i: the virtual element basis number
#
#            NV = mesh.number_of_vertices_of_cells()
#
#            val = bm.einsum('i, ijmk, jk->mji', ws, gphi0, nm, optimize=True)
#            idx = cell2dofLocation[edge2cell[:, [0]]] + \
#                    (edge2cell[:, [2]]*p + bm.arange(p+1))%(NV[edge2cell[:, [0]]]*p)
#            bm.add.at(B, (bm.s_[:], idx), val)
#
#
#            if isInEdge.sum() > 0:
#                val = bm.einsum('i, ijmk, jk->mji', ws, gphi1, -nm[isInEdge], optimize=True)
#                idx = cell2dofLocation[edge2cell[isInEdge, 1]].reshape(-1, 1) + \
#                        (edge2cell[isInEdge, 3].reshape(-1, 1)*p + bm.arange(p+1)) \
#                        %(NV[edge2cell[isInEdge, 1]].reshape(-1, 1)*p)
#                bm.add.at(B, (bm.s_[:], idx), val)
#            return B
#
#    def matrix_G(self, B, D):
#        p = self.p
#        if p == 1:
#            G = bm.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
#        else:
#            cell2dof, cell2dofLocation = self.cell_to_dof()
#            BB = bm.hsplit(B, cell2dofLocation[1:-1])
#            DD = bm.vsplit(D, cell2dofLocation[1:-1])
#            g = lambda x: x[0]@x[1]
#            G = list(map(g, zip(BB, DD)))
#        return G
#
#    def matrix_C(self, H, PI1):
#        p = self.p
#
#        smldof = self.smspace.number_of_local_dofs()
#        idof = (p-1)*p//2
#
#        mesh = self.mesh
#        NV = mesh.number_of_vertices_of_cells()
#        d = lambda x: x[0]@x[1]
#        C = list(map(d, zip(H, PI1)))
#        if p == 1:
#            return C
#        else:
#            l = lambda x: bm.r_[
#                    '0',
#                    bm.r_['1', bm.zeros((idof, p*x[0])), x[1]*bm.eye(idof)],
#                    x[2][idof:, :]]
#            return list(map(l, zip(NV, self.smspace.cellmeasure, C)))
#
#    def matrix_PI_0(self, H, C):
#        cell2dof, cell2dofLocation = self.cell_to_dof()
#        pi0 = lambda x: inv(x[0])@x[1]
#        return list(map(pi0, zip(H, C)))
#
#    def matrix_PI_1(self, G, B):
#        p = self.p
#        cell2dof, cell2dofLocation = self.cell_to_dof()
#        if p == 1:
#            return bm.hsplit(B, cell2dofLocation[1:-1])
#        else:
#            BB = bm.hsplit(B, cell2dofLocation[1:-1])
#            g = lambda x: inv(x[0])@x[1]
#            return list(map(g, zip(G, BB)))


    #def function(self, dim=None, array=None, **self.fkwargs):
    #    return Function(self, dim=dim, array=array, coordtype='cartesian', dtype=dtype)

    #def set_dirichlet_bc(self, gD, uh, threshold=None):
