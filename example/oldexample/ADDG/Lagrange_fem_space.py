import numpy as np
from scipy.sparse import csr_matrix
from fealpy.functionspace.Function import Function
from fealpy.functionspace.femdof import multi_index_matrix1d
from fealpy.functionspace.femdof import multi_index_matrix2d
from fealpy.functionspace.femdof import multi_index_matrix3d

from fealpy.functionspace.femdof import multi_index_matrix

from fealpy.functionspace.femdof import CPLFEMDof1d, CPLFEMDof2d, CPLFEMDof3d
from fealpy.functionspace.femdof import DPLFEMDof1d, DPLFEMDof2d, DPLFEMDof3d

from fealpy.quadrature import FEMeshIntegralAlg
from fealpy.decorator import timer


class LagrangeFiniteElementSpace():
    def __init__(self, mesh, p=1, spacetype='C', q=None, dof=None):
        self.mesh = mesh
        self.cellmeasure = mesh.entity_measure('cell')
        self.p = p
        if dof is None:
            if spacetype == 'C':
                if mesh.meshtype == 'interval':
                    self.dof = CPLFEMDof1d(mesh, p)
                    self.TD = 1
                elif mesh.meshtype == 'tri':
                    self.dof = CPLFEMDof2d(mesh, p)
                    self.TD = 2
                elif mesh.meshtype == 'halfedge2d':
                    assert mesh.ds.NV == 3
                    self.dof = CPLFEMDof2d(mesh, p)
                    self.TD = 2
                elif mesh.meshtype == 'stri':
                    self.dof = CPLFEMDof2d(mesh, p)
                    self.TD = 2
                elif mesh.meshtype == 'tet':
                    self.dof = CPLFEMDof3d(mesh, p)
                    self.TD = 3
            elif spacetype == 'D':
                if mesh.meshtype == 'interval':
                    self.dof = DPLFEMDof1d(mesh, p)
                    self.TD = 1
                elif mesh.meshtype == 'tri':
                    self.dof = DPLFEMDof2d(mesh, p)
                    self.TD = 2
                elif mesh.meshtype == 'tet':
                    self.dof = DPLFEMDof3d(mesh, p)
                    self.TD = 3
        else:
            self.dof = dof
            self.TD = mesh.top_dimension() 

        if len(mesh.node.shape) == 1:
            self.GD = 1
        else:
            self.GD = mesh.node.shape[1]

        self.spacetype = spacetype
        self.itype = mesh.itype
        self.ftype = mesh.ftype

        q = q if q is not None else p+3 
        self.integralalg = FEMeshIntegralAlg(
                self.mesh, q,
                cellmeasure=self.cellmeasure)
        self.integrator = self.integralalg.integrator

        self.multi_index_matrix = multi_index_matrix 
        self.stype = 'lagrange'

    def __str__(self):
        return "Lagrange finite element space!"

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def cell_to_dof(self):
        return self.dof.cell2dof

    def boundary_dof(self):
        return self.dof.boundary_dof()

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    
    def bc_to_point_cell(self,bcs):
        mesh = self.mesh
        node = mesh.node
        cell = mesh.ds.cell
        pp = np.einsum('...j, ijk->...ik', bcs, node[cell])
        return pp
    
   
    def bc_to_point_edge(self,IntervalQuadrature): 
        bcs = IntervalQuadrature.quadpts
        mesh = self.mesh
        node = mesh.node
        edge = mesh.entity('edge')
        p1 = np.einsum('ij,k->kij',(node[edge[:,1]] + node[edge[:,0]])/2, np.ones(len(bcs)))
        pp = np.einsum('ij,k ->kij',(node[edge[:,1]] - node[edge[:,0]])/2,bcs) + p1
        return pp
    
    def Basis_function(self,pp,cellidx):
        mesh = self.mesh
        node = mesh.node
        cell = mesh.ds.cell
        measure = mesh.entity_measure('cell')
        ex = pp[..., 0].T
        ey = pp[..., 1].T
        x = node[:,0:1]
        y = node[:,1:2]
        localEdge = np.array(
            [(1, 2),
             (2, 0),
             (0, 1)], dtype=np.int)
        K = cell[cellidx]
        x_ek = x[K]
        y_ek = y[K]
        a = np.einsum('mjk,m->mj', y_ek[:,localEdge[:,0]] - y_ek[:,localEdge[:,1]], 1/(2*measure[cellidx]))
        b = np.einsum('mjk,m->mj', x_ek[:,localEdge[:,1]] - x_ek[:,localEdge[:,0]], 1/(2*measure[cellidx]))
        c = (np.einsum('ijk, ijk->ij', x_ek[:,localEdge[:,0]], y_ek[:,localEdge[:,1]])-np.einsum('ijk, ijk->ij', x_ek[:,localEdge[:,1]], y_ek[:,localEdge[:,0]]))#所有K_+单元的常系数
        c = np.einsum('t, mj, m->tmj', np.ones(len(ex[0,:])), c, 1/(2*measure[cellidx]))
        Basis = np.einsum('mt,mj ->tmj', ex,a)+ np.einsum('mt,mj ->tmj', ey,b)+c#所有K_+单元基函数
        return Basis,a,b,c

    def Hat_function(self,pp,lidx):
        Basis,a,b,c = self.Basis_function(pp,lidx)
        p=self.p
        if p==2:
            dofs1 = [1,2,0]
            dofs2 = [2,0,1]
            Hat = np.c_[Basis*(2*Basis-1),4*Basis[:,:,dofs1]*Basis[:,:,dofs2]]
            dof = [0,5,4,1,3,2]
            
        if p==3:
            dofs1 = [1,2,2,0,0,1]
            dofs2 = [2,1,0,2,1,0]
            Hat1 = 1/2*Basis*(3*Basis-1)*(3*Basis-2)
            Hat2 = 9/2*Basis[:,:,dofs1]*Basis[:,:,dofs2]*(3*Basis[:,:,dofs1]-1)
            Hat3 = 27*Basis[:,:,0:1]*Basis[:,:,1:2]*Basis[:,:,2:3]
            Hat = np.c_[Hat1,Hat2,Hat3]
            dof = [0,7,6,8,9,5,1,3,4,2]
            
        if p==4:
            dofs = [1,2,0]
            H = Basis[:,:,0:1]*Basis[:,:,1:2]*Basis[:,:,2:3]
            Hat1 = 1/6*Basis*(4*Basis-1)*(4*Basis-2)*(4*Basis-3)
            Hat2 = 8/3*Basis*Basis[:,:,dofs]*(4*Basis[:,:,dofs]-1)*(4*Basis[:,:,dofs]-2)
            Hat3 = 8/3*Basis[:,:,dofs]*Basis*(4*Basis-1)*(4*Basis-2)
            Hat4 = 4*Basis[:,:,dofs]*Basis*(4*Basis-1)*(4*Basis[:,:,dofs]-1)
            Hat5 = 32*(4*Basis - 1)*H
            Hat = np.c_[Hat1,Hat2,Hat3,Hat4,Hat5]
            dof = [0,6,5,9,12,11,3,13,14,8,1,7,10,4,2]
        Hat = Hat[:,:,dof]
        return Hat
    
    def Hat_Gradients(self,pp,cellidx):
        Basis,a,b,c = self.Basis_function(pp,cellidx)
        p=self.p 
        if p==2:
            dofs1 = [1,2,0]
            dofs2 = [2,0,1]
            x1 = np.einsum('mj,tmj ->tmj',a,(4*Basis-1))
            x2 = np.einsum('mj,tmj ->tmj',4*a[:,dofs1],Basis[:,:,dofs2])+np.einsum('mj,tmj ->tmj',4*a[:,dofs2],Basis[:,:,dofs1])
            
            y1 = np.einsum('mj,tmj ->tmj', b,(4*Basis-1))
            y2 = np.einsum('mj,tmj ->tmj',4*b[:,dofs1],Basis[:,:,dofs2])+np.einsum('mj,tmj ->tmj',4*b[:,dofs2],Basis[:,:,dofs1])
            Hat_x = np.c_[x1,x2]
            Hat_y = np.c_[y1,y2]
            dof = [0,5,4,1,3,2]
            
        if p==3:
            dofs1 = [1,2,2,0,0,1]
            dofs2 = [2,1,0,2,1,0]
            x1 = np.einsum('mj,tmj ->tmj',a,(27/2*Basis**2-9*Basis+1))
            x2 = np.einsum('mj,tmj ->tmj',a[:,dofs2],9/2*Basis[:,:,dofs1]*(3*Basis[:,:,dofs1]-1))+np.einsum('mj,tmj ->tmj',a[:,dofs1],27*Basis[:,:,dofs1]*Basis[:,:,dofs2]-9/2*Basis[:,:,dofs2])
            x3 = np.einsum('mj,tmj,tmj ->tmj',27*a[:,0:1],Basis[:,:,1:2],Basis[:,:,2:3])+np.einsum('mj,tmj,tmj ->tmj',27*a[:,1:2],Basis[:,:,0:1],Basis[:,:,2:3])+np.einsum('mj,tmj,tmj ->tmj',27*a[:,2:3],Basis[:,:,0:1],Basis[:,:,1:2])
            
            y1 = np.einsum('mj,tmj ->tmj',b,(27/2*Basis**2-9*Basis+1))
            y2 = np.einsum('mj,tmj ->tmj',b[:,dofs2],9/2*Basis[:,:,dofs1]*(3*Basis[:,:,dofs1]-1))+np.einsum('mj,tmj ->tmj',b[:,dofs1],27*Basis[:,:,dofs1]*Basis[:,:,dofs2]-9/2*Basis[:,:,dofs2])
            y3 = np.einsum('mj,tmj,tmj ->tmj',27*b[:,0:1],Basis[:,:,1:2],Basis[:,:,2:3])+np.einsum('mj,tmj,tmj ->tmj',27*b[:,1:2],Basis[:,:,0:1],Basis[:,:,2:3])+np.einsum('mj,tmj,tmj ->tmj',27*b[:,2:3],Basis[:,:,0:1],Basis[:,:,1:2])
            Hat_x = np.c_[x1,x2,x3]
            Hat_y = np.c_[y1,y2,y3]
            dof = [0,7,6,8,9,5,1,3,4,2]
            
        if p==4:
            dofs = [1,2,0]
            dofs2 = [2,0,1]
            H = Basis[:,:,0:1]*Basis[:,:,1:2]*Basis[:,:,2:3]
            x1 = np.einsum('mj,tmj ->tmj',a,(128/3*Basis**3-48*Basis**2+44/3*Basis-1))
            x2 = np.einsum('mj,tmj ->tmj',a[:,dofs],128*Basis*Basis[:,:,dofs]**2-64*Basis*Basis[:,:,dofs] + 16/3*Basis)+np.einsum('mj,tmj ->tmj',a,128/3*Basis[:,:,dofs]**3-32*Basis[:,:,dofs]**2+16/3*Basis[:,:,dofs])
            x3 = np.einsum('mj,tmj ->tmj',a,128*Basis[:,:,dofs]*Basis**2-64*Basis*Basis[:,:,dofs] + 16/3*Basis[:,:,dofs])+np.einsum('mj,tmj ->tmj',a[:,dofs],128/3*Basis**3-32*Basis**2+16/3*Basis)
            x4 = np.einsum('mj,tmj ->tmj',a,128*Basis*Basis[:,:,dofs]**2-32*Basis*Basis[:,:,dofs] - 16*Basis[:,:,dofs]**2 + 4*Basis[:,:,dofs])+np.einsum('mj,tmj ->tmj',a[:,dofs],128*Basis[:,:,dofs]*Basis**2-32*Basis*Basis[:,:,dofs] - 16*Basis**2 + 4*Basis)
            x5 = np.einsum('mj,tmj ->tmj',a,32*Basis[:,:,dofs]*Basis[:,:,dofs2]*(4*Basis-1) + 128*H)+np.einsum('mj,tmj ->tmj',a[:,dofs],32*Basis*Basis[:,:,dofs2]*(4*Basis-1))+np.einsum('mj,tmj ->tmj',a[:,dofs2],32*Basis*Basis[:,:,dofs]*(4*Basis-1))
            
            y1 = np.einsum('mj,tmj ->tmj',b,(128/3*Basis**3-48*Basis**2+44/3*Basis-1))
            y2 = np.einsum('mj,tmj ->tmj',b[:,dofs],128*Basis*Basis[:,:,dofs]**2-64*Basis*Basis[:,:,dofs] + 16/3*Basis)+np.einsum('mj,tmj ->tmj',b,128/3*Basis[:,:,dofs]**3-32*Basis[:,:,dofs]**2+16/3*Basis[:,:,dofs])
            y3 = np.einsum('mj,tmj ->tmj',b,128*Basis[:,:,dofs]*Basis**2-64*Basis*Basis[:,:,dofs] + 16/3*Basis[:,:,dofs])+np.einsum('mj,tmj ->tmj',b[:,dofs],128/3*Basis**3-32*Basis**2+16/3*Basis)
            y4 = np.einsum('mj,tmj ->tmj',b,128*Basis*Basis[:,:,dofs]**2-32*Basis*Basis[:,:,dofs] - 16*Basis[:,:,dofs]**2 + 4*Basis[:,:,dofs])+np.einsum('mj,tmj ->tmj',b[:,dofs],128*Basis[:,:,dofs]*Basis**2-32*Basis*Basis[:,:,dofs] - 16*Basis**2 + 4*Basis)
            y5 = np.einsum('mj,tmj ->tmj',b,32*Basis[:,:,dofs]*Basis[:,:,dofs2]*(4*Basis-1) + 128*H)+np.einsum('mj,tmj ->tmj',b[:,dofs],32*Basis*Basis[:,:,dofs2]*(4*Basis-1))+np.einsum('mj,tmj ->tmj',b[:,dofs2],32*Basis*Basis[:,:,dofs]*(4*Basis-1))
            Hat_x = np.c_[x1,x2,x3,x4,x5]
            Hat_y = np.c_[y1,y2,y3,y4,y5]
            dof = [0,6,5,9,12,11,3,13,14,8,1,7,10,4,2]
        Hat_x =  Hat_x[:,:,dof]
        Hat_y =  Hat_y[:,:,dof]
        shape = Hat_x.shape + (1,)
        Hat_x = Hat_x.reshape(shape)
        Hat_y = Hat_y.reshape(shape)
        G_Hat = np.c_[Hat_x,Hat_y]  
        return G_Hat
    
    def Hat_2Gradients(self,pp,cellidx):
        Basis,a,b,c = self.Basis_function(pp,cellidx)
        p=self.p
        if p==2:
            dofs1 = [1,2,0]
            dofs2 = [2,0,1]
            H_xx1 = 4*np.einsum('mj,mj ->mj',a,a)
            H_xx2 = 8*np.einsum('mj,mj ->mj',a[:,dofs1],a[:,dofs2])
            H_xx = np.c_[H_xx1,H_xx2]
            H_xy1 = 4*np.einsum('mj,mj ->mj',a,b)
            H_xy2 = 4*np.einsum('mj,mj ->mj',a[:,dofs1],b[:,dofs2]) + 4*np.einsum('mj,mj ->mj',a[:,dofs2],b[:,dofs1])
            H_xy = np.c_[H_xy1,H_xy2]
            
            H_yy1 = 4*np.einsum('mj,mj ->mj',b,b)
            H_yy2 = 8*np.einsum('mj,mj ->mj',b[:,dofs1],b[:,dofs2])
            
            H_yy = np.c_[H_yy1,H_yy2]
            shape = (1,) + H_xx.shape
            H_xx = H_xx.reshape(shape)
            H_xy = H_xy.reshape(shape)
            H_yy = H_yy.reshape(shape)
            dof = [0,5,4,1,3,2]
            
        if p==3:
            dofs1 = [1,2,2,0,0,1]
            dofs2 = [2,1,0,2,1,0]
            dofs3 = [0,0,1,1,2,2]
            C1 = np.einsum('mj,mj ->mj',a[:,dofs1],b[:,dofs2])
            C2 = np.einsum('mj,mj ->mj',a[:,dofs2],b[:,dofs1])
            
            H_xx1 = np.einsum('mj,tmj ->tmj',a**2,27*Basis-9)
            H_xx2 = np.einsum('mj,tmj ->tmj',a[:,dofs1]**2,27*Basis[:,:,dofs2])+np.einsum('mj,mj,tmj ->tmj',a[:,dofs1],a[:,dofs2],(54*Basis[:,:,dofs1]-9))
            H_xx3 = np.einsum('mj,mj,tmj ->tm',a[:,dofs1],a[:,dofs3],27*Basis[:,:,dofs2])
            shape = H_xx3.shape+(1,)
            H_xx3 = H_xx3.reshape(shape)
            H_xx = np.c_[H_xx1,H_xx2,H_xx3]
            
            H_xy1 = np.einsum('mj,mj,tmj ->tmj',a,b,27*Basis-9)
            H_xy2 = np.einsum('mj,tmj ->tmj',C1+C2,27*Basis[:,:,dofs1]-9/2)+np.einsum('mj,mj,tmj ->tmj',a[:,dofs1],b[:,dofs1],27*Basis[:,:,dofs2])
            H_xy3 = np.einsum('mj,mj,tmj ->tm',a[:,dofs1],b[:,dofs3],27*Basis[:,:,dofs2])
            H_xy3 = H_xy3.reshape(shape)
            H_xy = np.c_[H_xy1,H_xy2,H_xy3]
            
            H_yy1 = np.einsum('mj,tmj ->tmj',b**2,27*Basis-9)
            H_yy2 = np.einsum('mj,tmj ->tmj',b[:,dofs1]**2,27*Basis[:,:,dofs2])+np.einsum('mj,mj,tmj ->tmj',b[:,dofs1],b[:,dofs2],(54*Basis[:,:,dofs1]-9))
            H_yy3 = np.einsum('mj,mj,tmj ->tm',b[:,dofs1],b[:,dofs3],27*Basis[:,:,dofs2])
            H_yy3 = H_yy3.reshape(shape)
            H_yy = np.c_[H_yy1,H_yy2,H_yy3]
            dof = [0,7,6,8,9,5,1,3,4,2]
            
        if p==4:
            dofs = [1,2,0]
            dofs2 = [2,0,1]
            c1 = a[:,0]*b[:,1]+a[:,1]*b[:,0]
            c2 = a[:,1]*b[:,2]+a[:,2]*b[:,1]
            c3 = a[:,2]*b[:,0]+a[:,0]*b[:,2]
            d1 = np.c_[c1,c2,c3]
            d2 = np.c_[c3,c1,c2]
            d3 = np.c_[c2,c3,c1]
        
            H_xx1 = np.einsum('mj,tmj ->tmj',a**2,128*Basis**2-96*Basis+44/3)
            H_xx2 = np.einsum('mj,tmj ->tmj',a[:,dofs]**2,256*Basis*Basis[:,:,dofs]-64*Basis)+np.einsum('mj,mj,tmj ->tmj',a,a[:,dofs],256*Basis[:,:,dofs]**2-128*Basis[:,:,dofs]+32/3)
            H_xx3 = np.einsum('mj,tmj ->tmj',a**2,256*Basis*Basis[:,:,dofs]-64*Basis[:,:,dofs])+np.einsum('mj,mj,tmj ->tmj',a,a[:,dofs],256*Basis**2-128*Basis+32/3)
            H_xx4 = np.einsum('mj,tmj ->tmj',a**2,128*Basis[:,:,dofs]**2-32*Basis[:,:,dofs])+np.einsum('mj,tmj ->tmj',a[:,dofs]**2,128*Basis**2-32*Basis) + np.einsum('mj,mj,tmj ->tmj',a,a[:,dofs],512*Basis*Basis[:,:,dofs]-64*Basis-64*Basis[:,:,dofs] + 8)
            H_xx5 = np.einsum('mj,tmj ->tmj',64*a*a[:,dofs],Basis[:,:,dofs2]*(8*Basis-1)) + np.einsum('mj,tmj ->tmj',64*a*a[:,dofs2],Basis[:,:,dofs]*(8*Basis-1))+ np.einsum('mj,tmj ->tmj',64*a[:,dofs]*a[:,dofs2],Basis*(4*Basis-1)) + np.einsum('mj,tmj ->tmj',256*a**2,Basis[:,:,dofs]*Basis[:,:,dofs2])
            H_xx = np.c_[H_xx1,H_xx2,H_xx3,H_xx4,H_xx5]
            
            
            H_yy1 = np.einsum('mj,tmj ->tmj',b**2,128*Basis**2-96*Basis+44/3)
            H_yy2 = np.einsum('mj,tmj ->tmj',b[:,dofs]**2,256*Basis*Basis[:,:,dofs]-64*Basis)+np.einsum('mj,mj,tmj ->tmj',b,b[:,dofs],256*Basis[:,:,dofs]**2-128*Basis[:,:,dofs]+32/3)
            H_yy3 = np.einsum('mj,tmj ->tmj',b**2,256*Basis*Basis[:,:,dofs]-64*Basis[:,:,dofs])+np.einsum('mj,mj,tmj ->tmj',b,b[:,dofs],256*Basis**2-128*Basis+32/3)
            H_yy4 = np.einsum('mj,tmj ->tmj',b**2,128*Basis[:,:,dofs]**2-32*Basis[:,:,dofs])+np.einsum('mj,tmj ->tmj',b[:,dofs]**2,128*Basis**2-32*Basis) + np.einsum('mj,mj,tmj ->tmj',b,b[:,dofs],512*Basis*Basis[:,:,dofs]-64*Basis-64*Basis[:,:,dofs] + 8)
            H_yy5 = np.einsum('mj,tmj ->tmj',64*b*b[:,dofs],Basis[:,:,dofs2]*(8*Basis-1)) + np.einsum('mj,tmj ->tmj',64*b*b[:,dofs2],Basis[:,:,dofs]*(8*Basis-1))+ np.einsum('mj,tmj ->tmj',64*b[:,dofs]*b[:,dofs2],Basis*(4*Basis-1)) + np.einsum('mj,tmj ->tmj',256*b**2,Basis[:,:,dofs]*Basis[:,:,dofs2])
            H_yy = np.c_[H_yy1,H_yy2,H_yy3,H_yy4,H_yy5]
            
            
            H_xy1 = np.einsum('mj,mj,tmj ->tmj',a,b,128*Basis**2-96*Basis+44/3)
            H_xy2 = np.einsum('mj,mj,tmj ->tmj',a[:,dofs],b[:,dofs],256*Basis*Basis[:,:,dofs]-64*Basis)+np.einsum('mj,tmj ->tmj',d1,128*Basis[:,:,dofs]**2-64*Basis[:,:,dofs]+16/3)
            H_xy3 = np.einsum('mj,mj,tmj ->tmj',a,b,256*Basis*Basis[:,:,dofs]-64*Basis[:,:,dofs])+np.einsum('mj,tmj ->tmj',d1,128*Basis**2-64*Basis+16/3)
            H_xy4 = np.einsum('mj,mj,tmj ->tmj',a,b,128*Basis[:,:,dofs]**2-32*Basis[:,:,dofs])+np.einsum('mj,mj,tmj ->tmj',a[:,dofs],b[:,dofs],128*Basis**2-32*Basis) + np.einsum('mj,tmj ->tmj',d1,256*Basis*Basis[:,:,dofs]-32*Basis-32*Basis[:,:,dofs] + 4)
            H_xy5 = np.einsum('mj,tmj ->tmj',32*d1,Basis[:,:,dofs2]*(8*Basis-1)) + np.einsum('mj,tmj ->tmj',32*d2,Basis[:,:,dofs]*(8*Basis-1))+ np.einsum('mj,tmj ->tmj',32*d3,Basis*(4*Basis-1)) + np.einsum('mj,mj,tmj ->tmj',256*a,b,Basis[:,:,dofs]*Basis[:,:,dofs2])
            H_xy = np.c_[H_xy1,H_xy2,H_xy3,H_xy4,H_xy5]
            dof = [0,6,5,9,12,11,3,13,14,8,1,7,10,4,2]
        H_xx =  H_xx[:,:,dof]
        H_xy =  H_xy[:,:,dof]
        H_yy =  H_yy[:,:,dof]
        return H_xx,H_xy,H_yy
    
    def DDG_stiff_matrix(self,qf, cfun=None):
        bcs =qf.quadpts
        ws = qf.weights
        mesh = self.mesh
        GD = mesh.node.shape[1]
        NC = mesh.number_of_cells()
        pp = self.bc_to_point_cell(bcs)
        cellidx = np.arange(NC)
        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()
        G_Basis = self.Hat_Gradients(pp,cellidx)
        if cfun is not None:
            d = cfun(pp)
            if isinstance(d, (int, float)):
                dG_Basis = d*G_Basis
            elif len(d) == GD:
                dG_Basis = np.einsum('m, ...im->...im', d, G_Basis)
            elif isinstance(d, np.ndarray):
                if len(d.shape) == 1:
                    dG_Basis = np.einsum('i, ...imn->...imn', d, G_Basis)
                elif len(d.shape) == 2:
                    dG_Basis = np.einsum('...i, ...imn->...imn', d, G_Basis)
                elif len(d.shape) == 3: #TODO:
                    dG_Basis = np.einsum('...imn, ...in->...im', d, G_Basis)
                elif len(d.shape) == 4: #TODO:
                    dG_Basis = np.einsum('...imn, ...ijn->...ijm', d, G_Basis)
                else:
                    raise ValueError("The ndarray shape length should < 5!")
            else:
                raise ValueError(
                        "The return of cfun is not a number or ndarray!"
                        )
        else:
            dG_Basis = G_Basis
        A = np.einsum('t, tmkj, tmpj,m->mkp', ws, dG_Basis, G_Basis,self.cellmeasure)
        
        cell2dof = self.cell_to_dof()
        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2) 
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return A

    def DDG_source_vector(self,f,qf):
        bcs = qf.quadpts
        ws = qf.weights
        mesh = self.mesh
        NC = mesh.number_of_cells()
        pp = self.bc_to_point_cell(bcs)
        cellidx = np.arange(NC)
        
        Hat = self.Hat_function(pp,cellidx)
        fval = f(pp)
        
        F = np.einsum('t, tm, tmk, m->mk', ws, fval, Hat,self.cellmeasure)
        B = F.reshape(-1)
        return B
    
    def jump(self,IntervalQuadrature):
        mesh = self.mesh
        index = mesh.ds.boundary_edge_index()
        pp = self.bc_to_point_edge(IntervalQuadrature)
        edge2cell = mesh.ds.edge_to_cell()
        cellidx_0 = edge2cell[:,0]
        cellidx_1 = edge2cell[:,1]
        Hat1 = self.Hat_function(pp,cellidx_0)
        Hat2 = self.Hat_function(pp,cellidx_1)
        Hat2[:,index] = 0*Hat2[:,index]
        jump = np.append(Hat1,-Hat2,axis = 2)#t*NE*2ldof
        return jump
    
    def grad_average(self,IntervalQuadrature,cfun,coefficient='C'):
        mesh = self.mesh
        index = mesh.ds.boundary_edge_index()
        pp = self.bc_to_point_edge(IntervalQuadrature)
        edge2cell = mesh.ds.edge_to_cell()
        cellidx_0 = edge2cell[:,0]
        cellidx_1 = edge2cell[:,1]
        G_Hat1 =  self.Hat_Gradients(pp,cellidx_0)
        G_Hat2 =  self.Hat_Gradients(pp,cellidx_1)
        G_Hat2[:,index] = 0*G_Hat2[:,index]
        G_Hat1[:,index] = 2*G_Hat1[:,index]
        d = cfun(pp)
        dG_Hat1 = np.einsum('...i, ...imn->...imn', d, G_Hat1)
        dG_Hat2 = np.einsum('...i, ...imn->...imn', d, G_Hat2)
        if coefficient =='D':
            bcs = np.array([1/3,1/3,1/3])
            ps = self.bc_to_point_cell(bcs)
            d = cfun(ps)
            dG_Hat1 = np.einsum('...i, ...imn->...imn', d[cellidx_0], G_Hat1)
            dG_Hat2 = np.einsum('...i, ...imn->...imn', d[cellidx_1], G_Hat2)
        dG_Hat= np.append(1/2*dG_Hat1,1/2*dG_Hat2,axis = 2)
        return dG_Hat
    
    def Hesse_jump(self,IntervalQuadrature):
        mesh = self.mesh
        index = mesh.ds.boundary_edge_index()
        pp = self.bc_to_point_edge(IntervalQuadrature)
        edge2cell = mesh.ds.edge_to_cell()
        ne = mesh.edge_unit_normal()
        cellidx_0 = edge2cell[:,0]
        cellidx_1 = edge2cell[:,1]
        H_xx1,H_xy1,H_yy1 = self.Hat_2Gradients(pp,cellidx_0)
        H_xx2,H_xy2,H_yy2 = self.Hat_2Gradients(pp,cellidx_1)
        H_xx = np.append(H_xx1,-H_xx2,axis = 2)
        H_xy = np.append(H_xy1,-H_xy2,axis = 2)
        H_yy = np.append(H_yy1,-H_yy2,axis = 2)
        Hesse = np.einsum('m, tmj->tmj', ne[:,0]**2, H_xx) + 2*np.einsum('m, tmj->tmj', ne[:,0]*ne[:,1], H_xy)+ np.einsum('m, tmj->tmj', ne[:,1]**2, H_yy)
        Hesse[:,index] = 0*Hesse[:,index]
        return Hesse
    
    def diffusion_coefficient(self,cfun):
        mesh = self.mesh
        p = mesh.barycenter()
        kval = cfun(p)
        return kval
    
    def We(self,IntervalQuadrature,cfun,indx,coefficient='C'):
        mesh = self.mesh
        pp = self.bc_to_point_edge(IntervalQuadrature)
        we = cfun(pp)
        if coefficient =='D':
            shape = pp.shape[0]
            bcs = np.array([[1/3,1/3,1/3]])
            bcs = np.repeat(bcs,shape).reshape(shape,-1)
            p = self.bc_to_point_cell(bcs)
            edge2cell = mesh.ds.edge_to_cell()
            lidx1 = edge2cell[:,0]
            lidx2 = edge2cell[:,1]
            kval = cfun(p)
            if indx == 1:
                we = 1/2*(kval[:,lidx1]+kval[:,lidx2])
            elif indx ==2:
                we = (2*kval[:,lidx1]*kval[:,lidx2])/(kval[:,lidx1]+kval[:,lidx2])
            elif indx ==3:
                we = np.sqrt(kval[:,lidx1]*kval[:,lidx2])
        return we
    
    def edge_matrix(self,IntervalQuadrature,cfun,indx,Cdtype):
        ws = IntervalQuadrature.weights
        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        length = mesh.edge_length()
        ne = mesh.edge_unit_normal()
        we = self.We(IntervalQuadrature,cfun,indx,Cdtype)
        jump = self.jump(IntervalQuadrature)
        P = np.einsum('t, tmk, tmp, tm->mkp', ws, jump, jump,we,optimize = True)
        
        G_Hat = self.grad_average(IntervalQuadrature,cfun,Cdtype)
        S = 1/2*np.einsum('t,tmk,tmpq,mq,m->mkp',ws, jump, G_Hat, ne, length,optimize = True)
        
        Hesse = self.Hesse_jump(IntervalQuadrature)
        H = np.einsum('t, tmk, jmi ,m, tm->mki', ws, jump, Hesse , length**2 ,we,optimize = True)
        
        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof()

        dof1  = cell2dof[edge2cell[:,0]]
        dof2  = cell2dof[edge2cell[:,1]]
        dof = np.append(dof1,dof2,axis = 1)
        I = np.einsum('k, ij->ijk', np.ones(2*ldof), dof)
        J = I.swapaxes(-1, -2)
        
        # Construct the flux matrix
        P = csr_matrix((P.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        S = csr_matrix((S.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        H = csr_matrix((H.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return P,S,H
    
    def Laplace_uh(self, uh, qf, cellidx=None):
        bcs, ws = qf.quadpts, qf.weights
        mesh = self.mesh
        pp = self.bc_to_point_cell(bcs)
        NC = mesh.number_of_cells()
        lidx = np.arange(NC)
        cell2dof = self.dof.cell2dof
        H_xx,H_xy,H_yy = self.Hat_2Gradients(pp,lidx)
        if cellidx is None:
            L_uh = np.einsum('t,tmj, mj->m',ws, H_xx, uh[cell2dof]) + np.einsum('t,tmj, mj->m',ws, H_yy, uh[cell2dof])
        else:
            L_uh = np.einsum('t,tmj, mj->m', ws,H_xx, uh[cell2dof[cellidx]]) + np.einsum('t,tmj, mj->m', ws,H_yy, uh[cell2dof[cellidx]])
        return L_uh
    
    def Laplace_value(self, uh, qf,cfun=None):
        bcs = qf.quadpts
        mesh = self.mesh
        pp = self.bc_to_point_cell(bcs)
        NC = mesh.number_of_cells()
        lidx = np.arange(NC)
        H_xx,H_xy,H_yy = self.Hat_2Gradients(pp,lidx)
        if cfun is not None:
            d = cfun(pp)
            dH_xx = np.einsum('...i, ...im->...im', d, H_xx)
            dH_yy = np.einsum('...i, ...im->...im', d, H_yy)
        else:
            dH_xx = H_xx
            dH_yy = H_yy
        return dH_xx + dH_yy
    
    def value(self, uh, bcs, cellidx=None):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        lidx = np.arange(NC)
        pp = self.bc_to_point_cell(bcs)
        phi = self.Hat_function(pp,lidx)
        phi = phi.swapaxes(0, 1)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        if cellidx is None:
            val = np.einsum(s1, phi, uh[cell2dof])
        else:
            val = np.einsum(s1, phi, uh[cell2dof[cellidx]])
        return val
    
    def grad_value(self, uh, bcs, cellidx=None):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        pp = self.bc_to_point_cell(bcs)
        lidx = np.arange(NC)
        gphi = self.Hat_Gradients(pp,lidx)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        if cellidx is None:
            val = np.einsum(s1, gphi, uh[cell2dof])
        else:
            val = np.einsum(s1, gphi, uh[cell2dof[cellidx]])
        return val

    def div_value(self, uh, bc, cellidx=None):
        dim = len(uh.shape)
        gdim = self.geo_dimension()
        if (dim == 2) & (uh.shape[1] == gdim):
            val = self.grad_value(uh, bc, cellidx=cellidx)
            return val.trace(axis1=-2, axis2=-1)
        else:
            raise ValueError("The shape of uh should be (gdof, gdim)!")

    def interpolation(self, u, dim=None):
        ipoint = self.dof.interpolation_points()
        uI = Function(self, dim=dim)
        uI[:] = u(ipoint)
        return uI

    def projection(self, u, up):
        pass

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=self.ftype)
