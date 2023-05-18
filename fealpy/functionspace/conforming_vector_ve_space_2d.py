import numpy as np
from numpy.linalg import inv

#from .Function import Function
#from .ScaledMonomialSpace2d import ScaledMonomialSpace2d


class ConformingNavierStokesVEMSpace2d:
    def __init__(self, mesh, p=1):
        self.mesh = mesh
        self.p = p
        self.itype = mesh.itype
        self.ftype = mesh.ftype

        self.dof = CNSVEMDof2d(mesh, p)


class CNSVEMDof2d:
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh

    def is_boundary_dof(self):
        mesh = self.mesh
        gdof = self.number_of_global_dofs()
        edge2dof = self.edge_to_dof()
        
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdEdge = mesh.ds.boundary_edge_flag()
        isBdDof[edge2dof[isBdEdge]] = True   
        return isBdDof

    def edge_to_dof(self):
        mesh = self.mesh
        p = self.p
        
        NE = mesh.number_of_edges()
        NN = mesh.number_of_nodes()
        edge = mesh.entity('edge')
        edge2dof = np.zeros((NE, (p+1)*2), dtype=np.int_)
        edge2dof[:, [0, p]] = edge
        edge2dof[:, [p+1, -1]] = edge + NN
        edge2dof[:, 1:p] = 2*NN + np.arange(NE*(p-1)).reshape(NE, p-1)
        edge2dof[:, p+2:-1] = edge2dof[:, 1:p] + NE*(p-1)    
        return edge2dof

    def cell_to_dof(self):
        mesh = self.mesh
        p = self.p
        ldof = self.number_of_local_dofs()
        edge2dof = self.edge_to_dof()
        
        
        isBdEdge = mesh.ds.boundary_edge_flag()
        edge2cell = mesh.ds.edge_to_cell()
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()
        NV = mesh.ds.number_of_vertices_of_cells()
        
        
        cell2dofLocation = np.zeros(NC+1, dtype=np.int_)  
        cell2dofLocation[1:] = np.add.accumulate(ldof)
        cell2dof = np.zeros(cell2dofLocation[-1], dtype=np.int_)
        
        rx = edge2dof[np.arange(NE)][:, 0:p+1]
        ry = edge2dof[np.arange(NE)][:, p+1:]
       
        idxrx = cell2dofLocation[edge2cell[:, 0]]+p*edge2cell[:, 2]
        idxrx = idxrx.reshape(-1,1)+np.arange(p+1)
        cell2dof[idxrx] = rx
            
        idxry = cell2dofLocation[edge2cell[:, 0]]+p*edge2cell[:, 2]+p*NV[edge2cell[:, 0]]
        idxry = idxry.reshape(-1,1)+np.arange(p+1)
        cell2dof[idxry] = ry            
       
        idxlx = cell2dofLocation[edge2cell[~isBdEdge][:, 1]]+p*edge2cell[~isBdEdge][:, 3]
        idxlx = idxlx.reshape(-1,1)+np.arange(p)
        cell2dof[idxlx] = rx[~isBdEdge][:, ::-1][..., 0:p]    
       
        idxly = cell2dofLocation[edge2cell[~isBdEdge][:, 1]]+p*edge2cell[~isBdEdge][:, 3]+p*NV[edge2cell[~isBdEdge][:, 1]]
        idxly = idxly.reshape(-1,1)+np.arange(p)
        cell2dof[idxly] = ry[~isBdEdge][:, ::-1][..., 0:p]
    
        
        idx1 = (2*NV*p).reshape(-1,1)+cell2dofLocation[0:-1].reshape(-1,1)+np.arange((p-2)*(p-1)/2)
        data1 = np.arange(((p-2)*(p-1)/2)*NC)+2*NN+2*NE*(p-1)
        cell2dof[idx1.flatten().astype(int)] = data1
    
        idx2 = (2*NV*p+(p-2)*(p-1)/2).reshape(-1,1)+cell2dofLocation[0:-1].reshape(-1,1)+np.arange(p*(p+1)/2-1)
        data2 = np.arange((p*(p+1)/2-1)*NC)+2*NN+2*NE*(p-1)+NC*((p-2)*(p-1)/2)
        cell2dof[idx2.flatten().astype(int)] = data2
        
        return np.hsplit(cell2dof, cell2dofLocation[1:-1])

    def number_of_global_dofs(self):
        """
        @brief 返回全局自由度的个数
        """
        return self.mesh.number_of_global_ipoints(self.p)*2
    
    def number_of_local_dofs(self, doftype='all'):
        """
        @brief  返回每个单元自由度的个数
        """
        return self.mesh.number_of_local_ipoints(self.p, iptype=doftype)*2
        
    def interpolation_points(self, index=np.s_[:], scale:float=0.4):
        p = self.p
        mesh = self.mesh
        node = mesh.entity('node')
        if p==1:
            return np.vstack((node,node))
        gdof = self.number_of_global_dofs()
        GD = mesh.geo_dimension()
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        
        start = 0
        ipoint = np.zeros((gdof,GD),dtype=np.float_)
        ipoint[start:2*NN,:] = np.vstack((node,node))
        
        start += 2*NN
        edge = self.mesh.entity('edge')
        qf = self.mesh.integrator(p+1, etype='edge', qtype='lobatto')
        bcs = qf.quadpts[1:-1, :]
        eips = np.einsum('ij, ...jm->...im', bcs, node[edge, :]).reshape(-1, GD)
        print(ipoint[start:(p-1)*NE*2,:].shape,(p-1)*NE*2)
        ipoint[start:start+(p-1)*NE*2, :] = np.vstack((eips,eips))  
        start += 2*(p-1)*NE
        if p == 2:
            ipoint[start:, :] = np.vstack((self.mesh.entity_barycenter('cell'),self.mesh.entity_barycenter('cell')))
            return ipoint
        h = np.sqrt(self.mesh.cell_area())[:, None]*scale
        bc = self.mesh.entity_barycenter('cell')
        t = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2]],dtype=np.float_)
        
        t -= np.array([0.5, np.sqrt(3)/6.0], dtype=np.float_)

        tri = np.zeros((NC, 3, GD))
        tri[:, 0, :] = bc + t[0]*h
        tri[:, 1, :] = bc + t[1]*h
        tri[:, 2, :] = bc + t[2]*h

        bcs = self.mesh.multi_index_matrix(2*p-2)/(2*p-2)
        ipoint[start:start+(p-2)*(p-1)//2,:] = np.einsum('ij, ...jm->...im',
                bcs[:(p-1)*(p-2)//2,:], tri).reshape(-1, GD)
        return ipoint



