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
        NV = mesh.number_of_vertices_of_cells()
        
        
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
        
        return cell2dof, cell2dofLocation

    def number_of_global_dofs(self):
        mesh = self.mesh
        p = self.p
        
        NE = mesh.number_of_edges()
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        gdof = 2*NN
        gdof += 2*NE*(p-1)
        gdof += NC*((p-1)*p)
        
        return gdof
    
    def number_of_local_dofs(self):
        mesh = self.mesh
        p = self.p
        
        NV = mesh.number_of_vertices_of_cells()
        ldof = 2*p*NV+(p-1)*p
        
        return ldof
        


