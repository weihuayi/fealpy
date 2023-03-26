import numpy as np

class EdgeMeshSpace():
    """
    Edge 网格空间，可用来计算 Truss，Frame, Beam
    """
    def __init__(self, mesh, p=1, spacetype='E', q=None, dof=None):
        self.mesh = mesh
        self.cellmeasure = mesh.entity_measure('cell')
        self.p = p
        self.dof = self.cell_to_dof()
        if len(mesh.node.shape) == 1:
            self.GD = 1
        else:
            self.GD = mesh.node.shape[1]
    
    def geo_dimension(self):
        return self.GD

    def cell_to_dof(self):
        '''
        自由度管理
        '''
        cell = self.mesh.entity('cell')
        GD = self.GD
        cell2dof = np.zeros((cell.shape[0], 2*GD), dtype=np.int_)
        for i in range(GD):
            cell2dof[:, i::GD] = cell + NN*i
        return cell2dof


