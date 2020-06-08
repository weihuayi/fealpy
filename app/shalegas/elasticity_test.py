#!/usr/bin/env python3
# 

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory

from fealpy.functionspace.femdof import multi_index_matrix2d

# 定义一个带裂缝的线弹性模型
class LinearElasticityModel():
    def __init__(self):
        self.mu = 1
        self.lam = 1.25

    def domain(self):
       return [0, 1, 0, 1] 

    def init_mesh(self):
        d = self.domain()
        mfactory = MeshFactory()
        mesh = mfactory.regular(d)
        return mesh

    def displacement(self, p):
        pass

    def jacobian(self, p):
        pass

    def strain(self, p):
        pass

    def stress(self, p):
        pass

    def source(self, p):
        pass

    def dirichlet(self, p):
        val = np.array([0.0])
        shape = len(p.shape)*(1,)
        return val.reshape(shape)

    def neumann(self, p):
        val = np.array([0.0, -1.0])
        shape = (len(p.shape) - 1)*(1, ) + (2, 0)
        return val.reshape(shape)

    def dirichlet_boundary(self, p, return_type='index'):
        """
        Get the Dirichlet boundary bool flag or int index

        """
        flag = p[..., 0] == 0
        if return_type == 'index':
            idx, = flag.nonzero()
            return idx 
        elif return_type == 'bool':
            return flag

    def neumann_boundary(self, p, return_type='index'):
        flag = p[..., 0] == 1 
        if return_type == 'index':
            idx, = flag.nonzero()
            return idx 
        elif return_type == 'bool':
            return flag

    def robin_boundary(self, p, return_type='index'):
        pass

    def fracture_boundary(self, p, return_type='index'):
        flag0 = (p[..., 1] == 0.5) & (p[..., 0] > 0.2) & (p[..., 0] < 0.8)
        flag1 = (p[..., 0] == 0.5) & (p[..., 1] > 0.2) & (p[..., 1] < 0.8)
        flag = flag0 | flag1
        if return_type == 'index':
            idx, = flag.nonzero()
            return idx 
        elif return_type == 'bool':
            return flag

## 定义一个线性有限元的自由度管理对象

class CPLFEMDofFracture2d():
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = multi_index_matrix2d(p)
        self.cell2dof = self.cell_to_dof()
        idx = mesh.meshdata['Fracture']
        
        NN = mesh.number_of_nodes()
        self.fracture = np.zeros(NN, dtype=np.bool)
        self.fracture[idx] = True

        valence = np.zeros(NN, dtype=np.int)

    def is_on_node_local_dof(self):
        p = self.p
        isNodeDof = np.sum(self.multiIndex == p, axis=-1) > 0
        return isNodeDof

    def is_on_edge_local_dof(self):
        return self.multiIndex == 0

    def boundary_dof(self, threshold=None):
        idx = self.mesh.ds.boundary_edge_index()
        if threshold is not None:
            bc = self.mesh.entity_barycenter('face', index=idx)
            flag = threshold(bc)
            idx  = idx[flag]
        gdof = self.number_of_global_dofs()
        edge2dof = self.edge_to_dof()
        isBdDof = np.zeros(gdof, dtype=np.bool)
        isBdDof[edge2dof[idx]] = True
        return isBdDof

    def face_to_dof(self):
        return self.edge_to_dof()

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh

        NE= mesh.number_of_edges()
        NN = mesh.number_of_nodes()

        edge = mesh.ds.edge
        edge2dof = np.zeros((NE, p+1), dtype=np.int)
        edge2dof[:, [0, -1]] = edge
        return edge2dof

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell

        N = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()

        cell2dof = cell

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')

        if p == 1:
            return node

        return ipoint

    def number_of_global_dofs(self):
        p = self.p
        NN = self.mesh.number_of_nodes()
        gdof = NN

    def number_of_local_dofs(self):
        return 3 



model = LinearElasticityModel()
mesh = model.init_mesh()
mesh.set_boundary_condition('Fracture', model.fracture_boundary) 
mesh.set_boundary_condition('Dirichlet', model.dirichlet_boundary)
mesh.set_boundary_condition('Neumann', model.neumann_boundary) 



fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_edge(axes, index=mesh.meshdata['Fracture'], ecolor='r')
mesh.find_edge(axes, index=mesh.meshdata['Dirichlet'], ecolor='b')
mesh.find_edge(axes, index=mesh.meshdata['Neumann'], ecolor='m')
plt.show()




