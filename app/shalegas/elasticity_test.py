#!/usr/bin/env python3
# 

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory

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
        flag = (p[..., 1] == 0.5) & (p[..., 0] > 0.2) & (p[..., 0] < 0.8)
        if return_type == 'index':
            idx, = flag.nonzero()
            return idx 
        elif return_type == 'bool':
            return flag


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




