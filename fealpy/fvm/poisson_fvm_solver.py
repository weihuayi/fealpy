from fealpy.backend import backend_manager as bm
from fealpy.mesh import *
from scipy.sparse import coo_matrix


class PoissonFvmSolver():
    def __init__(self, pde):

        self.pde = pde
        self.h = 1/pde.nx
        self.NC = pde.mesh.number_of_cells()
        self.e2e = pde.mesh.cell_to_cell()
        self.flag = self.e2e[bm.arange(self.NC)] == bm.arange(self.NC)[:, None] # 判断是边界的


    def Poisson_LForm(self):

        I = bm.arange(self.NC)
        J = bm.arange(self.NC)
        val = 4*self.h/self.h*bm.ones(self.NC)
        A0 = coo_matrix((val,(I, J)), shape=(self.NC, self.NC))

        I = bm.where(~self.flag)[0] 
        J = self.e2e[~self.flag]
        val = -self.h/self.h*bm.ones(I.shape)
        A0 += coo_matrix((val,(I, J)), shape=(self.NC, self.NC))

        return A0

    def Poisson_BForm(self):

        b = self.pde.mesh.integral(self.pde.source, celltype=True)

        return b

    def DirichletBC(self, A0, b):

        self.node = self.pde.mesh.entity('node')
        self.edge = self.pde.mesh.entity('edge')
        self.cell2edge = self.pde.mesh.cell_to_edge()
        I = bm.where(self.flag)[0] 
        J = bm.where(self.flag)[0]
        val = (self.h/(self.h/2)-self.h/self.h)*bm.ones(I.shape)
        A0 += coo_matrix((val,(I, J)), shape=(self.NC, self.NC))
        index = bm.where(self.flag)[0]
        index1 = self.cell2edge[self.flag]
        point = self.node[self.edge[index1]]
        point = point[:,0,:]*(1/2)+point[:,1,:]*(1/2)
        bu = self.pde.dirichlet(point)
        data = bu*self.h/(self.h/2)
        bm.add.at(b, index, data)

        return  A0, b