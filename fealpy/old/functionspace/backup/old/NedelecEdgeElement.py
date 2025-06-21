import numpy as np

class NedelecEdgeElement2d():
    def __init__(self, mesh, p=1, dtype=np.float):
        self.mesh=mesh
        self.p = p
    
    def basis(self, bc):
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        dim = mesh.geom_dimentsion()
        Dlambda, _ = mesh.grad_lambda() 
        if self.p == 1:
            phi = np.zeros((NC,3,2), dtype=self.dtype)
            phi


    def grad_basis(self, bc):
        pass

    def dual_basis(self, u):
        pass

    def array(self):
        pass

    def value(self, uh, bc):
        pass

    def grad_value(self, uh, bc):
        pass

    def number_of_global_dofs(self):
        pass

    def number_of_local_dofs(self):
        pass

    def cell_to_dof(self):
        pass


