import numpy as np

class FiniteElementSpace():
    def __init__(self):
        pass

    def __str__(self):
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
    
    def cell_to_dof(self):
        pass

    def number_of_global_dofs(self):
        pass

    def number_of_local_dofs(self):
        pass

    def interpolation_points(self):
        pass

    def interpolation(self, u, uI):
        pass

    def projection(self, u, up):
        pass

    def array(self):
        pass
