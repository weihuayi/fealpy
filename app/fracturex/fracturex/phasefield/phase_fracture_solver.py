from typing import Optional

from fealpy.experimental.typing import TensorLike
from fealpy.experimental.backend import backend_manager as bm

class PhaseFractureSolver:
    def __init__(self, mesh, space, material_properties, q=None, method='HybridModel'):
        """
        Parameters
        ----------
        mesh : Mesh
            The mesh object.
        space : Space
        material_properties : material properties
        q : integrator precision
        method : str
            The method of stress decomposition method
        """
        self.mesh = mesh
        self.space = space
        self.material_properties = material_properties
        self.q = q if q is not None else q = space.p+2
        self.method = method

        self.get_material()

    def solve(self):
        pass

    def update(self):
        pass

    def solve_displacement(self):
        """

        """
        pass

    def solve_phase_field(self):
        return self.phase_field

    def get_material(self):
        self.lam = self.material_properties['lam']
        self.mu = self.material_properties['mu']


    def get_mesh(self):
        return self.mesh

