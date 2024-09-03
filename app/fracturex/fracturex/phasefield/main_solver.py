from typing import Optional

from fealpy.experimental.typing import TensorLike
from fealpy.experimental.backend import backend_manager as bm
from energy_degradation_function import EnergyDegradationFunction
from constitutive_model import ConstitutiveModelFactory

class MainSolver:
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

        energy_degradation = EnergyDegradationFunction()
        self.model = ConstitutiveModelFactory.create_model(method,
                                                           material_properties, energy_degradation)
        self.get_material()

    def solve(self):
        """
        Solve the phase field fracture problem.
        """

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

