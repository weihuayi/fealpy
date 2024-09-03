from typing import Optional

from fealpy.experimental.typing import TensorLike
from fealpy.experimental.backend import backend_manager as bm
from energy_degradation_function import EnergyDegradationFunction as EDFunc
from phase_fracture_constitutive_model import PhaseFractureConstitutiveModelFactory as PFCMF

class MainSolver:
    def __init__(self, material_properties, method='HybridModel'):
        """
        Parameters
        ----------
        material_properties : material properties
        fracture_params : fracture parameters
        method : str
            The method of stress decomposition method
        """
#        self.q = q if q is not None else q = space.p+2

        self.decomp_model = PFCMF.create_model(method, material_properties, EDFunc())

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
        pass


