from typing import Any, Optional, Union

from fealpy.backend import bm
from fealpy.typing import TensorLike
from fealpy.decorator import variantmethod
from fealpy.model import ComputationalModel
from fealpy.model.stokes import StokesPDEDataT

from fealpy.functionspace import functionspace 
from fealpy.fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
from fealpy.fem import ScalarDiffusionIntegrator as DiffusionIntegrator
from fealpy.fem import DirichletBC

class DLDMicrofluidicChipLFEMModel(ComputationalModel):
    """
    A lagrange fem computational model class for Deterministic Lateral 
    Displacement (DLD) microfluidic chip simulation.

    Parameters:
        options(dict, optional): A dictionary containing computational options 
        for the model.

    Attributes:

    Methods:

    Notes:

    Todos:

    """
    def __init__(self, options: dict = None):
        self.options = options
        super().__init__(
            pbar_log=options['pbar_log'],
            log_level=options['log_level']
        )


    def set_mesh(self, mesh):
        """
        """

    def linear_system(self):
        pass

    def set_space_degree(self, p: int=2):
        """
        """
        self.p = p

    def linear_system(self):
        """
        """

        GD = self.mesh.geo_dimension()
        self.uspace = functionspace(self.mesh, ('Lagrange', self.p), shape=(-1, GD))
        self.pspace = functionspace(self.mesh, ('Lagrange', self.p-1))

        A00 = BilinearForm(self.uspace)
        
        di = DiffusionIntegrator(q=q)
        A00.add_integrator(di)
        
        A01 = BilinearForm((pspace, uspace))
        pwi = PressWorkIntegrator(q=q)
        A01.add_integrator(pwi)
       
        A = BlockForm([[A00, A01], [A01.T, None]]) 

        L0 = LinearForm(uspace)
        L1 = LinearForm(pspace)
        L = LinearBlockForm([L0, L1])

        return A, L

    @variantmethod
    def solve(self):
        pass

