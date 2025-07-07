from ...backend import backend_manager as bm
from ..model import ComputationalModel
from ..model import PDEDataManager
from ...fem import BilinearForm
from ...fem import LinearForm
from ...functionspace import LagrangeFESpace, TensorFunctionSpace
from ...fem import DirichletBC
from ...solver import spsolve
from typing import Union
from ..model.elastoplasticity import ElastoplasticityPDEDataT


def ElastoplasticityFEMModel(ComputationalModel):
    """
    ElastoplasticityFEMModel is a finite element model for solving elastoplasticity problems.

    This class implements a finite element method (FEM) solver for elastoplasticity problems, 
    suitable for analyzing materials that exhibit both elastic and plastic behavior under loading. 
    By specifying material parameters, boundary conditions, and loads, it automatically constructs 
    the finite element mesh, assembles the stiffness matrix and load vector, and solves for the displacement field. 
    It relies on PDEDataManager for physical parameters and boundary conditions, making it suitable for teaching, 
    research, and preliminary engineering analysis.
    
    Parameters
        example : str, optional, default='elastoplasticity2d'
            Selects a preset elastoplasticity problem example for initializing PDE parameters and mesh.
    Attributes
        pde : object
            Object returned by PDEDataManager containing physical parameters and boundary conditions.
        mesh : object
            Finite element mesh object describing the discretized domain.
        E : float
            Young's modulus, describing material stiffness.
        nu : float
            Poisson's ratio, describing material compressibility.
        yield_strength : float
            Yield strength of the material.
        f : float or callable
            Distributed load applied to the domain.
        l : float
            Characteristic length of the domain.
    Methods
        run()
            Executes the FEM solution process and returns the displacement vector.
        linear_system()
            Assembles and returns the stiffness matrix and load vector for the elastoplasticity problem.
        solve()
            Applies boundary conditions and solves the linear system, returning the displacement solution.
    Notes
        This class assumes the provided PDEDataManager example defines all necessary parameters and boundary conditions.
        Supports custom loads and boundary conditions for various elastoplasticity problems.
        Depends on external finite element spaces, integrators, and linear solvers.
    Examples
        >>> model = ElastoplasticityFEMModel(example='elastoplasticity2d')
        >>> displacement = model.run()
        >>> print(displacement)
        [0.0, 0.0012, 0.0023, ...]
    """
    def __init__(self, options):
        '''
        Initializes the ElastoplasticityFEMModel with the specified example.
        Parameters:
            example (str): The name of the elastoplasticity problem example to use. Default is 'elastoplasticity2d'.
            Initializes the PDE parameters, mesh, and material properties based on the example.
        Raises:
            ValueError: If the example is not recognized or cannot be initialized.
        Notes:
            The example should be a valid key in the PDEDataManager for elastoplasticity problems.
            It must define all necessary parameters and boundary conditions.
        Examples:
            >>> model = ElastoplasticityFEMModel(example='elastoplasticity2d')
            >>> print(model.pde)
            <PDEDataManager object with elastoplasticity parameters>
        '''
        self.options = options
        super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
        self.mesh = self.pde.init_mesh()

    def set_pde(self, pde:Union[ElastoplasticityPDEDataT, str]='elastoplasticity2d'):
        '''
        Set the PDE parameters for the elastoplasticity problem.
        Parameters:
            pde (PDEDataManager): The PDE data manager containing elastoplasticity parameters and boundary conditions.
        Raises:
            ValueError: If the provided pde is not valid or does not contain necessary parameters.
        Notes:
            This method updates the model's physical parameters and mesh based on the provided PDE data.
        Examples:
            >>> model.set_pde(new_pde)
        '''
        if isinstance(pde, str):
            self.pde = PDEDataManager('elastoplasticity').get_example(pde)
        else:
            self.pde = pde
        self.mesh = self.pde.init_mesh()

    
