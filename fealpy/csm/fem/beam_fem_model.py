from ...backend import backend_manager as bm
from ..model import ComputationalModel
from ..model import CSMModelManager
from ...fem import BilinearForm
from ...fem import LinearForm
from ...functionspace import LagrangeFESpace, TensorFunctionSpace
from ...fem import DirichletBC
from ..fem import BeamDiffusionIntegrator
from ..fem import BeamSourceIntegrator
from ...solver import spsolve
from typing import Union
from ..model.beam import BeamPDEDataT

import matplotlib.pyplot as plt

class BeamFEMModel(ComputationalModel):
    """
    BeamFEMModel is a finite element model for solving static problems of beam structures.

    This class implements a finite element method (FEM) solver for the static analysis of beams, 
    suitable for force and deformation analysis. By specifying material parameters, 
    section properties, and loads, it automatically constructs the finite element mesh, 
    assembles the stiffness matrix and load vector, and solves for the displacement field. 
    It relies on PDEModelManager for physical parameters and boundary conditions, 
    making it suitable for teaching, research, and preliminary engineering analysis.
    
    Parameters
        example : str, optional, default='beam2d'
            Selects a preset beam problem example for initializing PDE parameters and mesh.
    Attributes
        pde : object
            Object returned by PDEModelManager containing physical parameters and boundary conditions.
        mesh : object
            Finite element mesh object describing the discretized beam.
        E : float
            Young's modulus, describing material stiffness.
        A : float
            Cross-sectional area, used for axial stiffness calculation.
        I : float
            Moment of inertia, used for bending stiffness calculation.
        f : float or callable
            Distributed load applied to the beam.
        l : float
            Length of the beam.
    Methods
        run()
            Executes the FEM solution process and returns the displacement vector.
        linear_system()
            Assembles and returns the stiffness matrix and load vector for the beam problem.
        solve()
            Applies boundary conditions and solves the linear system, returning the displacement solution.
    Notes
        This class assumes the provided PDEModelManager example defines all necessary parameters and boundary conditions.
        Supports custom loads and boundary conditions for various beam problems.
        Depends on external finite element spaces, integrators, and linear solvers.
    Examples
        >>> model = BeamFEMModel(example='beam2d')
        >>> displacement = model.run()
        >>> print(displacement)
        [0.0, 0.0012, 0.0023, ...]
    """

    def __init__(self, options):
        '''
        Initializes the PoissonFDMModel with the specified example.
        Parameters:
            example (str): The name of the beam problem example to use. Default is 'beam2d'.
            Initializes the PDE parameters, mesh, and material properties based on the example.
        Raises:
            ValueError: If the example is not recognized or cannot be initialized.
        Notes:
            The example should be a valid key in the PDEModelManager for beam problems.
            It must define all necessary parameters and boundary conditions.
        Examples:
            >>> model = PoissonFDMModel(example='beam2d')
            >>> print(model.pde)
            <PDEModelManager object with beam parameters>
        '''
        self.options = options
        super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
        self.set_pde(options['pde'])
        self.beam_type = options['beam_type']
        self.E = options['modulus']
        self.A = options['area']
        self.I = options['inertia']
        self.f = options['load']
        self.l = options['length']
        self.mesh = self.pde.init_mesh()
        
    def set_pde(self, pde:Union[BeamPDEDataT, str]='beam2d'):
        '''
        Set the PDE parameters for the beam problem.
        Parameters:
            pde (PDEModelManager): The PDE data manager containing beam parameters and boundary conditions.
        Raises:
            ValueError: If the provided pde is not valid or does not contain necessary parameters.
        Notes:
            This method updates the model's physical parameters and mesh based on the provided PDE data.
        Examples:
            >>> model.set_pde(new_pde)
        '''
        if isinstance(pde, str):
            self.pde = CSModelManager('beam').get_example(pde)
        else:
            self.pde = pde
        self.mesh = self.pde.init_mesh()


    def run(self):
        '''
        Run the finite element method for the beam problem.
        Returns:
            uh (ndarray): Displacement solution vector.
        '''
        uh = self.solve()
        return uh


    def linear_system(self):
        '''
        Construct the linear system for the beam problem.
        Returns:
            K (csr_matrix): Stiffness matrix.
            F (ndarray): Load vector.
        '''
        mesh = self.mesh
        E = self.E
        A = self.A
        I = self.I
        f = self.f
        l = mesh.cell_length()
        scalar_space = LagrangeFESpace(mesh, 1)
        tensor_space = TensorFunctionSpace(scalar_space=scalar_space, shape=(-1, 2))
        bform = BilinearForm(tensor_space)
        beamintegrator = BeamDiffusionIntegrator(tensor_space, self.beam_type, E, A=A, I=I, l=l)
        bform.add_integrator(beamintegrator)
        K = bform.assembly()
        lform = LinearForm(tensor_space)
        FF  = BeamSourceIntegrator(tensor_space, self.beam_type, source=-f, l=l)
        lform.add_integrator(FF)
        F = lform.assembly()
        return K, F



    def solve(self):
        """
        Solves the linear system of the finite element beam model and returns the displacement solution vector.
        This function first constructs the linear system (stiffness matrix K and load vector F),
        then applies the given boundary conditions, and finally solves the linear system to obtain the beam displacement solution.
        Parameters
            None
        Returns
            uh : numpy.ndarray
            Displacement solution vector containing the numerical solution at all degrees of freedom, as a 1D array.
        Raises
            ValueError
            Raised if the linear system cannot be assembled or solved correctly.
        Notes
            This function assumes that the mesh, finite element space, boundary conditions, and PDE object are properly defined.
            Uses SciPy's sparse linear solver for computation.
        Examples
            >>> model = BeamFEMModel(mesh, pde)
            >>> uh = model.solve()
            >>> print(uh.shape)
            (number_of_dofs,)
        """
        K, F = self.linear_system()
        scalar_space = LagrangeFESpace(self.mesh, 1)
        tensor_space = TensorFunctionSpace(scalar_space=scalar_space, shape=(-1, 2))
        gdof = tensor_space.number_of_global_dofs()
        threshold = bm.zeros(gdof, dtype=bool)
        threshold[self.pde.dirichlet_dof_index()] = True
        uh = tensor_space.function()
        bc = DirichletBC(tensor_space, gd=self.pde.dirichlet, threshold=threshold)
        K,F= bc.apply(K, F)
        uh = spsolve(K, F, 'scipy')
        return uh
    
        
        
