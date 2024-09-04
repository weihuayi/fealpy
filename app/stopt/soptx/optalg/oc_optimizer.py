from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.typing import TensorLike, Tuple

from builtins import float
from app.stopt.soptx.utilfunc.filter_parameters import apply_filter
from app.stopt.soptx.utilfunc.calculate_KE import calculate_KE

class OCOptimizer:
    def __init__(self, displacement_solver, 
                 objective_function, 
                 sensitivity_function, 
                 termination_criteria, 
                 constraint_conditions,
                 filter_parameters=None):
        """
        Initialize the OC Optimizer.

        Args:
            displacement_solver: The solver used to solve the displacement field (e.g., FEMSolver).
            objective_function: An instance of a topology optimization problem (e.g., ComplianceMinimization).
            sensitivity_function: Function to compute the sensitivities (e.g., manual_objective_sensitivity).
            termination_criteria: An instance of TerminationCriteria to check if optimization should terminate.
            constraint_conditions: An instance of ConstraintConditions to manage constraints (e.g., volume fraction).
            filter_parameters: An instance of FilterProperties holding filter type, matrix, and scaling vector (default is None).
        """
        self.displacement_solver = displacement_solver
        self.objective_function = objective_function
        self.sensitivity_function = sensitivity_function
        self.termination_criteria = termination_criteria
        self.constraint_conditions = constraint_conditions
        self.filter_parameters = filter_parameters

    def update_design_variables(self, rho: TensorLike, 
                                dce: TensorLike, dve: TensorLike, 
                                rho_phys: TensorLike, H: TensorLike, Hs: TensorLike, 
                                ft: int, volfrac: float, NC: int) -> Tuple[TensorLike, TensorLike]:
        """
        Update the design variables using the Optimality Criteria (OC) method.

        Args:
            rho (TensorLike): Current design variables.
            dce (TensorLike): Compliance sensitivity.
            dve (TensorLike): Volume sensitivity.
            rho_phys (TensorLike): Physical density.
            H (TensorLike): Filter matrix.
            Hs (TensorLike): Scaling vector for filter matrix.
            ft (int): Filter type.
            volfrac (float): Volume fraction constraint.
            NC (int): Number of elements.

        Returns:
            Tuple[TensorLike, TensorLike]: Updated physical density and design variables.
        """
        l1 = 0.0
        l2 = 1e9
        move = 0.2

        while (l2 - l1) / (l2 + l1) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            rho_new = bm.maximum(
                0.0, bm.maximum(rho - move, 
                bm.minimum(1.0, bm.minimum(rho + move, rho * bm.sqrt(-dce / dve / lmid))))
            )
            if ft == 0:
                rho_phys = rho_new
            elif ft == 1:
                rho_phys = bm.asarray(H @ rho_new[bm.newaxis].T / Hs)[:, 0]
            if bm.sum(rho_phys) - volfrac * NC > 0:
                l1 = lmid
            else:
                l2 = lmid

        return rho_phys, rho_new

    def optimize(self, rho: TensorLike) -> TensorLike:
        """
        Perform the optimization using the OC method.

        Args:
            rho: TensorLike, initial density distribution.

        Returns:
            TensorLike: Optimized density distribution.
        """
        change = 1.0
        loop = 0

        volfrac = self.constraint_conditions.get_constraints()['volume']['vf']

        while not self.termination_criteria.should_terminate(loop, change):
            loop += 1
            # Step 1: Solve the displacement field using the specified solver
            uh = self.displacement_solver.solve()

            # Step 2: Compute objective function and sensitivities
            KE = calculate_KE(material_properties=self.displacement_solver.material_properties, 
                            tensor_space=self.displacement_solver.tensor_space)
            uhe = self.displacement_solver.get_element_displacement()
            E = self.displacement_solver.material_properties.material_model()
            
            # Calculate compliance and element-wise compliance
            c = self.objective_function.compute_objective(uhe, KE, E)
            ce = self.objective_function.compute_element_compliance(uhe, KE)

            # Compute sensitivities
            dE = self.displacement_solver.material_properties.material_model_derivative()
            dce = self.sensitivity_function(ce, dE)
            dve = bm.ones_like(dce)  # Volume sensitivity, assuming unit volume per element

            # Step 3: Apply filter to sensitivities if filter parameters are provided
            if self.filter_parameters:
                dce, dve = apply_filter(
                    ft=self.filter_parameters.ft, 
                    rho=rho, 
                    dc=dce, 
                    dv=dve, 
                    H=self.filter_parameters.H, 
                    Hs=self.filter_parameters.Hs
                )

            # Step 4: Update design variables using the OC method
            NC = ce.shape[0]
            rho_phys, rho_new = self.update_design_variables(rho, dce, dve, 
                                                            rho_phys=rho, 
                                                            H=self.filter_parameters.H, 
                                                            Hs=self.filter_parameters.Hs, 
                                                            ft=self.filter_parameters.ft,
                                                            volfrac=volfrac,
                                                            NC=NC)

            # Step 5: Calculate the change in design variables
            change = bm.linalg.norm(rho_new.reshape(NC, 1) - rho.reshape(NC, 1), bm.inf)

            rho = rho_new

            # Optional: Output the current iteration results
            print(f"Iteration: {loop}, Objective: {c:.3f}, Volume: {bm.mean(rho_phys):.3f}, Change: {change:.3f}")

        return rho_phys
