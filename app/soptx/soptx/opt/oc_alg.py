from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.backend import TensorLike as _DT

from fealpy.experimental.opt.optimizer_base import Optimizer

class OCAlg(Optimizer):
    def __init__(self, options) -> None:
        super().__init__(options)

    def update(self, rho: _DT, dce: _DT, dge: _DT, volume_constraint, filter_properties, mesh) -> _DT:
        """
        Update the design variables using the OC method.

        Parameters:
            rho (_DT): Current design variables (density distribution).
            dce (_DT): Gradient of the objective function (compliance).
            v (float): Current volume constraint value.
            dve (_DT): Gradient of the volume constraint.

        Returns:
            _DT: Updated design variables.
        """
        l1 = 0.0
        l2 = 1e9
        move = 0.2
        
        while (l2 - l1) / (l2 + l1) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            rho_new = bm.maximum(
                0.0, bm.maximum(rho - move, 
                bm.minimum(1.0, bm.minimum(rho + move, rho * bm.sqrt(-dce / dge / lmid))))
            )

            ft, H = filter_properties.ft, filter_properties.H
            cell_measure = mesh.entity_measure('cell')
            if ft == 0:
                rho_phys = H.matmul(rho_new[:] * cell_measure) / H.matmul(cell_measure)
            elif ft == 1:
                rho_phys = rho_new

            g = volume_constraint.fun(rho_phys)

            if g > 0:
                l1 = lmid
            else:
                l2 = lmid

        return rho_new, rho_phys

    def run(self):
        """
        Run the OC optimization algorithm.

        This method executes the OC algorithm to minimize the objective function 
        under the given constraints.
        """
        options = self.options
        objective = options['objective']
        rho = options['x0']
        rho_phys = bm.copy(rho)
        max_iters = options['MaxIters']
        tol_change = options['FunValDiff']

        mesh = objective.mesh
        volume_constraint = objective.volume_constraint
        filter_properties = objective.filter_properties

        for loop in range(max_iters):
            # Evaluate objective function and its gradient
            c = objective.fun(rho_phys)
            dce = objective.jac(rho)

            g = volume_constraint.fun(rho_phys)
            dge = volume_constraint.jac(rho)

            # Update design variables using OC method
            rho_new, rho_phys[:] = self.update(rho, dce, dge, volume_constraint, filter_properties, mesh)

            # Compute change in design variables
            change = bm.max(bm.abs(rho_new - rho))

            # Print the results for this iteration
            print(f"Iteration: {loop + 1}, Objective: {c:.3f}, \
                Volume: {bm.mean(rho_phys):.3f}, Change: {change:.3f}")

            # Check for convergence
            if change <= tol_change:
                print(f"Converged at iteration {loop + 1} with change {change}")
                break

            # Update rho for the next iteration
            rho = rho_new

        return rho
