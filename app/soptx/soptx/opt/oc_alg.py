from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.backend import TensorLike as _DT

from fealpy.experimental.opt.optimizer_base import Optimizer

class OCAlg(Optimizer):
    def __init__(self, options) -> None:
        super().__init__(options)

    def update(self, rho: _DT, dce: _DT, volume_constraint) -> _DT:
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

        dve = volume_constraint.jac()
        
        while (l2 - l1) / (l2 + l1) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            # Update design variable using OC formula
            rho_new = bm.maximum(
                0.0, bm.maximum(rho - move, 
                bm.minimum(1.0, bm.minimum(rho + move, rho * bm.sqrt(-dce / dve / lmid))))
            )

            v_new = volume_constraint.fun(rho_new)

            # Check volume constraint to update l1 and l2
            if v_new > 0:
                l1 = lmid
            else:
                l2 = lmid

        return rho_new

    def run(self):
        """
        Run the OC optimization algorithm.

        This method executes the OC algorithm to minimize the objective function 
        under the given constraints.
        """
        options = self.options
        rho = options['x0']
        max_iters = options['MaxIters']
        tol_change = options.get('tol_change', 0.01)
        volume_constraint = options['volume_constraint']

        for loop in range(max_iters):
            # Evaluate objective function and its gradient
            c = self.fun(rho)
            dce = self.options['objective'].jac(rho)

            v = volume_constraint.fun(rho)

            # Update design variables using OC method
            rho_new = self.update(rho, dce, volume_constraint)

            # Compute change in design variables
            change = bm.linalg.norm(rho_new.reshape(-1, 1) - rho.reshape(-1, 1), bm.inf)

            # Print the results for this iteration
            print(f"Iteration: {loop + 1}, Objective: {c:.3f}, 
                Volume: {v+volume_constraint.volfrac:.3f}, Change: {change:.3f}")

            # Check for convergence
            if change <= tol_change:
                print(f"Converged at iteration {loop + 1} with change {change}")
                break

            # Update rho for the next iteration
            rho = rho_new

        return rho
