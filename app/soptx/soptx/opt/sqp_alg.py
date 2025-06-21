from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike as _DT
from fealpy.opt.optimizer_base import Optimizer

class SQPAlg(Optimizer):
    def __init__(self, options) -> None:
        super().__init__(options)

    def solve_qp(self, hessian: _DT, grad: _DT, A: _DT) -> tuple:
        """
        Solve the quadratic programming problem using KKT conditions.

        Parameters:
            hessian (_DT): Hessian matrix of the Lagrangian.
            grad (_DT): Gradient of the objective function.
            A (_DT): Jacobian matrix of the constraints.

        Returns:
            tuple: The search direction d and Lagrange multipliers lambda.
        """
        n = hessian.shape[0]
        m = A.shape[0]

        # Construct the KKT matrix
        KKT_matrix = bm.zeros((n + m, n + m))
        KKT_matrix[:n, :n] = hessian
        KKT_matrix[:n, n:] = A.T
        KKT_matrix[n:, :n] = A

        # Construct the right-hand side
        rhs = bm.concatenate([-grad, bm.zeros(m)])

        # Solve the KKT system
        solution = bm.linalg.solve(KKT_matrix, rhs)
        d = solution[:n]
        lambda_ = solution[n:]

        return d, lambda_

    def line_search(self, rho: _DT, d: _DT, grad: _DT) -> float:
        """
        Perform a line search to find a suitable step size.

        Parameters:
            rho (_DT): Current design variables.
            d (_DT): Search direction.
            grad (_DT): Gradient of the objective function.

        Returns:
            float: Step size alpha.
        """
        alpha = 1.0
        c = 1e-4
        rho_new = rho + alpha * d
        obj_new = self.options['objective'].fun(rho_new)
        obj_current = self.options['objective'].fun(rho)

        # Simple backtracking line search
        while obj_new > obj_current + c * alpha * bm.dot(grad, d):
            alpha *= 0.5
            rho_new = rho + alpha * d
            obj_new = self.options['objective'].fun(rho_new)
        
        return alpha

    def run(self):
        """
        Run the SQP optimization algorithm.

        This method executes the SQP algorithm to minimize the objective function 
        under the given constraints.
        """
        options = self.options
        rho = options['x0']
        max_iters = options['MaxIters']
        tol_change = options.get('tol_change', 0.01)  # Default tolerance if not provided
        volume_constraint = options['volume_constraint']

        for loop in range(max_iters):
            # Evaluate objective function, gradient, and Hessian
            c = self.fun(rho)
            dce = self.options['objective'].jac(rho)
            lambda_ = {'ineq': volume_constraint.fun(rho)}
            hessian = self.options['objective'].hess(rho, lambda_)

            # Construct constraint Jacobian A
            A = bm.atleast_2d(volume_constraint.jac()).T  # Ensure A is 2D

            # Solve the QP problem
            d, lambda_ = self.solve_qp(hessian, dce, A)

            # Perform line search to find step size
            alpha = self.line_search(rho, d, dce)

            # Update design variables
            rho_new = rho + alpha * d

            # Compute change in design variables
            change = bm.linalg.norm(rho_new.reshape(-1, 1) - rho.reshape(-1, 1), bm.inf)

            # Print the results for this iteration
            print(f"Iteration: {loop + 1}, Objective: {c:.3f}, 
                Volume Fraction: {lambda_ + volume_constraint.volfrac:.3f}, 
                Change: {change:.3f}, Step Size: {alpha:.3f}")

            # Check for convergence
            if change <= tol_change:
                print(f"Converged at iteration {loop + 1} with change {change}")
                break

            # Update rho for the next iteration
            rho = rho_new

        return rho
