from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.backend import TensorLike as _DT

from fealpy.experimental.opt.optimizer_base import Optimizer

from app.soptx.soptx.utilfs.timer import timer
from time import time 

class OCAlg(Optimizer):
    def __init__(self, options) -> None:
        super().__init__(options)

    def update(self, rho: _DT, dce: _DT, dge: _DT, volume_constraint, filter_properties, mesh, beta) -> _DT:
        """
        Update the design variables using the OC method.
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
            elif ft == 2:
                rho_tilde = H.matmul(rho_new[:] * cell_measure) / H.matmul(cell_measure)
                rho_phys = 1 - bm.exp(-beta * rho_tilde) + rho_tilde * bm.exp(-beta)

            g = volume_constraint.fun(rho_phys)

            if g > 0:
                l1 = lmid
            else:
                l2 = lmid

        return rho_new, rho_phys

    def run(self):
        """
        Run the OC optimization algorithm.
        """
        options = self.options
        objective = options['objective']
        rho = options['x0']

        filter_properties = objective.filter_properties
        ft = filter_properties.ft

        loopbeta = 0
        beta = 1
        
        if ft == 0 or ft == 1:
            rho_phys = bm.copy(rho)
        elif ft == 2:    
            rho_tilde = bm.copy(rho)
            rho_phys = 1 - bm.exp(-beta * rho_tilde) + rho_tilde * bm.exp(-beta)

        # rho_phys = bm.copy(rho)
        max_iters = options['MaxIters']
        tol_change = options['FunValDiff']

        mesh = objective.mesh
        volume_constraint = objective.volume_constraint

        # tmr = timer("OC Algorithm")
        # next(tmr)
        tmr = None

        for loop in range(max_iters):
            start_time = time()

            c = objective.fun(rho_phys)
            if tmr:
                tmr.send('compliance')
            dce = objective.jac(rho)
            if tmr:
                tmr.send('compliance gradient')

            g = volume_constraint.fun(rho_phys)
            if tmr:
                tmr.send('volume constraint')
            dge = volume_constraint.jac(rho)
            if tmr:
                tmr.send('volume constraint gradient')

            rho_new, rho_phys[:] = self.update(rho, dce, dge, volume_constraint, filter_properties, mesh, beta)
            if tmr:
                tmr.send('OC update')

            if tmr:
                tmr.send(None)
            
            end_time = time()
            iteration_time = end_time - start_time
            
            change = bm.max(bm.abs(rho_new - rho))

            rho = rho_new

            if ft == 2 and beta < 512 and (loopbeta >= 50 or change <= 0.01):
                beta *= 2
                loopbeta = 0
                change = 1
                print(f"Parameter beta increased to {beta}")

            loopbeta += 1

            print(f"Iteration: {loop + 1}, Objective: {c:.3f}, "
                  f"Volume: {bm.mean(rho_phys):.3f}, Change: {change:.3f}, "
                  f"Iteration Time: {iteration_time:.3f} sec")

            if change <= tol_change:
                print(f"Converged at iteration {loop + 1} with change {change}")
                break

        return rho
