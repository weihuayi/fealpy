import os 
import numpy as np

from scipy.sparse.linalg import lgmres


class IterationCounter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i' % (self.niter))


class LSSolver():
    def __init__(self, space, phi0, u):
        self.space = space
        self.phi0 = phi0
        self.u = u

    def output(self, timestep, output_dir, filename_prefix):
        mesh = self.space.mesh
        if output_dir != 'None':
            mesh.nodedata['phi'] = self.phi0
            mesh.nodedata['velocity'] = self.u
            fname = os.path.join(output_dir, f'{filename_prefix}_{timestep:010}.vtu')
            mesh.to_vtk(fname=fname)

    def check_gradient_norm(self, phi):
        """
        Check the gradient magnitude of the level set function.

        Parameters:
        - phi: The level set function.

        Returns:
        - diff_avg: The average difference between the gradient magnitude and 1.
        - diff_max: The maximum difference between the gradient magnitude and 1.
        """
        # Compute the gradient of phi at quadrature points
        mesh = self.space.mesh
        qf = mesh.integrator(3)
        bcs, _ = qf.get_quadrature_points_and_weights()
        grad_phi = self.space.grad_value(uh=phi, bc=bcs)

        # Compute the magnitude of the gradient
        magnitude = np.linalg.norm(grad_phi, axis=-1)

        # Compute the difference between the magnitude and 1
        diff = np.abs(magnitude) - 1

        diff_avg = np.mean(diff)
        diff_max = np.max(diff)

        return diff_avg, diff_max

    def solve_system(self, A, b, tol=1e-8):
        counter = IterationCounter(disp = False)
        result, info = lgmres(A, b, tol = tol, atol = tol, callback = counter)
        # print("Convergence info:", info)
        # print("Number of iteration of gmres:", counter.niter)
        return result
