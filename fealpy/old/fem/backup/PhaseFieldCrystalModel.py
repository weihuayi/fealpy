import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

import vtk
import vtk.util.numpy_support as vnp

from fealpy.decorator import barycentric
from fealpy.functionspace import IsoLagrangeFiniteElementSpace

class PhaseFieldCrystalModel():
    def __init__(self, mesh, timeline, options):

        self.options = options
        self.timeline = timeline
        self.mesh = mesh 
        self.space = IsoLagrangeFiniteElementSpace(mesh, options['order'])

        self.A = self.space.stiff_matrix()
        self.M = self.space.mass_matrix()

        self.uh0 = self.space.function()
        self.uh1 = self.space.function()

        self.ftype = mesh.ftype
        self.itype = mesh.itype

        self.H = []
        self.G = []

    def options(
            self,
            c=1,
            s=0.3,
            epsilon=-1,
            order=1
            ):

        options = {
                'c': c,
                's': s,
                'epsilon': epsilon,
                'order': order
            }
        return options

    def set_init_solution(self, u):
        self.uh0[:] = u

    def get_current_left_matrix(self):
        dt = self.timeline.current_time_step_length()
        A = self.A
        M = self.M
        S = bmat([[M + dt*(M - 2*A), -dt*A], [A, M]], format='csr')
        return S

    def get_current_right_vector(self):
        dt = self.timeline.current_time_step_length()
        gdof = self.space.number_of_global_dofs()

        s = self.options['s']
        epsilon = self.options['epsilon']

        uh0 = self.uh0 
        M = self.M
        F = np.zeros((2*gdof, ), dtype=self.ftype)
        F[:gdof] = M@uh0
        F[:gdof] *= 1 - dt*epsilon

        @barycentric
        def f(bcs):
            val = uh0(bcs)
            return s*val**2/2 - val**3/6
        F[:gdof] += dt*self.space.source_vector(f)
        return F

    def one_step_solve(self):
        """

        Notes
        -----
            求解一个时间层的数值解
        """

        gdof = self.space.number_of_global_dofs()
        A = self.get_current_left_matrix()
        F = self.get_current_right_vector()
        x = spsolve(A, F)
        self.uh0[:] = x[:gdof]
        self.uh1[:] = x[gdof:]

    def post_process(self):
        area = np.sum(self.space.cellmeasure)
        self.uh0 -= self.space.integralalg.mesh_integral(self.uh0)/area

    def Hamilton(self):
        s = self.options['s']
        epsilon = self.options['epsilon']
        uh0 = self.uh0
        uh1 = self.uh1

        @barycentric
        def f0(bcs):
            val0 = uh0(bcs)
            val1 = uh1(bcs)
            val = (val0 + val1)**2/2
            val += epsilon*val0**2/2
            val -= s*val0**3/6
            val += val0**4/24
            return val

        H = self.space.integralalg.mesh_integral(f0)
        self.H.append(H)

        @barycentric
        def f1(bcs):
            val = uh0(bcs)
            return s*val**2/2 - val**3/6
        grad = -self.M*uh0 
        grad -= epsilon*self.M*uh0 
        grad += 2*self.A*uh0 
        grad += self.space.source_vector(f1) 
        grad += self.A*uh1

        self.G.append(np.linalg.norm(grad))


    def solve(self, disp=True, output=False, rdir='.', step=1, postprocess=False):
        """

        Notes
        -----

        计算所有的时间层。
        """

        timeline = self.timeline
        dt = timeline.current_time_step_length()
        timeline.reset() # 时间置零

        if postprocess:
            self.post_process()

        self.Hamilton()

        if output:
            fname = rdir + '/step_'+ str(timeline.current).zfill(10) + '.vtu'
            print(fname)
            self.write_to_vtk(fname)

        if disp:
            print(timeline.current, "Current Hamilton energy ", self.H[-1], " with gradient ",
                    self.G[-1] )
            print("Max phase value:", np.max(self.uh0))
            print("Min phase value:", np.min(self.uh0))

        while not timeline.stop():
            self.one_step_solve()
            if postprocess:
                self.post_process()
            self.Hamilton()
            timeline.current += 1

            if disp:
                print("Current Hamilton energy ", self.H[-1], " with gradient ",
                        self.G[-1])
                print("Max phase value:", np.max(self.uh0))
                print("Min phase value:", np.min(self.uh0))

            if output & (timeline.current%step == 0):
                fname = rdir + '/step_'+ str(timeline.current).zfill(10) + '.vtu'
                print(fname)
                self.write_to_vtk(fname)
        timeline.reset()

    def write_to_vtk(self, fname):
        self.mesh.nodedata['uh0'] = self.uh0
        self.mesh.nodedata['uh1'] = self.uh1
        self.mesh.to_vtk(fname=fname)

