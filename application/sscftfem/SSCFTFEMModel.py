import numpy as np

from fealpy.quadrature  import TriangleQuadrature
from fealpy.functionspace.surface_lagrange_fem_space import SurfaceLagrangeFiniteElementSpace
from fealpy.femmodel import doperator 
from scipy.sparse.linalg import spsolve

import plotly.offline as py
import plotly.figure_factory as FF
import fealpy.tools.colors as cs
import scipy.io as sio




class SSCFTParameter():
    def __init__(self):
        self.Nspecies = 2 
        self.Nblend   = 1
        self.Nblock   = 2 
        self.Ndeg     = 100  
        self.fA       = 0.2 
        self.chiAB    = 0.25 
        self.dim = 2
        self.dtMax = 0.005
        self.tol = 1.0e-6
        self.tolR = 1.0e-3
        self.maxit = 5000
        self.showstep = 200
        self.pdemethod = 'CN'
        self.integrator = TriangleQuadrature(3)
        self.fieldType = 'fieldmu'


class TimeLine():
    def __init__(self, interval, dt):
        self.T0 = interval[0]
        self.T1 = interval[1]
        self.Nt = int(np.ceil((self.T1 - self.T0)/dt))
        self.dt = (self.T1 - self.T0)/self.Nt
        self.current = 0

    def get_index_interval(self):
        return np.arange(0, self.Nt+1) 

    def get_number_of_time_steps(self):
        return self.Nt+1

    def get_current_time_step(self):
        return self.current

    def get_time_step_length(self):
        return self.dt

    def stop(self):
        return self.current >= self.Nt - 1 

    def step(self):
        self.current += 1

    def reset(self):
        self.current = 0
        
class PDESolver():
    def __init__(self, femspace, integrator, measure,  method='CN'):

        self.method = method
        self.femspace = femspace 
        self.integrator = integrator 
        self.measure = measure

        self.M = doperator.mass_matrix(self.femspace, self.integrator, self.measure)
        self.A = doperator.stiff_matrix(self.femspace, self.integrator, self.measure)

    def get_current_linear_system(self, u0, dt):
        M = self.M
        S = self.A
        F = self.F
        if self.method is 'FM':
            b = -dt*(S + F)@u0 + M@u0
            A = M   
            return A, b
        if self.method is 'BM':
            b = M@u0
            A = M + dt*(S + F)
            return A, b
        if self.method is 'CN':
            b = -0.5*dt*(S + F)@u0 + M@u0
            A = M + 0.5*dt*(S + F)
            return A, b

    def run(self, timeline, uh, F):
        self.F = F

        while timeline.current < timeline.Nt: 
            current = timeline.current
            dt = timeline.get_time_step_length()
            A, b = self.get_current_linear_system(uh[:, current], dt)
            uh[:, current+1] = spsolve(A, b)
            timeline.current += 1
        timeline.reset()
    

class SSCFTFEMModel():
    def __init__(self, surface, mesh, option, p=1, p0=1):
        self.femspace = SurfaceLagrangeFiniteElementSpace(mesh, surface, p=p, p0=p0) 
        self.mesh = self.femspace.mesh
        self.area = mesh.area()
        self.totalArea = np.sum(self.area)
        print(self.totalArea)
        self.surface = surface
        self.option = option

        self.timeline0 = TimeLine([0, option.fA], option.dtMax)
        self.timeline1 = TimeLine([option.fA, 1], option.dtMax)

        N = self.timeline0.Nt + self.timeline1.Nt + 1
        self.q0 = self.femspace.function(dim=N)
        self.q1 = self.femspace.function(dim=N) 

        self.Ndof = self.femspace.number_of_global_dofs()

        self.rho   = [self.femspace.function() for i in range(option.Nspecies)] 
        self.w   = [self.femspace.function() for i in range(option.Nspecies)] 
        self.mu   = [self.femspace.function() for i in range(option.Nspecies)] 
        self.grad = self.femspace.function(dim=option.Nspecies)
        self.sQ    = np.zeros((option.Nspecies-1, option.Nblend))

        self.solver = PDESolver(self.femspace, option.integrator, self.area, option.pdemethod)


    def initialize(self):
        option = self.option
        fields = option.fields 
        chiN = option.chiAB * option.Ndeg

        self.w[0][:] = fields[:, 0] - fields[:, 1]
        self.w[1][:] = fields[:, 0] + fields[:, 1]
         
#        if option.fieldType is 'fieldmu':
#            self.mu[0][:]   = fields[:, 0]
#            self.mu[1][:]   = fields[:, 1]
#            self.w[0][:] = fields[:, 0] - fields[:, 1]
#            self.w[1][:] = fields[:, 0] + fields[:, 1]
#        if option.fieldType is 'fieldw':
#            self.w[0][:]   = fields[:, 0]
#            self.w[1][:]   = fields[:, 1]
#            self.mu[0][:] = 0.5*(fields[:, 0] + fields[:, 1])
#            self.mu[1][:] = 0.5*(fields[:, 1] - fields[:, 0])

        self.mu[0][:]   = 0.5*(self.w[0] + self.w[1]) 
        self.mu[1][:]   = 0.5*(self.w[1] - self.w[0]) 
        self.rho[0][:] = 0.5 + self.mu[1]/chiN
        self.rho[1][:] = 1.0 - self.rho[0]

        self.data ={
                'node':self.mesh.mesh.point, 
                'elem':self.mesh.mesh.ds.cell+1, 
                'rhoA':[self.rho[0]],
                'rhoB':[self.rho[1]],
                'sQ':[],
                'ediff':[],
                'H':[]
                }

    def update_propagator(self):
        option = self.option
        n0 = self.timeline0.get_number_of_time_steps()
        n1 = self.timeline1.get_number_of_time_steps()

        F0 = doperator.mass_matrix(
                self.femspace, 
                option.integrator, 
                self.area, 
                cfun=self.w[0].value)

        F1 = doperator.mass_matrix(
                self.femspace, 
                option.integrator, 
                self.area, 
                cfun=self.w[1].value)

        self.q0[:, 0] = 1.0
        self.solver.run(self.timeline0, self.q0[:, 0:n0], F0)
        self.solver.run(self.timeline1, self.q0[:, n0-1:], F1)
        self.q1[:, 0] = 1.0
        self.solver.run(self.timeline1, self.q1[:, 0:n1], F1)
        self.solver.run(self.timeline0, self.q1[:, n1-1:], F0)

    def integral_time(self, q, dt):
        f = -0.625*(q[:, 0] + q[:, -1]) + 1/6*(q[:, 1] + q[:, -2]) - 1/24*(q[:, 2] + q[:, -3])
        f += np.sum(q, axis=1)
        f *= dt
        return f
    
    def integral_space(self, u):
        mesh = self.femspace.mesh
        qf = self.option.integrator 
        bcs, ws = qf.quadpts, qf.weights
        val = u(bcs)
        Q = np.einsum('i, ij, j', ws, val, self.area)
        return Q


    def update_density(self, q):
        n0 = self.timeline0.get_number_of_time_steps()
        self.rho[0][:] = self.integral_time(q[:, 0:n0],
                self.timeline0.dt)/self.sQ[0, 0]
        self.rho[1][:] = self.integral_time(q[:, n0-1:],
                self.timeline1.dt)/self.sQ[0, 0]

    def update_singleQ(self, integrand):
        u = integrand.value
        f = self.integral_space(integrand.value)/self.totalArea 
        return f

    def update_hamilton(self):
        u = self.mu[0].value
        mu1_int = self.integral_space(u)

        u = lambda x : self.mu[1].value(x)**2
        mu2_int = self.integral_space(u)

        chiN = self.option.chiAB * self.option.Ndeg
        H = -mu1_int + mu2_int/chiN
        return H

    def update_field(self):

        option = self.option
        chiN = option.chiAB * option.Ndeg
        
        lambd = np.array([2,2])

        self.grad[:, 0] = self.rho[0]  + self.rho[1] - 1.0
        self.grad[:, 1] = 2.0*self.mu[1]/chiN - self.rho[0] + self.rho[1]

        err = np.max(np.abs(self.grad), axis=0)

        self.mu[0] += lambd[0]*self.grad[:, 0]
        self.mu[1] -= lambd[1]*self.grad[:, 1]

        self.w[0][:] = self.mu[0] - self.mu[1] 
        self.w[1][:] = self.mu[0] + self.mu[1]
        
        return err 

    def find_saddle_point(self, datafile='data', showsolution=True, file_path=None):
        
        self.res = np.inf
        self.Hold = np.inf
        self.ediff = np.inf
        iteration = 0
        option = self.option
        
        if file_path is not None:
            file = open(file_path + '/log.txt', 'w')
        while (self.res > option.tol) and (iteration < option.maxit):
            self.H, self.res = self.one_step()
            self.ediff = self.H - self.Hold
            self.Hold = self.H
            iteration += 1


            if file_path is not None:
                string = 'Iter: %d ======> \n sQ: %f res: %f ediff: %f H: %f \n' % (iteration, self.sQ, self.res, self.ediff, self.H)
                file.write(string)

            print('Iter:',iteration,'======>','res:', self.res, 'ediff:',self.ediff, 'H:', self.H)
            print('\n')

            self.data['rhoA'].append(self.rho[0])
            self.data['rhoB'].append(self.rho[1])
            self.data['H'].append(self.H)
            self.data['ediff'].append(self.ediff)

            if (iteration%option.showstep == 0) and showsolution:
                self.show_solution(iteration)
        sio.matlab.savemat(datafile+'.mat', self.data)
        
        file.close()
        
    def one_step(self):
        self.update_propagator() 
        qq = self.q0*self.q1[:, -1::-1] 
        self.sQ[0,0] = self.update_singleQ(self.q0.index(-1))

        self.data['sQ'].append(self.sQ[0, 0])

        print('sQ:', self.sQ)
        self.update_density(qq)

        error = self.update_field()
    
        H = self.update_hamilton()

        H = H/self.totalArea - np.log(self.sQ[0,0])
        return H, error.max()


    def get_partial_F(self, radius):
 
        self.update_propagator()

        qq = self.q0 * self.q1[:, -1::-1] 
        self.sQ[0,0] = self.updateQ(self, qq)
        gradF = -np.sum(sQ.reshape(-1))

        return partialF

    def show_solution(self, i):
        mesh = self.mesh.mesh
        cell = mesh.ds.cell
        point = mesh.point
        c = self.rho[0].view(np.ndarray)
        c = np.sum(c[cell], axis=1)/3
        c = cs.val_to_color(c)
        fig = FF.create_trisurf(
                x = point[:, 0], 
                y = point[:, 1],
                z = point[:, 2],
                show_colorbar = True,
                plot_edges=False,
                simplices=cell)
        fig['data'][0]['facecolor'] = c
        py.plot(fig, filename='test{}'.format(i))
