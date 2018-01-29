import numpy as np
import sys
# import pdb 


class Cmp_mesh_data: 
    def __init__(self):
        self.surface = 'sphere'
        self.elem = 100
        self.node = 200
        self.radius = 2.0
#         %%%% computing simplex area
        self.area = np.random.rand(1,100)
        self.tol_area = 10

#         self.surface = initmesh.surface
#            self.elem = initmesh.elem
#            self.node = radius * initmesh.node
#            self.radius = radius
# 
# #         %%%% computing surface area
# #         %%%% Only for spherical surface 
# #         %%%area = spherearea(self.node, self.elem)
# #         %%%area_s = sum(area(:))
# # 
# #         %%%% computing simplex area
#         self.area = simplexvolume(self.node, self.elem)
#         self.tol_area = np.sum(self.area(:))
#################################################################

class Cmp_pmt_data(Cmp_mesh_data):

    def __init__(self, fields, fieldtype='mu',
                 Nspecies=2, Nblend=1, Nblock=2, 
                 Ndeg=100, fA=0.8, chiAB=0.3,
                 dim=2, dtMax=0.1, TOL=1.0e-6,
                 Maxiter=5000, Initer=200):

        Cmp_mesh_data.__init__(self)

        self.Nspecies = Nspecies
        self.Nblend   = Nblend
        self.Nblock   = Nblock
        self.Ndeg     = Ndeg
        self.fA       = fA
        self.chiAB    = chiAB

        self.dim   = dim
        self.Ndof  = self.node    ###
        self.Nelem = self.elem    ###


        self.dtMax = dtMax

        self.TOL = TOL
        self.Maxiter = Maxiter
        self.Initer = Initer
        
        self.Interval = np.zeros((self.Nblock, 4))

        self.Interval[0,0] = 0.0
        self.Interval[0,1] = self.fA
        self.Interval[0,2] = np.ceil( (self.Interval[0,1] - self.Interval[0,0])/self.dtMax )
        self.Interval[0,3] = (self.Interval[0,1] - self.Interval[0,0]) / self.Interval[0,2]

        self.Interval[1,0] = self.fA
        self.Interval[1,1] = 1.0
        self.Interval[1,2] = np.ceil( (self.Interval[1,1] - self.Interval[1,0])/self.dtMax )
        self.Interval[1,3] = (self.Interval[1,1] - self.Interval[1,0]) / self.Interval[1,2]

        self.Nt = int(self.Interval[0,2] + self.Interval[1,2])

        self.q     = np.zeros((self.Nt, self.Ndof))
        self.qplus = np.zeros((self.Nt, self.Ndof))
        self.sQ    = np.zeros((self.Nspecies-1, self.Nblend))
        self.rho   = np.zeros((self.Nspecies, self.Ndof))
        self.w     = np.zeros((self.Nspecies, self.Ndof))
        self.mu    = np.zeros((self.Nspecies, self.Ndof))
        self.grad  = np.zeros((self.Nspecies, self.Ndof))

        self.mu_old   = np.zeros((self.Nspecies, self.Ndof))
        self.rho_old  = np.zeros((self.Nspecies, self.Ndof))
        self.grad_old = np.zeros((self.Nspecies, self.Ndof))

#############################################################################
        
    def initialize(self, fields, fieldtype):

        chiN = self.chiAB * self.Ndeg

        if fieldtype is 'fieldmu':
            self.mu = fields
            self.w[0,:] = fields[0,:] - fields[1,:]
            self.w[1,:] = fields[0,:] + fields[1,:]
        if fieldtype is 'fieldw':
            self.w = fields
            self.mu[0,:] = 0.5*(fields[0,:]+fields[1,:])
            self.mu[1,:] = 0.5*(fields[1,:]-fields[0,:])

        self.rho[0,:] = 0.5 + self.mu[1,:]/chiN
        self.rho[1,:] = 1.0 - self.rho[0,:]

#############################################################################

    def updatePropagator(self)
        
        q_index = 0
        self.q[0,:] = 1.0

        t0 = self.Interval[0,0] 
        t1 = self.Interval[0,1]
        dn = self.Interval[0,2]
        dt = self.Interval[0,3]

        PDEsolve(self.q, q_index, dn, dt, q_index)

        q_index = q_index + dn



        

#     def evlSaddle(self)
# 
#         updatePropagator(self);
# 
#         while True:
#             
#             updatePropagator(cmp_pmt, cmp_mesh, scft, Mthd);
# 
#             q_times_qplus = scft.q.*flipdim(scft.qplus, 1);
# 
#             scft.sQ = updateQ(scft, cmp_mesh, q_times_qplus);
# 
#             scft = updateDensity(scft, cmp_pmt, q_times_qplus);
# 
#             [scft, err] = updateField(scft, Mthd.Iter, sys_pmt);





