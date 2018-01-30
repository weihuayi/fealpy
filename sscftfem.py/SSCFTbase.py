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
                 dim=2, dtMax=0.1, 
                 TOL=1.0e-6, TOL_R=1.0e-3, 
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
        self.TOL_R = TOL_R
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

    def updatePropagator(self):
        
    ### ====================================
        q_index = 0
        self.q[0,:] = 1.0

        t0 = self.Interval[0,0] 
        t1 = self.Interval[0,1]
        dn = int(self.Interval[0,2])
        dt = self.Interval[0,3]

        PDEsolve(self.q, self.w[0,:], q_index, dn, dt, q_index)

    ### ====================================
        q_index = q_index + dn
        t0 = self.Interval[1,0] 
        t1 = self.Interval[1,1]
        dn = int(self.Interval[1,2])
        dt = self.Interval[1,3]

        PDEsolve(self.q, self.w[1,:], q_index, dn, dt, q_index)

    ### ====================================
        qplus_index = 0
        self.qplus[0,:] = 1.0
        t0 = self.Interval[1,0] 
        t1 = self.Interval[1,1]
        dn = int(self.Interval[1,2])
        dt = self.Interval[1,3]
        
        PDEsolve(self.qplus, self.w[1,:], qplus_index, dn, dt, qplus_index)

    ### ====================================
        qplus_index = qplus_index + dn
        t0 = self.Interval[0,0] 
        t1 = self.Interval[0,1]
        dn = int(self.Interval[0,2])
        dt = self.Interval[0,3]
        
        PDEsolve(self.qplus, self.w[0,:], qplus_index, dn, dt, qplus_index)

#############################################################################


    def integral_space():


        return Q

#############################################################################

    def integral_nonlinear_term():

        return f

#############################################################################

    def integral_time(start, dt, final, integrand):
        
        f = -0.625* ( integrand[start,:] + integrand[start+final,:] ) +
            (1/6)* ( integrand[start+1,:] + integrand[start+final-1,:] ) -
            (1/24)* ( integrand[start+2,:] + integrand[start+final-2,:] )

        f = f + np.sum(integrand[start:start+final,:], 0);
        f = dt*f;
        
        return f

#############################################################################


    def updateQ(self, integrand):

        f = integral_space(self, integrand)

        return f

#############################################################################

    def updateDensity(self, integrand):
        
    ### ====================================
        index = 0

        t0 = self.Interval[0,0] 
        t1 = self.Interval[0,1]
        dn = int(self.Interval[0,2])
        dt = self.Interval[0,3]

        self.rho[0,:] = integral_time(index, dt, dn, integrand)

    ### ====================================
        index = index + dn 
        t0 = self.Interval[1,0] 
        t1 = self.Interval[1,1]
        dn = int(self.Interval[1,2])
        dt = self.Interval[1,3]

        self.rho[1,:] = integral_time(index, dt, dn, integrand)

        self.rho = self.rho/self.sQ[0,0]

#############################################################################

    def updateField(self):

        chiN = self.chiAB * self.Ndeg
        err = np.arange(self.grad.shape[0])
        
        lambd = np.array([2,2])

        self.grad[0,:] = self.rho[0,:]  + self.rho[1,:] - 1.0
        self.grad[1,:] = 2.0*self.mu[1,:]/chiN - self.rho[0,:] + self.rho[1,:]

        err[0] = (np.abs(self.grad[0,:])).max()
        err[1] = (np.abs(self.grad[1,:])).max()


        self.mu[0,:] = self.mu[0,:] + lambd[0]*self.grad[0,:]
        self.mu[1,:] = self.mu[1,:] - lambd[1]*self.grad[1,:]


        self.w[0,:] = self.mu[0,:] - self.mu[1,:] 
        self.w[1,:] = self.mu[1,:] + self.mu[1,:]

        return err 
    
#############################################################################

    def updateHamilton(self)

        chiN = self.chiAB * self.Ndeg

        mu1_int = integral_space(self.mesh, self.mu[0,:])
        mu2_int = integral_nonlinear_term(self.mesh, self.mu[1,:])
        
        H = -mu1_int + mu2_int/chiN

        return H

#############################################################################

    def calPartialF(self, radius)
 
        self.radius = radius   ##  ????????

        #######   
        ##  renew mesh ## ?????
 
        updatePropagator(self)

        q_times_qplus = np.mat( np.array(self.q) * np.array(self.qplus[::-1,:]) ) ## 矩阵元素相乘

        self.sQ[0,0] = self.updateQ(self, q_times_qplus)

        partialF = -np.sum(sQ.reshape(-1))

        return partialF

#############################################################################

    def evlSaddle(self):
        
        res = np.inf
        Hold = np.inf
        ediff = np.inf
        iteration = 0

        error = np.arange(self.grad.shape[0])

        while (res > self.TOL) and (iteration < self.Maxiter):
            
            updatePropagator(self); 

            q_times_qplus = np.mat( np.array(self.q) * np.array(self.qplus[::-1,:]) ) ## 矩阵元素相乘

            self.sQ[0,0] = self.updateQ(self, q_times_qplus)

            updateDensity(self, q_times_qplus)

            error = updateField(self)
        
            H = updateHamilton(self)

            H = H/self.tol_area - np.log(self.sQ(0,0))

            res = error.max()                      

            ediff = H - Hold
            Hold = H

#############################################################################

    def adaptSurface(self)
    
        dh = 1.0e-2
        gamma = 0.2

        eps = np.inf
        iteration = 0
        oldgradB = 0

        while(eps > self.TOL_R) and (iteration < 50)
            
            iteration = iteration + 1

    ####### compute difference quotient instead of calculating
    #######     the value of derivate
            rad = self.radius

            rrad = rad + dh
            fr = calPartialF(self, rrad)

            lrad = rad - dh 
            lr = calPartialF(self, lrad)

            gradB = (fr-fl) / (2.0*dh) 

            if iteration == 1:
                grad = -gradB
            else:
                crit = np.abs(gradB*oldgradB) - gamma*oldgradB*oldgradB
                if(crit >= 0 and iteration > 2):
#                     printf('iter %d: crit = %.15e\t Restart CG', iter, crit)
		####  restarted CG 
                    grad = -gradB;
                else:
                #### Fletcher-Reeves formula
	        ####  beta = gradB*gradB / (oldgradB*oldgradB)
		####  Polak-Ribiere formula
                    beta = gradB*(gradB-oldgradB) / (oldgradB*oldgradB)
                    beta = np.max((0, beta))
                    grad = grad*beta - gradB
            
            intval = lineEvalIntval(self, grad)
            alpha = lineSearchFMin(self, intval, grad, 1.0e-3)

            oldgradB = gradB
            self.radius = rad - alpha*gradB

            eps = np.sqrt(grad**2)

#############################################################################

    def lineEvalIntval(self, grad)
    
        b = 0.0
        a0= 0.0
        h = 0.01
        t = 2.0

        rad = self.radius + grad*a0
        phi0 = calPartialF(self, rad)
        k = 0

        intval = np.arange(2)

        while True:
            k = k+1

            a1 = a0 + h
            rad = self.radius + grad*a1
            phi1 = calPartialF(self, rad)

            if(phi1 < phi0):
                h = t*h
                b = a0
                a0 = a1
                phi0 = phi1
            elif k==1:
                h = -1.0*h
                a0 = a1
                phi0 = phi1
            else: 
                intval[0] = np.min((b,a1))
                intval[1] = np.max((b,a1))
                break

        return intval

#############################################################################

    def lineSearchFMin(self, intval, grad, tol)
        
        step = np.inf
        a0 = intval[0]
        b0 = intval[1]
        p0 = a0 + 0.382*(b0-a0)
        p1 = a0 + 0.618*(b0-a0)

        rad = self.radius 

        lrad = rad + grad*p0
        phi0 = calPartialF(self, lrad)

        rrad = rad + grad*p1 
        phi1 = calPartialF(self, rrad)

        err = 1

        while True:
            if(phi0 > phi1):
                if(np.abs(b0-p0)<=tol):
                    step = p1
                    break
                else:
                    a0 = p0
                    b0 = b0
                    p0 = p1
                    phi0 = phi1
                    p1 = a0 + 0.618*(b0-a0)

                    rrad = rad + grad*p1

                    phi1 = calPartialF(self, rrad)
            else:
                if(np.abs(a0-p1)<=tol):
                    step = p0
                    break
                else:
                    a0 = a0
                    b0 = p1
                    p1 = p0 
                    phi1 = phi0
                    p0 = a0 + 0.382*(b0-a0)
                    lrad = rad + grad*p0
                    phi0 = calPartialF(self, lrad)

    return step

#############################################################################

    def scftIteration(self)


