import numpy as np
import matplotlib.pyplot as plt

from fealpy.functionspace import ConformingVirtualElementSpace2d
from fealpy.mesh.mesh_tools import show_mesh_2d

from timeit import default_timer as timer

from fealpy.timeintegratoralg.timeline import ChebyshevTimeLine,UniformTimeLine
from ParabolicVEMSolver2d import ParabolicVEMSolver2d

def scftmodel2d_options(
        nspecies = 2,
        nblend = 1,
        nblock = 2,
        ndeg = 100,
        fA = 0.2,
        chiAB = 0.25,
        dim = 2,
        T0 = 4,
        T1 = 16,
        nupdate = 1,
        order = 1,
        rdir='.'):
    """
    Get the options used in model.
    """

    # the parameter for scft model
    options = {
            'nspecies'    :nspecies,
            'nblend'      :nblend,
            'nblock'      :nblock,
            'ndeg'        :ndeg,
            'fA'          :fA,
            'chiAB'       :chiAB,
            'dim'         :dim,
            'T0'          :T0,
            'T1'          :T1,
            'nupdate'     :nupdate,
            'order'       :order,
            'rdir'        :rdir
            }

    options['chiN'] = options['chiAB']*options['ndeg']
    return options

class SCFTVEMModel2d():
    def __init__(self, mesh, options=None):
        if options == None:
            options = scftmodel_options()
        self.options = options

        self.vemspace = ConformingVirtualElementSpace2d(mesh, p=options['order'])
        self.mesh = self.vemspace.mesh
        self.totalArea = np.sum(self.vemspace.smspace.cellmeasure)
        self.count = 0

        fA = options['fA']
        T0 = options['T0']
        T1 = options['T1']
        self.timeline0 = ChebyshevTimeLine(0, fA, T0)
        self.timeline1 = ChebyshevTimeLine(fA, 1, T1)

        N = T0 + T1 + 1
        gdof = self.vemspace.number_of_global_dofs()
        self.gof = gdof

        self.q0 = np.ones((gdof, N), dtype=self.mesh.ftype)
        self.q1 = np.ones((gdof, N), dtype=self.mesh.ftype)

        self.rho = np.zeros((gdof, 2), dtype=self.mesh.ftype)
        self.grad = np.zeros((gdof, 2), dtype=self.mesh.ftype)
        self.w = np.zeros((gdof, 2), dtype=self.mesh.ftype)
        self.sQ1 = np.zeros((N, 1), dtype=self.mesh.ftype)

        self.sQ = 0.0
        self.nupdate = options['nupdate']
        self.A = self.vemspace.stiff_matrix()
        self.M = self.vemspace.mass_matrix()
        self.smodel = ParabolicVEMSolver2d(self.A, self.M)

        self.eta_ref = 0

    def recover_estimate_simple(self, rho):
        vemspace = self.vemspace
        NC = self.mesh.number_of_cells()
        NV = self.mesh.number_of_vertices_of_cells()
        cell, cellLocation = self.mesh.entity('cell')
        B = vemspace.matrix_B()

        barycenter = vemspace.smspace.barycenter
        ldof = vemspace.smspace.number_of_local_dofs()

        idx = np.repeat(range(NC), NV)
        S = vemspace.project_to_smspace(rho)
        grad = S.grad_value(barycenter)

        S0 = vemspace.smspace.function()
        S1 = vemspace.smspace.function()
        p2c = self.mesh.ds.node_to_cell()
        d = p2c.sum(axis=1)
        ruh = np.asarray((p2c@grad)/d.reshape(-1, 1))

        for i in range(ldof):
            S0[i::ldof] = np.bincount(idx, weights=B[i, :]*ruh[cell, 0], minlength=NC)
            S1[i::ldof] = np.bincount(idx, weights=B[i, :]*ruh[cell, 1], minlength=NC)

        node = self.mesh.node
        gx = S0.value(node[cell], idx) - np.repeat(grad[:, 0], NV)
        gy = S1.value(node[cell], idx) - np.repeat(grad[:, 1], NV)
        eta = np.bincount(idx, weights=gx**2+gy**2)/NV*self.area
        return np.sqrt(eta)

    def mix_estimate(self, u, w=0.5):
        #recovery grad
        gu = self.vemspace.grad_recovery(u[:,0])
        S = self.vemspace.project_to_smspace(gu)
        def f0(x, index):
            val = S.value(x, index)
            return np.sum(val**2, axis=-1)
        eta0 = self.vemspace.integralalg.integral(f0, celltype=True)
        gu = self.vemspace.grad_recovery(u[:,1])
        S = self.vemspace.project_to_smspace(gu)
        eta0 += self.vemspace.integralalg.integral(f0, celltype=True)

        #grad
        S0 = self.vemspace.project_to_smspace(u[:,0])
        def f1(x, index):
            val = S0.grad_value(x, index) - S.value(x, index)
            return np.sum(val**2, axis=-1)
        eta1 = self.vemspace.integralalg.integral(f1, celltype=True)

        S0 = self.vemspace.project_to_smspace(u[:,1])
        eta1 += self.vemspace.integralalg.integral(f1, celltype=True)

        return w*np.sqrt(eta0) + (1-w)*np.sqrt(eta1)

    def estimate(self, u):
        vemspace = self.vemspace
        NC = self.mesh.number_of_cells()
        NV = self.mesh.number_of_vertices_of_cells()
        cell = self.mesh.ds.cell

        barycenter = vemspace.smspace.cellbarycenter
        area = vemspace.smspace.cellmeasure
        ldof = vemspace.smspace.number_of_local_dofs()

        idx = np.repeat(range(NC), NV)
        S = vemspace.project_to_smspace(u[:,0])
        grad = S.grad_value(barycenter)
        eta = np.sum(grad*grad, axis=-1)*area

        S = vemspace.project_to_smspace(u[:, 1])
        grad = S.grad_value(barycenter)
        eta += np.sum(grad*grad, axis=-1)*area
        return np.sqrt(eta)


    def reinit(self, mesh):
        options = self.options
        self.vemspace = ConformingVirtualElementSpace2d(mesh, p=options['order'])
        self.mesh = self.vemspace.mesh
        self.totalArea = np.sum(self.vemspace.smspace.cellmeasure)

        fA = options['fA']
        T0 = options['T0']
        T1 = options['T1']
        self.timeline0 = ChebyshevTimeLine(0, fA, T0)
        self.timeline1 = ChebyshevTimeLine(fA, 1, T1)
        N = T0 + T1 + 1
        gdof = self.vemspace.number_of_global_dofs()
        self.gdof = gdof

        self.q0 = np.ones((gdof, N), dtype=self.mesh.ftype)
        self.q1 = np.ones((gdof, N), dtype=self.mesh.ftype)

        self.rho = np.zeros((gdof, 2), dtype=self.mesh.ftype)
        self.grad = np.zeros((gdof, 2), dtype=self.mesh.ftype)
        self.w = np.zeros((gdof, 2), dtype=self.mesh.ftype)
        self.sQ = np.zeros((N, 1), dtype=self.mesh.ftype)

        self.nupdate = options['nupdate']

        self.A = self.vemspace.stiff_matrix()
        self.M = self.vemspace.mass_matrix()

        self.smodel = ParabolicVEMSolver2d(self.A, self.M)

        self.eta_ref= 0

    def init_value(self, fieldstype = 1):
        gdof = self.vemspace.number_of_global_dofs()
        mesh = self.vemspace.mesh
        node = mesh.node
        chiN = self.options['chiN']
        fields = np.zeros((gdof, 2), dtype = mesh.ftype)
        mu = np.zeros((gdof, 2), dtype = mesh.ftype)
        w = np.zeros((gdof, 2), dtype = mesh.ftype)
        if fieldstype == 1:
            fields[:, 0] = chiN*(-1 + 2*np.random.rand(1, gdof))
            fields[:, 1] = chiN*(-1 + 2*np.random.rand(1, gdof))
        elif fieldstype == 2:
            pi = np.pi
            fields[:, 1] = chiN * (np.sin(3*node[:,0]) + np.cos(3*node[:, 1]))
        elif fieldstype == 3:
            pi = np.pi
            def f(p, k):
                return np.cos(np.cos(pi/3*k)*p[..., 0] + np.sin(pi/3*k)*p[..., 1])
            for k in range(6):
                fields[:, 1] += np.cos(self.vemspace.interpolation(lambda p :
                    f(p, k)))

            #fields[:, 1] *= chiN
        elif fieldstype == 4:
            def f(p):
                return np.sin(3*p[..., 0])
            fields[:, 1] += self.vemspace.interpolation(f)

        w[:, 0] = fields[:, 0] - fields[:, 1]
        w[:, 1] = fields[:, 0] + fields[:, 1]

        mu[:, 0] = 0.5*(w[:, 0] + w[:, 1])
        mu[:, 1] = 0.5*(w[:, 1] - w[:, 0])

        return mu

    def __call__(self, mu, returngrad=True):
        """
        目标函数，给定外场，计算哈密尔顿量及其梯度
        """
        self.w[:, 0] = mu[:, 0] - mu[:, 1]
        self.w[:, 1] = mu[:, 0] + mu[:, 1]

        start = timer()
        self.compute_propagator()
        print('Times for PDE solving:', timer() - start)
        self.compute_eta_ref(eta_ref='etamaxmin')

        self.compute_singleQ()
        self.compute_density1()


        u = mu[:,0]
        mu1_int = self.integral_space(u)

        u1 = mu[:,1]
        S = self.vemspace.project_to_smspace(u1)
        u = lambda x, index : S.value(x, index=index)**2

        mu2_int = self.vemspace.integralalg.integral(u)###TODO 区别
        #mu2_int = self.integral_space(u1**2)

        chiN = self.options['chiN']
        self.H = -mu1_int + mu2_int/chiN
        self.H = self.H/self.totalArea - np.log(self.sQ)

        self.save_data(fname= self.options['rdir']+'/test'+str(self.count)+'.mat')
        self.show_solution(self.count)
        self.count +=1
        self.grad[:, 0] = self.rho[:, 0]  + self.rho[:, 1] - 1.0
        self.grad[:, 1] = 2.0*mu[:, 1]/chiN - self.rho[:, 0] + self.rho[:, 1]

        if returngrad:
            return self.H, self.grad
        else:
            return self.H

    def compute_eta_ref(self,eta_ref=None):
        #TODO
        q = self.w.copy()
        q[:,0] = self.q0[:,-1]
        q[:,1] = self.q1[:,-1]

        self.eta = self.mix_estimate(q, w=1)
        if eta_ref == 'etamaxmin':
            self.eta_ref = np.std(self.eta)/(np.max(self.eta)-np.min(self.eta))
        if eta_ref == 'etamean':
            self.eta_ref = np.std(self.eta)/np.mean(self.eta)
        print('第'+str(self.count)+'次迭代的etaref:', self.eta_ref)

    def compute_propagator(self):
        n0 = self.timeline0.NL
        n1 = self.timeline1.NL

        w = self.w[:,0]
        S = self.vemspace.project_to_smspace(w)#TODO
        F = self.vemspace.cross_mass_matrix(S.value)
        smodel0 = ParabolicVEMSolver2d(self.A, self.M, F)

        w = self.w[:, 1]
        S = self.vemspace.project_to_smspace(w)
        F = self.vemspace.cross_mass_matrix(S.value)
        smodel1 = ParabolicVEMSolver2d(self.A, self.M, F)
        self.timeline0.time_integration(self.q0[:, 0:n0], smodel0,
                self.nupdate)
        self.timeline1.time_integration(self.q0[:, n0-1:], smodel1,
                self.nupdate)
        ###TODO 反向传播子的时间线时间步长对应位置
        self.timeline1.time_integration(self.q1[:, 0:n1], smodel1,
                self.nupdate)
        self.timeline0.time_integration(self.q1[:, n1-1:], smodel0,
                self.nupdate)

    def compute_singleQ(self):
        q = self.q0*self.q1[:, -1::-1]
        for i in range(len(q[0,:])):
            self.sQ1[i,:] = self.integral_space(q[:, i])/self.totalArea
        #print('sQ', self.sQ1)
        self.sQ = self.integral_space(self.q0[:, -1])/self.totalArea
        print('Q',self.sQ)

    def compute_density(self):
        q = self.q0*self.q1[:, -1::-1]
        n0 = self.timeline0.NL
        self.rho[:, 0] = self.integral_time(q[:, 0:n0], self.timeline0.dt)/self.sQ
        self.rho[:, 1] = self.integral_time(q[:, n0-1:], self.timeline1.dt)/self.sQ

    def compute_density1(self):
        q = self.q0*self.q1[:, -1::-1]
        n0 = self.timeline0.NL
        self.rho[:, 0] = self.timeline0.dct_time_integral(q[:, 0:n0],
                return_all=False)/self.sQ
        self.rho[:, 1] = self.timeline1.dct_time_integral(q[:,
            n0-1:],return_all=False)/self.sQ

    def compute_hamiltonian(self):
        wA = self.w[:,0]
        wB = self.w[:,1]
        rhoA = self.rho[:,0]
        rhoB = self.rho[:,1]

        chiN = self.options['chiN']

        H1 = chiN*self.integral_space(rhoA*rhoB)/self.totalArea

        H2 = self.integral_space(wA*rhoA+wB*rhoB)/self.totalArea
        H2 += np.log(self.sQ)

        self.H = H1 - H2

    def integral_time(self, q, dt):
        f = -0.625*(q[:, 0] + q[:, -1]) + 1/6*(q[:, 1] + q[:, -2]) - 1/24*(q[:, 2] + q[:, -3])
        f += np.sum(q, axis=1)
        f *= dt
        return f

    def integral_space(self, u):
        S = self.vemspace.project_to_smspace(u)
        Q = self.vemspace.integralalg.integral(S.value)
        return Q

    def output(self, tag, queue=None, stop=False):
        if queue is not None:
            if not stop:
                queue.put({tag:self.rho[:, 0]})
            else:
                queue.put(-1)


    def save_data(self, fname='test.mat'):
        import scipy.io as sio

        mesh = self.mesh
        node = mesh.entity('node')
        node = np.array(node)
        cell, cellLocation = mesh.entity('cell')
        Q = self.sQ1
        H = self.H
        q = self.q0
        q1 = self.q1

        mu = np.zeros((self.w.shape))
        mu[:, 0] = 0.5*(self.w[:, 0] + self.w[:, 1])
        mu[:, 1] = 0.5*(self.w[:, 1] - self.w[:, 0])

        eta = self.eta
        eta_ref = self.eta_ref

        data = {
                'node':node,
                'cell':cell,
                'cellLocation':cellLocation,
                'rho':self.rho,
                'Q':Q,
                'H':H,
                'mu':mu,
                'q0':q,
                'q1':q1,
                'eta':eta,
                'eta_ref':eta_ref
                }
        sio.savemat(fname, data)

    def show_solution(self,i):
        options = self.options
        mesh = self.mesh
        cell, cellLocation = mesh.entity('cell')
        N = len(cellLocation)
        d = np.zeros(N-1)
        for j in range(N-1):
            newcell = cell[cellLocation[j]:cellLocation[j+1]]
            c = self.rho[:,0].view(np.ndarray)
            #c = u[:,1].view(np.ndarray)
            d[j] = np.sum(c[newcell],axis=0)/len(newcell)
        node = mesh.node
        fig = plt.figure()
        axes = fig.gca()
        show_mesh_2d(axes,mesh,cellcolor=d,cmap='jet',linewidths=0.0,markersize=10, showaxis=False,showcolorbar= True)
        plt.savefig(options['rdir']+'/figure'+str(i)+'.png')
        plt.savefig(options['rdir']+'/figure'+str(i)+'.pdf')
        plt.close()

