from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod,cartesian
from fealpy.model import ComputationalModel
from fealpy.fem import DirichletBC
from .simulation.fem import Newton
from .equation import IncompressibleNS

class IncompressibleNSLFEM2DModel(ComputationalModel):
    def __init__(self, pde, mesh=None, options = None):
        super().__init__(pbar_log = True, log_level = "INFO")
        self.pde = pde
        self.equation = IncompressibleNS(self.pde)
        self.fem = self.method[options['method']]()
        self.init_timeline = self.pde.init_timeline(options['T0'], options['T1'], options['nt'])
        self.fem.dt = self.pde.dt

        if mesh is None:
            if hasattr(pde, 'mesh'):
                self.mesh = pde.mesh
            else:
                raise ValueError("Not found mesh!")
        else:
            self.mesh = mesh
        if options is not None:
            self.solve.set(options['solve'])
            self.fem = self.method[options['method']]()

            run = self.run[options['run']]
            if options['run'] == 'uniform_refine':
                run(maxit=options['maxit'], maxstep=options['maxstep'], tol=options['tol'], apply_bc=options['apply_bc'], postprocess=options.get('postprocess', 'error'))
            else:  # 'one_step' 或其他
                run(maxstep=options['maxstep'], tol=options['tol'], apply_bc=options['apply_bc'], postprocess=options.get('postprocess', 'error'))

    def lagrange_multiplier(self, A, b):
        from fealpy.fem import LinearForm, SourceIntegrator, BlockForm
        from fealpy.sparse import COOTensor

        LagLinearForm = LinearForm(self.fem.pspace)
        LagLinearForm.add_integrator(SourceIntegrator(source=1))
        LagA = LagLinearForm.assembly()
        LagA = bm.concatenate([bm.zeros(self.fem.uspace.number_of_global_dofs()), LagA], axis=0)

        A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                                 bm.arange(len(LagA), dtype=bm.int32)]), LagA, spshape=(1, len(LagA)))

        A = BlockForm([[A, A1.T], [A1, None]])
        A = A.assembly_sparse_matrix(format='csr')
        b0 = bm.array([0])
        b  = bm.concatenate([b, b0], axis=0)

        return A, b

    @variantmethod("Newton")
    def method(self):
        self.fem = Newton(self.equation)
        self.method_str = "Newton"
        return self.fem
    
    @method.register("Ossen")
    def method(self):
        from .simulation.fem import Ossen
        self.fem = Ossen(self.equation)
        self.method_str = "Ossen"
        return self.fem
    
    @method.register("IPCS")
    def method(self):
        from .simulation.fem import IPCS
        self.fem = IPCS(self.equation)
        self.method_str = "IPCS"
        return self.fem
    
    @method.register("BDF2")
    def method(self):
        from .simulation.fem import BDF2
        self.fem = BDF2(self.equation)
        self.method_str = "BDF2"
        return self.fem

    def update(self, u0, u00):
        self.fem.update(u0, u00)

    def linear_system(self):
        BForm = self.fem.BForm()
        LForm = self.fem.LForm()
        return BForm, LForm
    
    @variantmethod("dirichlet")
    def apply_bc(self, A, b):
        BC = DirichletBC(
            (self.fem.uspace, self.fem.pspace), 
            gd=(lambda p: self.pde.velocity(p, self.t), lambda p: self.pde.pressure(p, self.t)), 
            threshold=(self.pde.is_velocity_boundary, self.pde.is_pressure_boundary),
            method='interp')
        A, b = BC.apply(A, b)
        return A, b
    
    @apply_bc.register("None")
    def apply_bc(self, A, b):
        BC = DirichletBC(
            (self.fem.uspace, self.fem.pspace), 
            gd=(lambda p: self.pde.velocity(p, self.t), lambda p: self.pde.pressure(p, self.t)), 
            threshold=(None, None),
            method='interp')
        A, b = BC.apply(A, b)
        return A, b

    @apply_bc.register("neumann")
    def apply_bc(self, A, b):
        pass

    @variantmethod('direct')
    def solve(self, A, F, solver = 'scipy'):
        from fealpy.solver import spsolve
        return spsolve(A, F, solver=solver)

    @solve.register('amg')
    def solve(self, A, F):
        raise NotImplementedError("AMG solver not yet implemented.")

    @solve.register('pcg')
    def solve(self, A, F):
        pass

    @variantmethod('one_step_iter')
    def run(self, u0, p0, maxstep=10, tol=1e-12, apply_bc = 'dirichlet'):
        fem = self.fem
        pde = self.pde

        if self.method_str == "IPCS":
            BCu = DirichletBC(space=fem.uspace, 
                gd = cartesian(lambda p : pde.velocity(p, self.t)), 
                threshold=pde.is_velocity_boundary, 
                method='interp')

            BCp = DirichletBC(space=fem.pspace, 
                gd = cartesian(lambda p : pde.pressure(p, self.t)), 
                threshold=pde.is_pressure_boundary, 
                method='interp')
            
            uh1 = u0.space.function()
            uhs = u0.space.function()
            ph1 = p0.space.function()

            A0, b0 = self.fem.predict_velocity(u0, p0, BC=BCu, threshold=False, return_form=False)
            uhs[:] = self.solve(A0, b0)

            A1, b1 = self.fem.pressure(uhs, p0, BC=BCp, return_form=False)
            ph1[:] = self.solve(A1, b1)

            A2, b2 = self.fem.correct_velocity(uhs, p0, ph1, return_form=False)
            uh1[:] = self.solve(A2, b2)
            return uh1, ph1
        else:
            BForm, LForm = self.linear_system()
            ugdof = fem.uspace.number_of_global_dofs()
            pde.mesh.nodedata['u'] = u0.reshape(2, -1).T
            pde.mesh.nodedata['p'] = p0
            u1 = u0
            p1 = p0

            inneru_0 = fem.uspace.function()
            inneru_0[:] = u0[:]
            inneru_1 = fem.uspace.function()

            for j in range(maxstep): 
                fem.update(inneru_0, u0)
                
                A = BForm.assembly()
                b = LForm.assembly()
                A,b = self.apply_bc[apply_bc](A, b)
                # A, b = self.lagrange_multiplier(A, b)
            
                x = self.solve(A, b, 'mumps')
                inneru_1[:] = x[0:ugdof]
                p1[:] = x[ugdof:]
                res_u = pde.mesh.error(inneru_0, inneru_1)
                inneru_0[:] = inneru_1
                if res_u < tol:
                    # print(error)
                    u1[:] = inneru_1
                    break
            return u1, p1

    @run.register('time_step')
    def run(self, t0 = 0,nt = 300, maxstep = 10, tol = 1e-10, apply_bc = 'dirichlet', postprocess = 'error'):
        fem = self.fem
        pde = self.pde

        u0 = fem.uspace.interpolate(pde.velocity_0)
        p0 = fem.pspace.interpolate(pde.pressure_0)
        # u0 = fem.uspace.function()
        # p0 = fem.pspace.function()


        for i in range(nt):
            self.t = t0 + (i+1)*self.fem.dt
            # print(f"第{i+1}步")
            # print("time=", self.t)
            u1,p1 = self.run['one_step'](u0, p0, maxstep, tol, apply_bc)

            u0[:] = u1
            p0[:] = p1
            
            self.pde.mesh.nodedata['u'] = u1.reshape(2,-1).T
            self.pde.mesh.nodedata['p'] = p1

            # uerror = self.pde.mesh.error(self.pde.velocity, u1)
            # perror = self.pde.mesh.error(self.pde.pressure, p1)
            # print("uerror:", uerror)
            # print("perror:", perror)
        uerror, perror = self.postprocess[postprocess](u1, p1) 
        self.logger.info(f"final uerror: {uerror}, final perror: {perror}")

    @run.register('uniform_refine')
    def run(self, maxit=5, t0 = 0, nt = 300, maxstep = 10, tol = 1e-12, apply_bc = 'dirichlet', postprocess = 'error'):
        fem = self.fem
        for i in range(maxit):
            self.logger.info(f'mesh: {self.pde.mesh.number_of_cells()}')
            self.run['time_step'](t0, nt, maxstep, tol,apply_bc, postprocess)
            self.pde.mesh.uniform_refine()
            self.equation = IncompressibleNS(self.pde)

        
    @variantmethod('error')
    def postprocess(self, uh, ph):
        uerror = self.pde.mesh.error(lambda p : self.pde.velocity(p, t = self.t), uh)
        perror = self.pde.mesh.error(lambda p : self.pde.pressure(p, t = self.t), ph)
        # self.logger.info(f"uerror: {uerror}, perror: {perror}")
        return uerror, perror