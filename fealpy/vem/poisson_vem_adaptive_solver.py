import numpy as np


class PoissonACVEMSolver:

    def __init__(self, pde, mesh, p=1):
        self.mesh = mesh
        self.pde = pde
        self.p = p


    def solve(self):

        mesh = self.mesh
        pde = self.pde
        space = ConformingScalarVESpace2d(mesh, p=self.p)
        uh = space.function()
        
        NDof[i] = space.number_of_global_dofs()
      
        #组装刚度矩阵 A 
        m = ScaledMonomialSpaceMassIntegrator2d()
        M = m.assembly_cell_matrix(space.smspace)

        d = ConformingVEMDoFIntegrator2d()
        D = d.assembly_cell_matrix(space, M)

        h1 = ConformingScalarVEMH1Projector2d(D)
        PI1 = h1.assembly_cell_matrix(space)
        G = h1.G

        li = ConformingScalarVEMLaplaceIntegrator2d(PI1, G, D)
        bform = BilinearForm(space)
        bform.add_domain_integrator(li)
        A = bform.assembly()

        #组装右端 F
        l2 = ConformingScalarVEML2Projector2d(M, PI1)
        PI0 = l2.assembly_cell_matrix(space)

        si = ConformingVEMScalarSourceIntegrator2d(pde.source, PI0)
        lform = LinearForm(space)
        lform.add_domain_integrator(si)
        F = lform.assembly()

        #处理边界 
        bc = DirichletBC(space, pde.dirichlet)
        A, F = bc.apply(A, F, uh)

        uh[:] = spsolve(A, F)

        uh.M = M
        uh.PI1 = PI1

        return uh

    def adaptive_solve(self, maxit=40, theta=0.2):
        for i in range(maxit):
            uh = self.solve()
            space = uh.space
            sh = space.project_to_smspace(uh, uh.PI1)

            estimator = PoissonCVEMEstimator(space, uh.M, uh.PI1)
            eta = estimator.residual_estimate(uh, pde.source)
            
            errorMatrix[0, i] = mesh.error(pde.solution, sh.value)
            errorMatrix[1, i] = mesh.error(pde.gradient, sh.grad_value)

            errorMatrix[2, i] = np.sqrt(np.sum(eta))
            options = Hmesh.adaptive_options(HB=None)
            Hmesh.adaptive(eta, options)
            newcell, cellocation = Hmesh.entity('cell')
            newnode = Hmesh.entity("node")[:]
            self.mesh = PolygonMesh(newnode, newcell, cellocation)
