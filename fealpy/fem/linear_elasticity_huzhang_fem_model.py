import matplotlib.pyplot as plt
from typing import Union

from ..backend import bm
from ..mesh import Mesh
from ..functionspace import HuZhangFESpace, TensorFunctionSpace, LagrangeFESpace
from ..material import LinearElasticMaterial
from ..fem import BilinearForm, LinearForm, BlockForm
from ..fem import VectorSourceIntegrator
from ..fem.huzhang_stress_integrator import HuZhangStressIntegrator
from ..fem.huzhang_mix_integrator import HuZhangMixIntegrator
from ..model import PDEModelManager, ComputationalModel
from ..model.linear_elasticity import LinearElasticityPDEDataT
from ..decorator import variantmethod
from ..solver import spsolve,LinearElasticityHZFEMFastSolver
from ..tools.show import show_error_table, showmultirate

class LinearElasticityHuzhangFEMModel(ComputationalModel):
    """
    A class to represent a linear elasticity problem using the
    HuZhang finite element method.

    Reference:
        https://wnesm678i4.feishu.cn/wiki/P3FWwKgx8iURXVkLY8UcdHcPnHg?fromScene=spaceOverview
    """
    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
        self.set_pde(options['pde'])
        self.set_material_parameters(self.pde.lam(), self.pde.mu())
        self.set_init_mesh(options['init_mesh'])
        self.set_space_degree(options['space_degree'])

    def set_pde(self, pde: Union[LinearElasticityPDEDataT, str]="boxtri2d"):
        if isinstance(pde, str):
            self.pde = PDEModelManager('linear_elasticity').get_example(pde)
            self.logger.info(f"PDE initialized from string: '{pde}'")
        else:
            self.pde = pde
            pde_name = type(pde).__name__
            self.logger.info(f"PDE initialized from instance: {pde_name}") 

    def set_init_mesh(self, mesh: Union[Mesh, str] = "uniform_tri", **kwargs):
        if isinstance(mesh, str):
            self.mesh = self.pde.init_mesh[mesh](**kwargs)
            mesh_type = mesh
            self.logger.info(f"Mesh type: '{mesh_type}'")
        else:
            self.mesh = mesh
            mesh_type = type(mesh).__name__
            self.logger.info(f"Mesh type: custom {mesh_type} instance")

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")

    def set_material_parameters(self, lam: float, mu: float):
        self.material = LinearElasticMaterial("isoparametric", lame_lambda=lam, shear_modulus=mu)
        self.logger.info(f"Material parameters set: λ (Lamé first parameter) = {lam}, μ (shear modulus) = {mu}")



    def set_space_degree(self, p: int):
        self.p = p

    def linear_system(self, mesh, p):
        GD = self.mesh.geo_dimension()
        lambda0, lambda1 = self.pde.stress_matrix_coefficient() 

        self.space_sigma = HuZhangFESpace(mesh, p=p)
        self.space = LagrangeFESpace(mesh, p=p-1, ctype='D')
        self.space_u = TensorFunctionSpace(scalar_space=self.space, shape=(GD, -1))

        bform1 = BilinearForm(self.space_sigma)
        bform1.add_integrator(HuZhangStressIntegrator(lambda0=lambda0, lambda1=lambda1))

        bform2 = BilinearForm((self.space_u, self.space_sigma))
        bform2.add_integrator(HuZhangMixIntegrator())

        A = BlockForm([[bform1,   bform2],
                       [bform2.T, None]])
        A = A.assembly()

        lform1 = LinearForm(self.space_u)
        lform1.add_integrator(VectorSourceIntegrator(source=self.pde.body_force))

        b = lform1.assembly()

        gdof_sigma = self.space_sigma.number_of_global_dofs()
        F = bm.zeros(A.shape[0], dtype=A.dtype)
        F[gdof_sigma:] = -b

        return A, F, self.space_sigma, self.space_u
    
    def apply_bc(self, A, F):
        pass

    @variantmethod("direct")
    def solve(self):
        A, F, space_sigma, space_u = self.linear_system(self.mesh, self.p)
        X = spsolve(A, F, solver='mumps')
        
        gdof_sigma = space_sigma.number_of_global_dofs()
        sigma_h = space_sigma.function()
        u_h = space_u.function()

        sigma_h[:] = X[:gdof_sigma]
        u_h[:] = X[gdof_sigma:]

        return sigma_h, u_h
    
    @solve.register('fast')
    def solve(self):
        pass
    
    @variantmethod('onestep')
    def run(self):
        sigma_h, u_h = self.solve()
        l2_u = self.mesh.error(u_h, self.pde.displacement)
        l2_sigma = self.mesh.error(sigma_h, self.pde.stress)

        self.logger.info(f"u L2 error (u): {l2_u}, L2 error (σ): {l2_sigma}")

    @run.register('uniform_refine')
    def run(self, maxit=4):
        errorType = [
                 '$|| \\boldsymbol{\\sigma} - \\boldsymbol{\\sigma}_h||_{L_2}$',
                 '$|| \\boldsymbol{u} - \\boldsymbol{u}_h||_{L_2}$',
                 ]
        errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
        h = bm.zeros(maxit, dtype=bm.float64)

        for i in range(maxit):
            N =  2**(i+1)
            sigma_h, u_h = self.solve()
            l2_u = self.mesh.error(u_h, self.pde.displacement)
            l2_sigma = self.mesh.error(sigma_h, self.pde.stress)

            h[i] = 1 / N
            errorMatrix[0, i] = l2_sigma
            errorMatrix[1, i] = l2_u 

            if i < maxit - 1:
                self.mesh.uniform_refine()
    
        show_error_table(h, errorType, errorMatrix)
        showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
        plt.show()

    def set_space_degree(self, p: int):
        self.p = p

    def linear_system(self, mesh, p):
        GD = self.mesh.geo_dimension()
        lambda0, lambda1 = self.pde.stress_matrix_coefficient() 

        self.space_sigma = HuZhangFESpace(mesh, p=p)
        self.space = LagrangeFESpace(mesh, p=p-1, ctype='D')
        self.space_u = TensorFunctionSpace(scalar_space=self.space, shape=(-1,GD))

        bform1 = BilinearForm(self.space_sigma)
        bform1.add_integrator(HuZhangStressIntegrator(lambda0=lambda0, lambda1=lambda1))

        bform2 = BilinearForm((self.space_u, self.space_sigma))
        bform2.add_integrator(HuZhangMixIntegrator())

        A = BlockForm([[bform1,   bform2],
                       [bform2.T, None]])
        A = A.assembly()

        lform1 = LinearForm(self.space_u)
    
        lform1.add_integrator(VectorSourceIntegrator(source=self.pde.body_force))

        b = lform1.assembly()

        gdof_sigma = self.space_sigma.number_of_global_dofs()
        F = bm.zeros(A.shape[0], dtype=A.dtype)
        F[gdof_sigma:] = -b

        return A, F, self.space_sigma, self.space_u
    
    def apply_bc(self, A, F):
        pass

    @variantmethod("direct")
    def solve(self):
        A, F, space_sigma, space_u = self.linear_system(self.mesh, self.p)
        X = spsolve(A, F, solver='scipy')
        info = {}
        info['residual'] = bm.linalg.norm(A @ X - F, ord=2)
        gdof_sigma = space_sigma.number_of_global_dofs()
        sigma_h = space_sigma.function()
        u_h = space_u.function()

        sigma_h[:] = X[:gdof_sigma]
        u_h[:] = X[gdof_sigma:]

        return sigma_h, u_h, info
    
    @solve.register('gmres')
    def solve(self):
        A, F, space_sigma, space_u = self.linear_system(self.mesh, self.p)
        X,info = LinearElasticityHZFEMFastSolve(A, F, self.space, solver='gmres', rtol=1e-8, restart=20, maxit=None).solve()
    
        gdof_sigma = space_sigma.number_of_global_dofs()
        sigma_h = space_sigma.function()
        u_h = space_u.function()

        sigma_h[:] = X[:gdof_sigma]
        u_h[:] = X[gdof_sigma:]

        return sigma_h, u_h,info
    
    @solve.register('minres')
    def solve(self):
        A, F, space_sigma, space_u = self.linear_system(self.mesh, self.p)
        X,info = LinearElasticityHZFEMFastSolver(A, F, self.space, solver='minres', rtol=1e-8).solve()
    
        gdof_sigma = space_sigma.number_of_global_dofs()
        sigma_h = space_sigma.function()
        u_h = space_u.function()

        sigma_h[:] = X[:gdof_sigma]
        u_h[:] = X[gdof_sigma:]

        return sigma_h, u_h,info
        
    
    @variantmethod('onestep')
    def run(self):
        sigma_h, u_h = self.solve['direct']()
        l2_u = self.mesh.error(u_h, self.pde.displacement)
        l2_sigma = self.mesh.error(sigma_h, self.pde.stress)

        self.logger.info(f"u L2 error (u): {l2_u}, L2 error (σ): {l2_sigma}")

    @run.register('uniform_refine')
    def run(self, maxit=4):
        errorType = [
                 '$|| \\boldsymbol{\\sigma} - \\boldsymbol{\\sigma}_h||_{L_2}$',
                 '$|| \\boldsymbol{u} - \\boldsymbol{u}_h||_{L_2}$',
                 ]
        errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
        h = bm.zeros(maxit, dtype=bm.float64)

        for i in range(maxit):
            N =  2**(i+1)
            sigma_h, u_h = self.solve()
            l2_u = self.mesh.error(u_h, self.pde.displacement)
            l2_sigma = self.mesh.error(sigma_h, self.pde.stress)

            h[i] = 1 / N
            errorMatrix[0, i] = l2_sigma
            errorMatrix[1, i] = l2_u 

            if i < maxit - 1:
                self.mesh.uniform_refine()
    
        show_error_table(h, errorType, errorMatrix)
        showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
        plt.show()
        
    
    @run.register('performance')
    def run(self, maxit=2):
        
        methods = ['direct', 'gmres', 'minres']
        results = {m: {'time': [], 'res': [], 'iters': []} for m in methods}
        hs = []

        import time
        for i in range(maxit):
            h = 10*2**(i)
            hs.append(h)

            for m in methods:
                start = time.perf_counter()
                if m == 'direct':
                    _, _, info = self.solve['direct']()
                    elapsed = time.perf_counter() - start
                    results[m]['time'].append(elapsed)
                    results[m]['res'].append(info.get('residual', None))
                    results[m]['iters'].append(None)
                else:
                    _, _, info = self.solve[m]()
                    elapsed = time.perf_counter() - start
                    results[m]['time'].append(elapsed)
                    results[m]['res'].append(info.get('residual'))
                    results[m]['iters'].append(info.get('niter'))
            
            if i < maxit - 1:
                self.mesh.uniform_refine()

        hdr = 'lvl |   h   ' \
            + ''.join(f'| {m}_time ' for m in methods) \
            + ''.join(f'| {m}_res  ' for m in methods) \
            + ''.join(f'| {m}_iters ' for m in ['gmres','minres'])
        print(hdr)
        print('-'*len(hdr))

        for i, h in enumerate(hs):
            row = f'{i:3d} | {h:.3e} '
            for m in methods:
                row += f'| {results[m]["time"][i]:8.4f} '
            for m in methods:
                res = results[m]['res'][i]
                row += f'| {res if res is not None else "   N/A":8} '
            for m in ['gmres','minres']:
                it = results[m]['iters'][i]
                row += f'| {it if it is not None else "   N/A":6} '
            print(row)
