from typing import Union

from ..backend import bm
from ..decorator import variantmethod
from ..functionspace.lagrange_fe_space import LagrangeFESpace
from ..model.interface_poisson import InterfacePoissonDataT
from ..model import PDEModelManager, ComputationalModel
from ..fem import BilinearForm, LinearForm
from ..fem import ScalarSourceIntegrator, InterFaceSourceIntegrator, ScalarDiffusionIntegrator

class InterfacePoissonLFEMModel(ComputationalModel):
    """
    """
    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=True, log_level="INFO")
        self.pdm = PDEModelManager("interface_poisson")
        self.set_pde(options['pde'])
        self.set_init_mesh(options['init_mesh'])
        self.set_space_degree(options['space_degree'])

    def set_pde(self, pde: Union[InterfacePoissonDataT, str]="eletronic"):
        """
        """
        if isinstance(pde, str):
            self.pde = self.pdm.get_example(pde)
        else:
            self.pde = pde

    def set_init_mesh(self, meshtype: str = "tri", **kwargs):
        self.mesh, self.back_mesh = self.pde.init_mesh[meshtype](**kwargs)

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")

    def set_space_degree(self, p: int = 1) -> None:
        self.p = p

    def get_interface_face(self, mesh, level_function):
        ndim = mesh.number_of_nodes_of_cells()
        kwargs = bm.context(mesh.node)

        vertex = mesh.node[mesh.cell, :]
        bc = bm.ones(ndim, **kwargs)/(ndim)

        central_value = bm.einsum("c, qcd->qd", bc, vertex)
        adjCell = self.mesh.face_to_cell()

        isInterfaceCell = level_function(central_value)
        isInterfaceFace = (isInterfaceCell[adjCell][:, 0]*isInterfaceCell[adjCell][:, 1])<=0

        return isInterfaceFace
    
    def linear_system(self, return_form=False):
        """
        """
        mesh = self.mesh
        
        self.space = LagrangeFESpace(mesh, p=self.p)

        LDOF = self.space.number_of_local_dofs()
        GDOF = self.space.number_of_global_dofs()
        self.logger.info(f"local DOFs: {LDOF}, global DOFs: {GDOF}")

        pde = self.pde
        level_function = pde.level_function
        self.uh = self.space.function()

        isInterfaceFace = self.get_interface_face(mesh, level_function)
        space = self.space

        bform = BilinearForm(space)
        bform.add_integrator(ScalarDiffusionIntegrator())
        A = bform.assembly()

        lform = LinearForm(space)
        lform.add_integrator(InterFaceSourceIntegrator(pde.gN, threshold=isInterfaceFace))
        F = lform.assembly()

        lform1 = LinearForm(space)
        lform1.add_integrator(ScalarSourceIntegrator(pde.source))
        F1 = lform1.assembly()

        F = F + F1

        if return_form:
            return bform, lform
         
        return A, F

    def apply_bc(self, A, F):
        """
        Apply boundary conditions to the linear system.
        """
        from ..fem import DirichletBC
        if hasattr(self.pde, 'dirichlet'):
            A, F = DirichletBC(
                    self.space,
                    gd=self.pde.dirichlet,
                    threshold=self.pde.is_dirichlet_boundary).apply(A, F)
        else:
            A, F = DirichletBC(self.space, gd=self.pde.solution).apply(A, F)
        return A, F
    
    @variantmethod("direct")
    def solve(self, A, F):
        from ..solver import spsolve
        uh = spsolve(A, F, solver='scipy')
        return uh 

    @solve.register('cg')
    def solve(self, A, F):
        from ..solver import cg 
        uh, info = cg(A, F, maxit=5000, atol=1e-14, rtol=1e-14, returninfo=True)
        res = info['residual']
        res_0 = bm.linalg.norm(F)
        stop_res = res/res_0
        self.logger.info(f"CG solver with {info['niter']} iterations"
                         f" and relative residual {stop_res:.4e}")
        return uh 

    @variantmethod('onestep')
    def run(self):
        """
        """
        self.set_space_degree()
        A, F = self.linear_system()
        A, F = self.apply_bc(A, F)
        self.uh[:] = self.solve(A, F)
        l2, h1 = self.postprocess()
        self.logger.info(f"L2 Error: {l2},  H1 Error: {h1}.")

    @run.register('uniform_refine')
    def run(self, maxit=4):
        from ..mesh import TriangleMesh
        self.set_space_degree()
        for i in range(maxit):
            A, F = self.linear_system()
            A, F = self.apply_bc(A, F)
            self.uh[:] = self.solve(A, F)
            l2, h1 = self.postprocess()
            self.logger.info(f"{i}-th step with  L2 Error: {l2},  H1 Error: {h1}.")
            if i < maxit - 1:
                self.logger.info(f"Refining mesh {i+1}/{maxit}.")
                self.back_mesh.uniform_refine()
                self.mesh = TriangleMesh.interfacemesh_generator(self.back_mesh, self.pde.level_function)
    
    @variantmethod("error")
    def postprocess(self):
        """
        """
        l2 = self.mesh.error(self.pde.solution, self.uh)
        h1 = self.mesh.error(self.pde.gradient, self.uh.grad_value)
        return l2, h1