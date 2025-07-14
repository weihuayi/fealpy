from typing import Union
from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')
from fealpy.model import PDEDataManager, ComputationalModel
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.fvm import (
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    ScalarCrossDiffusionIntegrator,
    DirichletBC,
)
from fealpy.fem import (
    BilinearForm,
    LinearForm
)
from fealpy.solver import spsolve
from fealpy.utils import timer


class PoissonFvmModel(ComputationalModel):
    """
    The Poisson equation in two-dimensional cases is solved by the finite volume method. 
    Through the iterative method, it has good applicability to various grid divisions.
    """
    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options.get("pbar_log", False),
                         log_level=options.get("log_level", "WARNING"))
        self.set_pde(options["pde"])
        self.set_mesh(options["nx"], options["ny"])
        self.set_space(options["space_degree"])

    def set_pde(self, pde: Union[str, object]):
        if isinstance(pde, str):
            self.pde = PDEDataManager('poisson').get_example(pde)
        else:
            self.pde = pde

    def set_mesh(self, nx: int = 10, ny: int = 10):
        box = self.pde.domain()
        self.mesh = TriangleMesh.from_box(box, nx=nx, ny=ny)

    def set_space(self, degree: int = 0):
        self.p = degree
        self.space = ScaledMonomialSpace2d(self.mesh, self.p)

    def solution(self, max_iter=6, tol=1e-6):
        pde = self.pde
        space = self.space
        mesh = self.mesh

        A = BilinearForm(space)
        A.add_integrator(ScalarDiffusionIntegrator(q=self.p + 2))
        A = A.assembly()

        lform = LinearForm(space)
        lform.add_integrator(ScalarSourceIntegrator(pde.source, q=self.p + 2))
        f = lform.assembly()

        dbc = DirichletBC(mesh, pde.dirichlet)
        A, f = dbc.apply(A, f)

        uh = spsolve(A, f)
        for i in range(max_iter):
            cross = ScalarCrossDiffusionIntegrator(uh, q=self.p + 2).assembly(space)
            rhs = f + cross
            uh_new = spsolve(A, rhs)
            err = bm.max(bm.abs(uh_new - uh))
            if err < tol:
                self.logger.info(f"Converged in {i+1} iterations, error={err}")
                break
            uh = uh_new

        self.uh = uh
        return uh

    def compute_error(self):
        cell_center = self.mesh.bc_to_point(bm.array([1 / 3, 1 / 3, 1 / 3]))
        uI = self.pde.solution(cell_center)
        error = bm.sqrt(bm.sum(self.mesh.entity_measure('cell') * (uI - self.uh)**2))
        return error