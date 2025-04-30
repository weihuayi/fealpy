from typing import Optional
from ..model import ComputationalModel, mregister
from ..model.elliptic import EllipticPDEDataT, get_example
from ..mesh import TriangleMesh

from ..functionspace import RaviartThomasFESpace2d
from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm, LinearForm
from ..fem import ScalarMassIntegrator, SourceIntegrator
from ..fem import DivIntegrator


class EllipticRTFEMModel(ComputationalModel):
    """
    Class for the elliptic RT FEM model.
    """

    def __init__(self, pde: Optional[EllipticPDEDataT] = None):
        if pde is None:
            pde = get_example('coscos')()
        self.pde = pde



    def run(self, maxit=4):
        pde = self.pde 
        mesh = self.init_mesh()

    def init_mesh(self):
        """
        Initialize the mesh for the elliptic RT FEM model.
        """
        domain = self.pde.domain()
        self.mesh = TriangleMesh.from_box(domain, nx=2, ny=2)
        return self.mesh

    def linear_system(self):
        """
        """
        pspace = RaviartThomasFESpace2d(mesh, p=0)
        uspace = LagrangeFESpace(mesh, p=0, ctype='D') # discontinuous space
        ph = pspace.function()
        uh = uspace.function()

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        pgdof = pspace.number_of_global_dofs()
        udgof = uspace.number_of_global_dofs()

        print(f"Number of edges: {NE}")
        print(f"Number of cells: {NC}")
        print(f"Number of global dofs for pspace: {pgdof}")
        print(f"Number of global dofs for uspace: {udgof}")
        pass

    def solve(self):
        pass




    def show_mesh(self):

        import matplotlib.pyplot as plt

        fig = plt.figure()
        axes = fig.add_subplot(111)
        self.mesh.add_plot(axes)
        plt.show()







