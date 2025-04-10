from fealpy.fem import BilinearForm, ScalarDiffusionIntegrator
from fealpy.fem import LinearForm, InterFaceSourceIntegrator,ScalarSourceIntegrator
from fealpy.fem import DirichletBC
from fealpy.solver import spsolve, cg
from fealpy.backend import backend_manager as bm

class InterfacePoissonLFEMModel():
    def __init__(self, mesh, space, pde, level_function):
        """
        Elliptic interface problem solver.

        Parameters:
            mesh: 2D/3D mesh.
            space: A FEM space, such as LagrangeFeSpace.
            pde: A PDE solver class. Must have a flux boundary condition attribute 'pde.flux'.
            level_function: A level set function used to describe the interface Γ.
        """

        self.mesh = mesh
        self.cell = mesh.cell
        self.node = mesh.node

        self.space= space
        self.pde = pde
        self.level_function = level_function

    def run(self):
        space = self.space
        uh = space.function()
        A, f = self.linear_system(return_form=False)
        uh[:] = self.solve(A, f)
        return uh

    def get_interface_face(self, mesh, level_function):
        ndim = mesh.number_of_nodes_of_cells()
        kwargs = bm.context(self.node)

        vertex = mesh.node[mesh.cell, :]
        bc = bm.ones(ndim, **kwargs)/(ndim)

        central_value = bm.einsum("c, qcd->qd", bc, vertex)
        adjCell = self.mesh.face_to_cell()

        isInterfaceCell = level_function(central_value)
        isInterfaceFace = (isInterfaceCell[adjCell][:, 0]*isInterfaceCell[adjCell][:, 1])<=0

        return isInterfaceFace

    def linear_system(self, return_form=False):
        mesh = self.mesh
        space = self.space
        pde = self.pde
        level_function = self.level_function

        isInterfaceFace = self.get_interface_face(mesh, level_function)

        bform = BilinearForm(space)
        bform.add_integrator(ScalarDiffusionIntegrator())
        A = bform.assembly()

        if hasattr(pde, 'flux'):
            lform = LinearForm(space)
            lform.add_integrator(InterFaceSourceIntegrator(pde.flux, threshold=isInterfaceFace))
            F = lform.assembly()

        lform1 = LinearForm(space)
        lform1.add_integrator(ScalarSourceIntegrator(pde.source))
        F1 = lform1.assembly()

        F2 =  F1 - F
        A1, F2 = DirichletBC(space, gd = pde.solution).apply(A, F2)

        if return_form:
            return bform, lform
         
        return A1, F2

    def solve(self, A, f, solver='cg'):
        if solver == 'scipy':
            return spsolve(A, f, solver='scipy')

        return cg(A, f, atol=1e-10, rtol=1e-10)