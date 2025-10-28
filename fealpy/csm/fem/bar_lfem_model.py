from fealpy.backend import backend_manager as bm
from fealpy.model import ComputationalModel
from fealpy.decorator import variantmethod

from fealpy.mesh import Mesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import BilinearForm, DirichletBC, LinearForm
from fealpy.solver import spsolve

from fealpy.csm.fem.bar_integrator import BarIntegrator
from fealpy.csm.material import BarMaterial
from fealpy.csm.model.truss.truss_data_3d import TrussData3D
from fealpy.csm.model.model_manager import CSMModelManager

class BarLFEMModel(ComputationalModel):
    """
    3D linear FEM model for truss structures with blocked DOF layout.

    Parameters:
        options (dict): Model and solver options.

    Attributes:
        options (dict): Model options.
        pde: PDE data manager.
        mesh (Mesh): The mesh object.
        space: The finite element space.
        material: Material properties.
        E (float): Young's modulus.
        nu (float): Poisson's ratio.
        A (float): Cross-sectional area.
        GD (int): Geometric dimension.

    Methods:
        set_pde(pde): Set PDE parameters and update model.
        set_mesh(mesh): Set the mesh.
        set_space(p): Set the finite element space.
        set_material(E, nu, A): Set material properties.
        linear_system(): Assemble and return stiffness matrix and load vector.
        solve(): Solve the linear system and return the solution.
        post_process(uh): Compute displacement, strain, and stress.
        print_results(uh): Print displacement, strain, and stress.
        run(): Solve and show results.
        show(uh/strain/stress): Export results to VTK.
    """
    def __init__(self, options):
        self.options = options
        super().__init__(
                pbar_log=options['pbar_log'],
                log_level=options['log_level'])
        self.set_pde(options['pde'])
        mesh = self.pde.init_mesh()
        self.set_mesh(mesh)
        self.set_space(p=1)

        self.E = options['E']
        self.nu = options['nu']
        self.A = options['A']
        self.set_material(E=self.E, nu=self.nu, A=self.A)
        
    def set_pde(self, pde=3) -> None:
        """
        Set PDE parameters and update model.

        Parameters:
            pde: PDE data manager or int.
        """
        if isinstance(pde, int):
            self.pde = CSMModelManager('truss').get_example(pde)
        else:
            self.pde = pde
        self.GD = self.pde.geo_dimension()
        self.logger.info(self.pde)
        
    def set_mesh(self, mesh: Mesh) -> None:
        """
        Set the mesh.

        Parameters:
            mesh (Mesh): The mesh object.
        """
        self.mesh = mesh
        
    def set_space(self, p: int=1) -> None:
        """
        Set the finite element space.

        Parameters:
            p (int): Polynomial degree.
        """
        scalar_space = LagrangeFESpace(self.mesh, p)
        self.space = TensorFunctionSpace(scalar_space=scalar_space, shape=(-1, self.GD))

    def set_material(self, E: float, nu: float, A: float) -> None:
        """
        Set material properties.

        Parameters:
            E (float): Young's modulus.
            nu (float): Poisson's ratio.
            A (float): Cross-sectional area.
        """
        self.material = BarMaterial(model=self.pde,
                                    name='BarMaterial',
                                    elastic_modulus=E,
                                    poisson_ratio=nu,
                                    A=A)
        
    def linear_system(self):
        """
        Assemble and return stiffness matrix and load vector.

        Returns:
            K: Stiffness matrix.
            F_load: Load vector.
        """
        bform = BilinearForm(self.space)
        bform.add_integrator(BarIntegrator(space=self.space, material=self.material))
        K = bform.assembly()
        
        lform = LinearForm(self.space)
        lform.add_integrator(VectorSourceIntegrator(source=self.pde.source))
        F = lform.assembly()
        F_load = F.reshape(-1, self.GD)
        node = self.mesh.entity('node')
        F_load[node[...,2]==5080] = bm.array([0.0, 900, 0])
        F_load = F_load.reshape(-1)
        return K, F_load

    def solve(self):
        """
        Solve the linear system and return the solution.

        Returns:
            uh: Solution vector.
        """
        K, F = self.linear_system()
        uh = self.space.function()
        gdof = self.space.number_of_global_dofs()
        threshold = bm.zeros(gdof, dtype=bool)
        threshold[self.pde.is_displacement_boundary()] = True
        bc = DirichletBC(self.space, gd=self.pde.displacement_bc, 
                         threshold=threshold)
        K,F= bc.apply(K, F)
        uh[:] = spsolve(K, F, solver='scipy')
        return uh

    def post_process(self,uh):
        """
        Compute displacement, strain, and stress.

        Parameters:
            uh: Solution vector.

        Returns:
            uh_mat: Displacement matrix.
            strain: Strain per element.
            stress: Stress per element.
        """
        NN = self.mesh.number_of_nodes()
        uh_mat = uh[:].reshape(-1,self.GD)

        mesh = self.mesh
        E = self.E
        edge = mesh.entity('edge')
        l = mesh.edge_length()
        tan = mesh.edge_tangent()
        unit_tan = tan / l.reshape(-1, 1)

        u_edge = uh_mat[edge]
        delta_u = u_edge[:, 1, :] - u_edge[:, 0, :]
        delta_l = bm.einsum('ij,ij->i', delta_u, unit_tan)
        strain = delta_l / l
        stress = E * strain
        return uh_mat, strain, stress

    def print_results(self,uh):
        """
        Print displacement, strain, and stress.

        Parameters:
            uh: Solution vector.

        Returns:
            uh_mat: Displacement matrix.
            strain: Strain per element.
            stress: Stress per element.
        """
        uh_mat, strain, stress = self.post_process(uh=uh)
        uh_mat = bm.to_numpy(uh_mat)
        strain = bm.to_numpy(strain)
        stress = bm.to_numpy(stress)

        print("--- Displacement (uh) ---")
        print(uh_mat)
        print("------------------------")

        print("\n--- Strain per element ---")
        print(strain)
        print("--------------------------")

        print("\n--- Stress per element ---")
        print(stress)
        print("--------------------------")
        return uh_mat, strain, stress

    def run(self):
        """
        Solve and show results.

        Returns:
            uh_mat: Displacement matrix.
            strain: Strain per element.
            stress: Stress per element.
        """
        self.solve()
        return self.show()
    
    @variantmethod("displacement")
    def show(self, uh):

        mesh = self.mesh
        uh = uh.reshape(-1, self.GD)
        mesh.nodedata['displacement']=uh
        save_path = "../bar_result"
        import os
        os.makedirs(save_path, exist_ok=True)
        mesh.to_vtk(f"{save_path}/bar_displacement.vtu")

    @show.register("strain")
    def show(self, strain):
        mesh = self.mesh
        mesh.edgedata['strain'] = strain
        save_path = "../bar_result"
        import os
        os.makedirs(save_path, exist_ok=True)
        mesh.to_vtk(f"{save_path}/bars_strain.vtu")

    @show.register("stress")
    def show(self, stress):
        mesh = self.mesh

        mesh.edgedata['stress'] = stress
        save_path = "../bar_result"
        import os
        os.makedirs(save_path, exist_ok=True)
        mesh.to_vtk(f"{save_path}/bar_stress.vtu")