from typing import Union
from fealpy.backend import backend_manager as bm
from fealpy.model import ComputationalModel
from fealpy.decorator import variantmethod

from fealpy.mesh import Mesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import BilinearForm, DirichletBC, LinearForm
from fealpy.solver import spsolve

from ..fem.bar_integrator import BarIntegrator
from ..material import BarMaterial
from ..model.truss import TrussPDEDataT
from ..model.model_manager import CSMModelManager


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
        set_material(): Set material properties.
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
        self.set_space_degree(options['space_degree'])
        
        self.GD = self.pde.geo_dimension()

        self.E = options['E']
        self.nu = options['nu']
        
        self.set_space()
        self.set_material()
    
    def set_pde(self, pde: Union[TrussPDEDataT, int] = 3) -> None:
        """
        Set PDE parameters and update model.

        Parameters:
            pde: PDE data manager or int.
        """
        if isinstance(pde, int):
            self.pde = CSMModelManager('truss').get_example(pde)
        else:
            self.pde = pde
        
        self.logger.info(self.pde)
        
    def set_mesh(self, mesh: Mesh) -> None:
        """
        Set the mesh.

        Parameters:
            mesh (Mesh): The mesh object.
        """
        self.mesh = mesh
        
    def set_space_degree(self, p: int) -> None:
        self.p = p
        
    def set_space(self):
        """Initialize the finite element space."""
        mesh = self.mesh
        p = self.p
        
        scalar_space = LagrangeFESpace(mesh, p=p, ctype='C')
        self.space = TensorFunctionSpace(scalar_space, shape=(-1, self.GD))

    def set_material(self) -> None:
        """
        Set material properties.

        Parameters:
            E (float): Young's modulus.
            nu (float): Poisson's ratio.
        """
        self.material = BarMaterial(name='BarMaterial',
                                    model=self.pde,
                                    elastic_modulus=self.E,
                                    poisson_ratio=self.nu)

    def linear_system(self):
        """
        Assemble and return stiffness matrix and load vector.

        Returns:
            K: Stiffness matrix.
            F_load: Load vector.
        """
        mesh = self.space.mesh
        
        bform = BilinearForm(self.space)
        bform.add_integrator(BarIntegrator(space=self.space, 
                                        model=self.pde, 
                                        material=self.material))
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
        
        self.logger.info(f"Displacement uh:\n{uh_mat}")
        self.logger.info(f"Strain per element:\n{strain}")
        self.logger.info(f"Stress per element:\n{stress}")
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