from typing import Union
from fealpy.sparse import COOTensor

from fealpy.backend import bm
from fealpy.decorator import variantmethod
from fealpy.model import ComputationalModel

from fealpy.mesh import Mesh
from fealpy.functionspace import (
        LagrangeFESpace, 
        TensorFunctionSpace
        )
from fealpy.fem import DirichletBC
from fealpy.solver import spsolve, cg

from ..model.truss import TrussPDEDataT
from ..model import CSMModelManager
from ..material.bar_meterial import BarMaterial
from ..fem.bar_integrator import BarIntegrator


class TrussTowerModel(ComputationalModel):
    """
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
        self.set_bar_sections()
               
    def __str__(self) -> str:
        """Returns a formatted multi-line string summarizing the configuration of the truss tower model.
        
        Returns:
            str: A multi-line string containing the current model configuration,
            displaying information such as PDE, mesh, material properties, and more.
        """
        s = f"{self.__class__.__name__}(\n"
        s += "  --- Truss Tower Model ---\n"
        s += f"  pde            : {self.pde.__class__.__name__}\n"  # Assuming pde is a class object
        s += f"  mesh           : {self.mesh.__class__.__name__}\n"  # Assuming mesh is a class object
        s += f"  E           : {self.E}\n"
        s += f"  nu          : {self.nu}\n"
        s += f"  mu          : {self.E/(2*(1+self.nu)):.3e}\n"  # 自动算梁剪切模量
        s += f"  geo_dimension  : {self.GD}\n"
        s += "  --- Bar Sections ---\n"
        s += f"  Total bars     : {self.mesh.number_of_cells()}\n"
        s += f"  Vertical bars Area  : {self.pde.Av:.6e}\n"
        s += f"  Vertical bars Inertia: {self.pde.Iv:.6e}\n"
        s += f"  Other bars Area     : {self.pde.Ao:.6e}\n"
        s += f"  Other bars Inertia  : {self.pde.Io:.6e}\n"
        s += ")"
        self.logger.info(f"\n{s}")
        return s
               
    def set_pde(self, pde: Union[TrussPDEDataT, int] = 4) -> None:
        if isinstance(pde, int):
            self.pde = CSMModelManager("truss_tower").get_example(pde)
        else:
            self.pde = pde
        self.logger.info(f"\n{self.pde.external_load()}")
        self.logger.info(f"\n{self.pde.dirichlet_dof()}")
                
    def set_mesh(self, mesh: Mesh) -> None:
        self.mesh = mesh
              
    def set_space_degree(self, p: int) -> None:
        self.p = p
        
    def set_space(self):
        """Initialize the finite element space."""
        mesh = self.mesh
        p = self.p
        
        scalar_space = LagrangeFESpace(mesh, p=p, ctype='C')
        self.space = TensorFunctionSpace(scalar_space, shape=(-1, self.GD))
            
    def set_material(self):
        self.material = BarMaterial(
            name='BarMaterial',
            model=self.pde,
            elastic_modulus=self.E,
            poisson_ratio=self.nu
        )
    
    def set_bar_sections(self):
        """Assign cross-sectional areas to each bar element based on type.
        
        Returns:
            A: (NC,) array of cross-sectional areas for each bar
            I: (NC,) array of area moments of inertia for each bar
            
        Note: Bar classification,
            - Vertical bars: mainly along z-axis (立柱)
            - Other bars: diagonal braces and horizontal bars (斜撑和横杆)
        """
        mesh = self.mesh
        NC = mesh.number_of_cells()
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        
        # Calculate bar vectors
        bar_vectors = node[cell[:, 1]] - node[cell[:, 0]]  # (NC, 3)
        
        # Calculate bar lengths and unit vectors
        bar_length = bm.linalg.norm(bar_vectors, axis=1, keepdims=True)  # (NC, 1)
        unit_vectors = bar_vectors / (bar_length + 1e-12)  # (NC, 3)
        
        z_component = bm.abs(unit_vectors[:, 2])  # (NC,)
        xy_component = bm.sqrt(unit_vectors[:, 0]**2 + unit_vectors[:, 1]**2)  # (NC,)
        
        # A bar is vertical: (|cos(θ)| > 0.95) & (|sin(θ)| < 0.3)
        is_vertical = (z_component > 0.95) & (xy_component < 0.3)
        
        A = bm.zeros(NC, dtype=bm.float64)
        I = bm.zeros(NC, dtype=bm.float64)
        
        A[is_vertical] = self.pde.Av   # Vertical columns area
        A[~is_vertical] = self.pde.Ao  # Diagonal braces and horizontal bars area
        
        I[is_vertical] = self.pde.Iv   # Vertical columns inertia
        I[~is_vertical] = self.pde.Io  # Other bars inertia
        
        # Store in model
        self.A = A  # Cross-sectional areas (NC,)
        self.I = I  # Area moments of inertia (NC,)
        self.is_vertical = is_vertical  # Bar type flags (NC,)
        
        n_vertical = bm.sum(is_vertical)
        n_other = NC - n_vertical
        # self.logger.info(f"Vertical bars: {n_vertical} bars")
        # self.logger.info(f"Other bars: {n_other} bars")
        
        return A, I
       
    def critical_buckling_load(self):
        """Compute critical buckling loads.
        
        Parameters:
            K: Effective length factor (dimensionless)
            E: Young's modulus (Pa)
            L: Total height (m)
            I1, I2: Area moments of inertia (m^4)
            
        Returns:
            (Fc1, Fc2) : Critical buckling loads (N)
            Fc1: Critical load for buckling about X-axis (in Y-Z plane).
            Fc2: Critical load for buckling about Y-axis (in X-Z plane)
            
        Note:
            Fc = pi^2 * E * I / (K*L)^2
        """
        K = 2.0
        node = self.mesh.entity('node')
        L = bm.max(node[:, 2]) - bm.min(node[:, 2])  # Height in z-direction
        
        I1, I2 = self.pde.structural_inertia()
        
        Fc1 = bm.pi**2 * self.E * I1 / (K*L)**2  
        Fc2 = bm.pi**2 * self.E * I2 / (K*L)**2  
        
        self.logger.info(f"\n{L}")
        self.logger.info(f"\n{I1}, {I2}")
        self.logger.info(f"\n{Fc1}, {Fc2}")
        return Fc1, Fc2
            
    def linear_system(self):
        """Assemble the linear system: K*u = F.
        Returns:
            K: Global stiffness matrix (NDOF, NDOF)
            F: Load vector (NDOF,)
        """
        NDOF = self.space.number_of_global_dofs()
        
        K = bm.zeros((NDOF, NDOF), dtype=bm.float64)
        
        # 获取立柱和斜杆的索引
        vertical_indices = bm.where(self.is_vertical)[0]
        other_indices = bm.where(~self.is_vertical)[0]

        self.logger.info(f"Assembling vertical bars: {len(vertical_indices)} elements")
        vertical_integrator = BarIntegrator(
                space=self.space,
                model=self,
                material=self.material,
                index=vertical_indices
            )
        KE_vertical = vertical_integrator.assembly(self.space) # (NC_v, ldof, ldof)
        ele_dofs_vertical = vertical_integrator.to_global_dof(self.space)  # (NC_v, ldof)
        
        for i in range(len(ele_dofs_vertical)):
            dof = ele_dofs_vertical[i]
            K[dof[:, None], dof] += KE_vertical[i]
        
        self.logger.info(f"Assembling other bars: {len(other_indices)} elements")
        other_integrator = BarIntegrator(
                space=self.space,
                model=self,
                material=self.material,
                index=other_indices
            )
        KE_other = other_integrator.assembly(self.space)  # (NC_o, ldof, ldof)
        ele_dofs_other = other_integrator.to_global_dof(self.space)  # (NC_o, ldof)
        
        for i in range(len(ele_dofs_other)):
            dof = ele_dofs_other[i]
            K[dof[:, None], dof] += KE_other[i]

        F = self.pde.external_load(load_total=1.0)
        
        # self.logger.info(f"KE_vertical shape: {KE_vertical.shape}")
        # self.logger.info(f"KE_other shape: {KE_other.shape}")
        # self.logger.info(f"  Shape: {K.shape}")
        # self.logger.info(f"  Shape: {F.shape}")
        return K, F
    
    def solve(self):
        """
        Solve the linear system and return the solution.

        Returns:
            uh: Solution vector.
        """
        K, F = self.linear_system()
        gdof = self.space.number_of_global_dofs()
        
        threshold = bm.zeros(gdof, dtype=bool)
        fixed_dofs = self.pde.dirichlet_dof()
        threshold[fixed_dofs] = True
        
        rows, cols = bm.nonzero(K)
        values = K[rows, cols]
        K_sparse = COOTensor(bm.stack([rows, cols], axis=0), values, spshape=K.shape)
       
        bc = DirichletBC(
                space=self.space,
                gd=lambda p: bm.zeros(p.shape, dtype=bm.float64),  # 返回与插值点相同形状的零数组
                threshold=threshold
            )
        K_sparse, F = bc.apply(K_sparse, F)
        
        uh = bm.zeros(gdof, dtype=bm.float64)
        uh = spsolve(K_sparse, F, solver='scipy')

        # self.logger.info(f"Solution shape: {uh.shape}")
        # self.logger.info(f"Solution : {uh}")
    
        return uh
    
    def show(self, uh, filename='truss_result.vtu'):
        """输出 VTU 文件用于 ParaView 可视化."""
        
        mesh = self.space.mesh
        NN = mesh.number_of_nodes()
        
        # (NN, GD)
        disp = uh.reshape(NN, self.GD)
        mesh.nodedata['displacement'] = disp
        mesh.to_vtk(fname=filename)