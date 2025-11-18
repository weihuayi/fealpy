from typing import Union
from scipy.sparse.linalg import eigsh

from fealpy.backend import bm
from fealpy.sparse import COOTensor,  CSRTensor
from fealpy.model import ComputationalModel

from fealpy.mesh import Mesh
from fealpy.functionspace import (
        LagrangeFESpace, 
        TensorFunctionSpace
        )
from fealpy.fem import DirichletBC
from fealpy.solver import spsolve

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
        self.neigen = options['neigen']
        self.load = options['load']
        
        self.set_space()
        self.set_material()
        self.Fc1, self.Fc2 = self.critical_buckling_load()
               
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
        s += f"  geo_dimension  : {self.GD}\n"
        s += f"  E           : {self.E}\n"
        s += f"  nu          : {self.nu}\n"
        s += f"  mu          : {self.E/(2*(1+self.nu)):.3e}\n"  
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
        # self.logger.info(f"\n{self.pde.external_load()}")
        # self.logger.info(f"\n{self.pde.dirichlet_dof()}")
                
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
        
        return Fc1, Fc2
    
    def coo(self, K):
        """Convert a dense matrix K to COO format.
        
        Parameters:
            K: Dense matrix (NDOF, NDOF)
            
        Returns:
            K_coo: COO format matrix
        """
        rows, cols = bm.nonzero(K)
        values = K[rows, cols]
        K = COOTensor(bm.stack([rows, cols], axis=0), values, spshape=K.shape)

        return K
    
    def csr(self, K):
        """Convert a dense matrix K to CSR format.
        
        Parameters:
            K: Dense matrix (NDOF, NDOF)
            
        Returns:
            K_csr: CSR format matrix
        """
        rows, cols = bm.nonzero(K)
        values = K[rows, cols]
        crow = bm.zeros(K.shape[0] + 1, dtype=bm.int64)
        for i in range(len(rows)):
            crow[rows[i] + 1] += 1
        crow = bm.cumsum(crow)
        
        K = CSRTensor(crow, cols, values, spshape=K.shape)

        return K


    def linear_system(self):
        """Assemble the linear system: K*u = F.
        
        Returns:
            K: Global stiffness matrix (NDOF, NDOF)
            F: Load vector (NDOF,)
        """
        NDOF = self.space.number_of_global_dofs()
        
        K = bm.zeros((NDOF, NDOF), dtype=bm.float64)
        
        # 获取立柱和斜杆的索引
        vertical_indices = bm.where(self.pde.is_vertical)[0]
        other_indices = bm.where(~self.pde.is_vertical)[0]

        vertical_integrator = BarIntegrator(
                space=self.space,
                model=self.pde,
                material=self.material,
                index=vertical_indices
            )
        KE_vertical = vertical_integrator.assembly(self.space) # (NC_v, ldof, ldof)
        ele_dofs_vertical = vertical_integrator.to_global_dof(self.space)  # (NC_v, ldof)
        
        for i in range(len(ele_dofs_vertical)):
            dof = ele_dofs_vertical[i]
            K[dof[:, None], dof] += KE_vertical[i]
        
        other_integrator = BarIntegrator(
                space=self.space,
                 model=self.pde,
                material=self.material,
                index=other_indices
            )
        KE_other = other_integrator.assembly(self.space)  # (NC_o, ldof, ldof)
        ele_dofs_other = other_integrator.to_global_dof(self.space)  # (NC_o, ldof)
        
        for i in range(len(ele_dofs_other)):
            dof = ele_dofs_other[i]
            K[dof[:, None], dof] += KE_other[i]

        F = self.pde.external_load(load_total=self.load)
        
        return K, F
    
    def apply_bc(self, K, F):
        """Apply Dirichlet boundary conditions to the system."""
        
        gdof = self.space.number_of_global_dofs()
        
        threshold = bm.zeros(gdof, dtype=bool)
        fixed_dofs = self.pde.dirichlet_dof()
        threshold[fixed_dofs] = True
        
        # K_sparse = self.coo(K)
        K_sparse = self.csr(K)
        bc = DirichletBC(
                space=self.space,
                gd=lambda p: bm.zeros(p.shape, dtype=bm.float64),  # 返回与插值点相同形状的零数组
                threshold=threshold
            )
        K, F = bc.apply(K_sparse, F)
        
        isFreeDof = bm.logical_not(bc.is_boundary_dof)
        S = K.to_scipy()[isFreeDof, :][:, isFreeDof]

        return K, F, S

    def solve(self):
        """
        Solve the linear system and return the solution.

        Returns:
            uh: Solution vector.
        """
        K, F = self.linear_system()
        gdof = self.space.number_of_global_dofs()
        
        K_sparse, F, S = self.apply_bc(K, F)
        uh = bm.zeros(gdof, dtype=bm.float64)
        uh = spsolve(K_sparse, F, solver='scipy')

        # self.logger.info(f"Solution : {uh}")
    
        return uh
    
    def compute_strain_and_stress(self, disp):
        """Compute axial strain and stress for axle elements."""
                
        uh = disp.reshape(-1, 3)
        strain, stress = self.material.compute_strain_and_stress(
                        self.mesh,
                        uh,
                        ele_indices=None)
        
        # self.logger.info(f"strain: {strain}")
        # self.logger.info(f"stress: {stress}")

        return strain, stress

    def show(self, uh, strain, stress):
        """Visualize displacement field, strain field, and stress field by saving to VTU files."""
        
        mesh = self.space.mesh
        save_path = "../truss_tower_result"
        
        disp = uh.reshape(-1, self.GD)
    
        import os
        os.makedirs(save_path, exist_ok=True)
        
        mesh.nodedata['displacement'] = disp
        mesh.to_vtk(f"{save_path}/disp.vtu")
        
        mesh.edgedata['strain'] = strain
        mesh.to_vtk(f"{save_path}/strain.vtu")

        mesh.edgedata['stress'] = stress
        mesh.to_vtk(f"{save_path}/stress.vtu")

    def geometric_stiffness_matrix(self, stress):
        """Assemble the global geometric stiffness matrix for buckling analysis.
        
        Parameters:
            stress: (NC,) array of axial stresses in each bar element.
            
        Returns:
            Kg: geometric stiffness matrix.
        """
        NDOF = self.space.number_of_global_dofs()
        
        Kg = bm.zeros((NDOF, NDOF), dtype=bm.float64)
        
        vertical_indices = bm.where(self.pde.is_vertical)[0]
        other_indices = bm.where(~self.pde.is_vertical)[0]
        vertical_integrator = BarIntegrator(
                space=self.space,
                 model=self.pde,
                material=self.material,
                index=vertical_indices,
                method='geometric'
            )
        Kg_vertical = vertical_integrator.assembly(self.space, stress)
        ele_dofs_vertical = vertical_integrator.to_global_dof(self.space)
        
        for i in range(len(ele_dofs_vertical)):
            dof = ele_dofs_vertical[i]
            Kg[dof[:, None], dof] += Kg_vertical[i]
            
        other_integrator = BarIntegrator(
                space=self.space,
                model=self.pde,
                material=self.material,
                index=other_indices,
                method='geometric'
            )
        Kg_other = other_integrator.assembly(self.space, stress)
        ele_dofs_other = other_integrator.to_global_dof(self.space)
    
        for i in range(len(ele_dofs_other)):
            dof = ele_dofs_other[i]
            Kg[dof[:, None], dof] += Kg_other[i]
        
        # self.logger.info(f"kg: {Kg}")

        return Kg
    
    def apply_matrix(self, K):
        """Apply boundary conditions to matrix K."""
        
        gdof = self.space.number_of_global_dofs()
        
        threshold = bm.zeros(gdof, dtype=bool)
        fixed_dofs = self.pde.dirichlet_dof()
        threshold[fixed_dofs] = True
        
        # K_sparse = self.coo(K)
        K_sparse = self.csr(K)
        
        bc = DirichletBC(
                space=self.space,
                gd=lambda p: bm.zeros(p.shape, dtype=bm.float64),  # 返回与插值点相同形状的零数组
                threshold=threshold
            )
        Kg = bc.apply_matrix(K_sparse)
        
        isFreeDof = bm.logical_not(bc.is_boundary_dof)
        M = Kg.to_scipy()[isFreeDof, :][:, isFreeDof]

        return Kg, M
    
    def buckling_analysis(self, stress):
        """Perform buckling analysis using the geometric stiffness matrix.
        
        Parameters:
            stress: (NC,) array of axial stresses in each bar element
            neigen: Number of eigenvalues to compute (default: 6)
            
        Returns:
            val: Eigenvalues (buckling load factors)
            vec: Eigenvectors (buckling modes)
        """
        K, F = self.linear_system()
        K, F, S = self.apply_bc(K, F)
        
        Kg = self.geometric_stiffness_matrix(stress)
        Kg, M = self.apply_matrix(Kg)
       

        # 求解广义特征值问题: K * v = λ * M * v
        val, vec = eigsh(S, k=self.neigen, M=M, which='SM', tol=1e-6, maxiter=1000)

        # self.logger.info(f"Buckling eigenvalues: {val}")
        # self.logger.info(f"Critical buckling load factor: {val[0]:.6e}")
    
        return val, vec