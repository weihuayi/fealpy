from ...backend import backend_manager as bm
from ..model import ComputationalModel
from ..model import PDEDataManager
from ...fem import BilinearForm
from ...fem import LinearForm
from ...functionspace import LagrangeFESpace, TensorFunctionSpace
from ...fem import DirichletBC
from ...solver import spsolve
from typing import Union
from ..model.elastoplasticity import ElastoplasticityPDEDataT


class ElastoplasticityFEMModel(ComputationalModel):
    """
    ElastoplasticityFEMModel is a finite element model for solving elastoplasticity problems.

    This class implements a finite element method (FEM) solver for elastoplasticity problems, 
    suitable for analyzing materials that exhibit both elastic and plastic behavior under loading. 
    By specifying material parameters, boundary conditions, and loads, it automatically constructs 
    the finite element mesh, assembles the stiffness matrix and load vector, and solves for the displacement field. 
    It relies on PDEDataManager for physical parameters and boundary conditions, making it suitable for teaching, 
    research, and preliminary engineering analysis.
    
    Parameters
        example : str, optional, default='elastoplasticity2d'
            Selects a preset elastoplasticity problem example for initializing PDE parameters and mesh.
    Attributes
        pde : object
            Object returned by PDEDataManager containing physical parameters and boundary conditions.
        mesh : object
            Finite element mesh object describing the discretized domain.
        E : float
            Young's modulus, describing material stiffness.
        nu : float
            Poisson's ratio, describing material compressibility.
        yield_strength : float
            Yield strength of the material.
        f : float or callable
            Distributed load applied to the domain.
        l : float
            Characteristic length of the domain.
    Methods
        run()
            Executes the FEM solution process and returns the displacement vector.
        linear_system()
            Assembles and returns the stiffness matrix and load vector for the elastoplasticity problem.
        solve()
            Applies boundary conditions and solves the linear system, returning the displacement solution.
    Notes
        This class assumes the provided PDEDataManager example defines all necessary parameters and boundary conditions.
        Supports custom loads and boundary conditions for various elastoplasticity problems.
        Depends on external finite element spaces, integrators, and linear solvers.
    Examples
        >>> model = ElastoplasticityFEMModel(example='elastoplasticity2d')
        >>> displacement = model.run()
        >>> print(displacement)
        [0.0, 0.0012, 0.0023, ...]
    """
    def __init__(self, options):
        '''
        Initializes the ElastoplasticityFEMModel with the specified example.
        Parameters:
            example (str): The name of the elastoplasticity problem example to use. Default is 'elastoplasticity2d'.
            Initializes the PDE parameters, mesh, and material properties based on the example.
        Raises:
            ValueError: If the example is not recognized or cannot be initialized.
        Notes:
            The example should be a valid key in the PDEDataManager for elastoplasticity problems.
            It must define all necessary parameters and boundary conditions.
        Examples:
            >>> model = ElastoplasticityFEMModel(example='elastoplasticity3d')
            >>> print(model.pde)
            <PDEDataManager object with elastoplasticity parameters>
        '''
        self.options = options
        super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
        self.set_pde(options['pde'])
        self.set_init_mesh()
        self.E = self.pde.E
        self.nu = self.pde.nu
        self.yield_strength = self.pde.sigma_y0
        self.f = self.pde.Ft_max
        self.a = self.pde.a
        self.N =10


    def set_pde(self, pde:Union[ElastoplasticityPDEDataT, str]='1'):
        '''
        Set the PDE parameters for the elastoplasticity problem.
        Parameters:
            pde (PDEDataManager): The PDE data manager containing elastoplasticity parameters and boundary conditions.
        Raises:
            ValueError: If the provided pde is not valid or does not contain necessary parameters.
        Notes:
            This method updates the model's physical parameters and mesh based on the provided PDE data.
        Examples:
            >>> model.set_pde(new_pde)
        '''
        if isinstance(pde, str):
            self.pde = PDEDataManager('elastoplasticity').get_example(pde)
        else:
            self.pde = pde

    def set_init_mesh(self):
        '''
        Initialize the finite element mesh based on the PDE parameters.
        This method generates the mesh according to the domain and boundary conditions defined in the PDE data.
        Raises:
            ValueError: If the mesh cannot be initialized due to invalid parameters.
        Notes:
            The mesh is generated using the parameters defined in the PDE data manager.
        Examples:
            >>> model.set_init_mesh()
        '''
        self.mesh = self.pde.init_mesh()

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")

    def linear_system(self):
        '''
        Assemble the linear system for the elastoplasticity problem.
        This method constructs the stiffness matrix and load vector based on the finite element space and PDE parameters.
        Returns:
            tuple: (A, b) where A is the stiffness matrix and b is the load vector.
        Raises:
            ValueError: If the assembly fails due to invalid parameters or mesh.
        Notes:
            The method uses bilinear forms and linear forms defined in the FEM library.
        Examples:
            >>> A, b = model.linear_system()
        '''
        pass
    

    def plasticity_update_2d(
        delta_eps, sigma_n, alpha_n, ep_n,
        E, nu, H, beta, kappa_func
    ):
        """
        2D von Mises 弹塑性更新函数（平面应变，Voigt 表示）
        
        输入参数：
        delta_eps : ndarray(3,)      # 应变增量 Δε = [ε11, ε22, γ12]
        sigma_n : ndarray(3,)        # 上步应力 σ_n
        alpha_n : ndarray(3,)        # 上步背应力 α_n
        ep_n : float                 # 上步有效塑性应变 ep_n
        E, nu : float                # 弹性模量 E 与泊松比 ν
        H : float                    # 总硬化模量 H
        beta : float                 # 混合硬化参数 β (0≤β≤1)
        kappa_func : function        # κ(ep)，屈服应力函数

        输出结果：
        sigma_np1 : ndarray(3,)      # 更新应力 σ_{n+1}
        alpha_np1 : ndarray(3,)      # 更新背应力 α_{n+1}
        ep_np1 : float               # 更新有效塑性应变 ep_{n+1}
        D_alg : ndarray(3,3)         # 一致切线刚度 D_alg (Voigt)
        """

        # -------- 弹性参数 --------
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))  # λ
        mu = E / (2 * (1 + nu))                   # μ
        K = lam + 2 * mu / 3                      # 体积模量

        # -------- 常量与辅助函数 --------
        I = bm.array([1.0, 1.0, 0.0])             # Voigt 中的单位张量
        dev = lambda v: v - bm.dot(v, I)/3 * I    # Voigt 中 dev 操作

        # -------- 1. 弹性预测 --------
        delta_eps_m = bm.dot(delta_eps, I) / 3                # Δε_m
        delta_e = dev(delta_eps)                              # Δe
        sigma_m_n = bm.dot(sigma_n, I) / 3                    # σ_m^n
        s_n = dev(sigma_n)                                    # s^n

        sigma_m = sigma_m_n + (3 * lam + 2 * mu) * delta_eps_m  # σ_m^{n+1}
        s_tr = s_n + 2 * mu * delta_e                           # s^tr
        alpha_tr = alpha_n
        ep_tr = ep_n
        eta_tr = s_tr - alpha_tr
        eta_norm = bm.linalg.norm(eta_tr)

        # -------- 2. 检查屈服 --------
        f_tr = eta_norm - bm.sqrt(2/3) * kappa_func(ep_tr)

        if f_tr <= 0:
            # 弹性步
            sigma_np1 = sigma_m * I + s_tr
            alpha_np1 = alpha_tr
            ep_np1 = ep_tr
            D_alg = bm.array([
                [lam + 2*mu, lam, 0],
                [lam, lam + 2*mu, 0],
                [0, 0, mu]
            ])
            return sigma_np1, alpha_np1, ep_np1, D_alg

        # -------- 3. 塑性修正 --------
        N = eta_tr / eta_norm
        H_alpha = beta * H
        H_kappa = (1 - beta) * H
        gamma_hat = f_tr / (2*mu + H_alpha + (2/3)*H_kappa)

        ep_np1 = ep_n + bm.sqrt(2/3) * gamma_hat
        s_np1 = s_tr - 2 * mu * gamma_hat * N
        alpha_np1 = alpha_n + H_alpha * gamma_hat * N
        sigma_np1 = sigma_m * I + s_np1

        # -------- 4. 一致切线刚度 --------
        theta1 = 1 - 2 * mu * gamma_hat / eta_norm
        theta2 = (2 * mu) / (2 * mu + H_alpha + (2/3)*H_kappa) - 2 * mu * gamma_hat / eta_norm

        I_dev = bm.eye(3) - bm.outer(I, I) / 3
        N_outer = bm.outer(N, N)
        D_alg = K * bm.outer(I, I) + 2 * mu * (
            theta1 * (I_dev - N_outer) + theta2 * N_outer
        )

        return sigma_np1, alpha_np1, ep_np1, D_alg

    
    
    
    def solve(self):
        '''
        Solve the linear system for the elastoplasticity problem.
        This method applies boundary conditions and solves the assembled linear system using a linear solver.
        Returns:
            ndarray: The displacement solution vector.
        Raises:
            ValueError: If the solution cannot be computed due to invalid parameters or assembly issues.
        Notes:
            The method uses Dirichlet boundary conditions defined in the PDE data manager.
        Examples:
            >>> displacement = model.solve()
            >>> print(displacement)
        '''
        pass
        
       
