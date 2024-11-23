from dataclasses import dataclass
from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Union
from fealpy.functionspace import TensorFunctionSpace
from fealpy.fem import LinearElasticIntegrator, BilinearForm, DirichletBC
from fealpy.sparse import CSRTensor
from fealpy.solver import cg, spsolve

from soptx.material import ElasticMaterialInstance, ElasticMaterialProperties

@dataclass
class IterativeSolverResult:
    """迭代求解器的结果"""
    displacement: TensorLike

@dataclass
class DirectSolverResult:
    """直接求解器的结果"""
    displacement: TensorLike

class ElasticFEMSolver:
    """专门用于求解线弹性问题的有限元求解器
    
    该求解器负责：
    1. 密度状态的管理
    2. 对应密度下的材料实例管理
    3. 有限元离散化和求解
    4. 边界条件的处理
    """
    
    def __init__(self, 
                material_properties: ElasticMaterialProperties, 
                tensor_space: TensorFunctionSpace,
                pde,
                solver_type: str = 'cg',
                solver_params: Optional[dict] = None):
        """
        Parameters
        ----------
        material_properties : 材料属性计算器
        tensor_space : 张量函数空间
        pde : 包含荷载和边界条件的PDE模型
        solver_type : 求解器类型, 'cg' 或 'direct' 
        solver_params : 求解器参数
            cg: maxiter, atol, rtol
            direct: solver_type
        """
        self.material_properties = material_properties
        self.tensor_space = tensor_space
        self.pde = pde
        self.solver_type = solver_type
        self.solver_params = solver_params or {}

        # 状态管理
        self._current_density = None
        self._current_material = None
            
        # 缓存
        self._base_local_stiffness_matrix = None
        self._global_stiffness_matrix = None
        self._global_force_vector = None

    #---------------------------------------------------------------------------
    # 公共属性
    #---------------------------------------------------------------------------
    @property
    def current_density(self) -> Optional[TensorLike]:
        """获取当前密度"""
        return self._current_density

    #---------------------------------------------------------------------------
    # 状态管理相关方法
    #---------------------------------------------------------------------------
    def update_density(self, density: TensorLike) -> None:
        """更新密度并更新相关状态
        
        Parameters
        ----------
        density : 新的密度场
        """
        if density is None:
            raise ValueError("'density' cannot be None")
            
        self._current_density = density
        # 根据新密度计算材料属性
        E = self.material_properties.calculate_elastic_modulus(density)
        self._current_material = ElasticMaterialInstance(E, self.material_properties.config)
        
        # 清除依赖于密度的缓存
        self._global_stiffness_matrix = None
        self._global_force_vector = None

    def get_current_material(self) -> Optional[ElasticMaterialInstance]:
        """获取当前材料实例"""
        if self._current_material is None:
            raise ValueError("Material not initialized. Call update_density first.")
        return self._current_material

    #---------------------------------------------------------------------------
    # 矩阵计算和缓存相关方法
    #---------------------------------------------------------------------------
    def get_global_stiffness_matrix(self) -> Optional[CSRTensor]:
        """获取最近一次求解时的全局刚度矩阵"""
        return self._global_stiffness_matrix
        
    def get_global_force_vector(self) -> Optional[TensorLike]:
        """获取最近一次求解时的全局载荷向量"""
        return self._global_force_vector
    
    def get_base_local_stiffness_matrix(self) -> TensorLike:
        """获取基础材料的局部刚度矩阵（会被缓存）"""
        if self._base_local_stiffness_matrix is None:
            base_material = self.material_properties.get_base_material()
            integrator = LinearElasticIntegrator(
                                            material=base_material,
                                            q=self.tensor_space.p + 3
                                        )
            self._base_local_stiffness_matrix = integrator.assembly(space=self.tensor_space)
        return self._base_local_stiffness_matrix
    
    def compute_local_stiffness_matrix(self) -> TensorLike:
        """计算当前材料的局部刚度矩阵（每次重新计算）"""
        if self._current_material is None:
            raise ValueError("Material not initialized. Call update_density first.")
        
        integrator = LinearElasticIntegrator(
                                        material=self._current_material,
                                        q=self.tensor_space.p + 3
                                    )

        KE = integrator.assembly(space=self.tensor_space)
        
        return KE

    #---------------------------------------------------------------------------
    # 内部方法：组装和边界条件
    #---------------------------------------------------------------------------
    def _assemble_global_stiffness_matrix(self) -> CSRTensor:
        """组装全局刚度矩阵"""
        if self._current_material is None:
            raise ValueError("Material not initialized. Call update_density first.")
            
        integrator = LinearElasticIntegrator(
                                        material=self._current_material,
                                        q=self.tensor_space.p + 3
                                    )
        bform = BilinearForm(self.tensor_space)
        bform.add_integrator(integrator)
        K = bform.assembly(format='csr')
        self._global_stiffness_matrix = K

        return K
    
    def _assemble_global_force_vector(self) -> TensorLike:
        """组装全局载荷向量"""
        force = self.pde.force
        F = self.tensor_space.interpolate(force)
        self._global_force_vector = F

        return F
    
    def _apply_boundary_conditions(self, K: CSRTensor, F: TensorLike) -> tuple[CSRTensor, TensorLike]:
        """应用边界条件
        
        Parameters
        ----------
        K : 全局刚度矩阵
        F : 全局载荷向量
        
        Returns
        -------
        K : 处理边界条件后的刚度矩阵
        F : 处理边界条件后的载荷向量
        """
        dirichlet = self.pde.dirichlet
        threshold = self.pde.threshold()
        
        uh_bd = bm.zeros(self.tensor_space.number_of_global_dofs(),
                        dtype=bm.float64, device=bm.get_device(self.tensor_space))
                        
        isBdDof = self.tensor_space.is_boundary_dof(threshold=threshold, method='interp')
        
        F = F - K.matmul(uh_bd)
        F[isBdDof] = uh_bd[isBdDof]
        
        dbc = DirichletBC(space=self.tensor_space, gd=dirichlet,
                        threshold=threshold, method='interp')
        K = dbc.apply_matrix(matrix=K, check=True)
        
        return K, F

    #---------------------------------------------------------------------------
    # 求解方法
    #---------------------------------------------------------------------------
    def solve(self) -> Union[IterativeSolverResult, DirectSolverResult]:
        """统一的求解接口"""
        if self.solver_type == 'cg':
            return self.solve_cg(**self.solver_params)
        elif self.solver_type == 'direct':
            return self.solve_direct(**self.solver_params)
        else:
            raise ValueError(f"Unsupported solver type: {self.solver_type}")
               
    def solve_cg(self, 
                maxiter: int = 5000,
                atol: float = 1e-12,
                rtol: float = 1e-12,
                x0: Optional[TensorLike] = None) -> IterativeSolverResult:
        """使用共轭梯度法求解
        
        Parameters
        ----------
        maxiter : 最大迭代次数
        atol : 绝对收敛容差
        rtol : 相对收敛容差
        x0 : 初始猜测值
        
        Returns
        -------
        IterativeSolverResult
            求解结果，包含位移场
        """
        if self._current_density is None:
            raise ValueError("Density not set. Call update_density first.")
    
        K0 = self._assemble_global_stiffness_matrix()
        F0 = self._assemble_global_force_vector()
        K, F = self._apply_boundary_conditions(K0, F0)
        
        uh = self.tensor_space.function()
        try:
            # logger.setLevel('INFO')
            # uh[:], info = cg(K, F[:], x0=x0, maxiter=maxiter, atol=atol, rtol=rtol)
            # TODO 目前 FEALPy 中的 cg 只能通过 logger 获取迭代步数，无法直接返回
            uh[:] = cg(K, F[:], x0=x0, maxiter=maxiter, atol=atol, rtol=rtol)
        except Exception as e:
            raise RuntimeError(f"CG solver failed: {str(e)}")
        
        return IterativeSolverResult(displacement=uh)
    
    def solve_direct(self, solver_type: str = 'mumps') -> DirectSolverResult:
        """使用直接法求解
        
        Parameters
        ----------
        solver_type : 求解器类型，默认为'mumps'
        
        Returns
        -------
        DirectSolverResult
            求解结果，包含位移场
        """
        if self._current_density is None:
            raise ValueError("Density not set. Call update_density first.")
    
        K0 = self._assemble_global_stiffness_matrix()
        F0 = self._assemble_global_force_vector()
        K, F = self._apply_boundary_conditions(K0, F0)
        
        uh = self.tensor_space.function()
        try:
            uh[:] = spsolve(K, F[:], solver=solver_type)
        except Exception as e:
            raise RuntimeError(f"Direct solver failed: {str(e)}")
            
        return DirectSolverResult(displacement=uh)