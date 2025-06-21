from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Union
from fealpy.functionspace import TensorFunctionSpace
from fealpy.fem import LinearElasticIntegrator, BilinearForm, DirichletBC
from fealpy.sparse import CSRTensor
from fealpy.solver import cg, spsolve

from soptx.material import ElasticMaterialInstance
from soptx.utils import timer

@dataclass
class IterativeSolverResult:
    """迭代求解器的结果"""
    displacement: TensorLike

@dataclass
class DirectSolverResult:
    """直接求解器的结果"""
    displacement: TensorLike

class AssemblyMethod(Enum):
    """矩阵组装方法的枚举类"""
    STANDARD = auto()             # 标准组装
    VOIGT = auto()                # Voigt 格式组装
    VOIGT_UNIFORM = auto          # Voigt 格式组装 (一致网格)
    FAST_STRESS_UNIFORM = auto()  # 2D 应力快速组装 (一致网格)
    FAST_3D_UNIFORM = auto()      # 3D 快速组装 (一致网格)
    SYMBOLIC = auto()             # 符号组装

class ElasticFEMSolver:
    """专门用于求解线弹性问题的有限元求解器
    
    该求解器负责：
    1. 密度状态的管理
    2. 对应密度下的材料实例管理
    3. 有限元离散化和求解
    4. 边界条件的处理
    """
    
    def __init__(self, 
                materials: ElasticMaterialInstance, 
                tensor_space: TensorFunctionSpace,
                pde,
                assembly_method: AssemblyMethod,
                solver_type: str,
                solver_params: Optional[dict]):
        """
        Parameters
        - materials : 材料
        - tensor_space : 张量函数空间
        - pde : 包含荷载和边界条件的 PDE 模型
        - assembly_method : 矩阵组装方法
        - solver_type : 求解器类型, 'cg' 或 'direct' 
        - solver_params : 求解器参数
            - cg: maxiter, atol, rtol
            - direct: solver_type
        """
        self.materials = materials
        self.tensor_space = tensor_space
        self.pde = pde
        self.assembly_method = assembly_method
        self.solver_type = solver_type
        self.solver_params = solver_params or {}

        self._integrator = self._create_integrator()

        # 状态管理
        self._current_density = None
            
        # 缓存
        self._base_local_stiffness_matrix = None
        self._global_stiffness_matrix = None
        self._global_force_vector = None


    #---------------------------------------------------------------------------
    # 公共属性
    #---------------------------------------------------------------------------
    @property
    def get_current_density(self) -> Optional[TensorLike]:
        """获取当前密度"""
        return self._current_density
    
    @property
    def get_current_material(self) -> Optional[ElasticMaterialInstance]:
        """获取当前材料实例"""
        if self.materials is None:
            raise ValueError("Material not initialized. Call update_density first.")
        
        return self.materials

    #---------------------------------------------------------------------------
    # 状态管理相关方法
    #---------------------------------------------------------------------------
    def update_status(self, density: TensorLike) -> None:
        """更新相关状态"""
        if density is None:
            raise ValueError("'density' cannot be None")
            
        # 1. 更新密度场
        self._current_density = density

        # 2. 根据新密度更新材料属性
        self.materials.update_elastic_modulus(self._current_density)

        # 3. 清除依赖于密度的缓存
        self._global_stiffness_matrix = None
        self._global_force_vector = None

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
                                q=self.tensor_space.p+3,
                                method='voigt_uniform'
                            )
            self._base_local_stiffness_matrix = integrator.voigt_assembly_uniform(space=self.tensor_space)

        return self._base_local_stiffness_matrix
    
    def compute_local_stiffness_matrix(self) -> TensorLike:
        """计算当前材料的局部刚度矩阵（每次重新计算）"""
        if self._current_material is None:
            raise ValueError("Material not initialized. Call update_density first.")
        
        integrator = self._integrator
 
        # 根据 assembly_config.method 选择对应的组装函数
        method_map = {
            AssemblyMethod.STANDARD: integrator.assembly,
            AssemblyMethod.VOIGT: integrator.voigt_assembly,
            AssemblyMethod.VOIGT_UNIFORM: integrator.voigt_assembly_uniform,
            AssemblyMethod.FAST_STRESS_UNIFORM: integrator.fast_assembly_stress_uniform,
            AssemblyMethod.FAST_3D_UNIFORM: integrator.fast_assembly_uniform,
            AssemblyMethod.SYMBOLIC: integrator.symbolic_assembly,
        }
        
        try:
            assembly_func = method_map[self.assembly_method]
        except KeyError:
            raise RuntimeError(f"Unsupported assembly method: {self.assembly_method}")
        
        # 调用选定的组装函数
        KE = assembly_func(space=self.tensor_space)
        
        return KE

    #---------------------------------------------------------------------------
    # 内部方法：组装和边界条件处理
    #---------------------------------------------------------------------------
    def _create_integrator(self) -> LinearElasticIntegrator:
        """创建适当的积分器实例"""
        # 确定积分方法
        method_map = {
            AssemblyMethod.STANDARD: 'assembly',
            AssemblyMethod.VOIGT: 'voigt',
            AssemblyMethod.VOIGT_UNIFORM: 'voigt_uniform',
            AssemblyMethod.FAST_STRESS_UNIFORM: 'fast_stress_uniform',
            AssemblyMethod.FAST_3D_UNIFORM: 'fast_3d_uniform',
            AssemblyMethod.SYMBOLIC: 'symbolic',
        }
        
        method = method_map[self.assembly_method]

        # 创建积分器
        q = self.tensor_space.p + 3
        integrator = LinearElasticIntegrator(
                            material=self.materials, 
                            q=q, method=method
                        )
        integrator.keep_data()
        
        return integrator
    
    def _assemble_global_stiffness_matrix(self) -> CSRTensor:
        """组装全局刚度矩阵"""    
        integrator = self._integrator
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
        """应用边界条件"""
        dirichlet = self.pde.dirichlet
        threshold = self.pde.threshold()
        
        uh_bd = bm.zeros(self.tensor_space.number_of_global_dofs(),
                            dtype=bm.float64, device=bm.get_device(self.tensor_space))
                        
        uh_bd, isBdDof = self.tensor_space.boundary_interpolate(
                            gd=dirichlet, threshold=threshold, method='interp')

        F = F - K.matmul(uh_bd[:])  
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
        - maxiter : 最大迭代次数
        - atol : 绝对收敛容差
        - rtol : 相对收敛容差
        - x0 : 初始猜测值
        """
        if self._current_density is None:
            raise ValueError("Density not set. Call update_density first.")
        
        # 创建计时器
        t = timer("CG Solver Timing")
        next(t)  # 启动计时器
    
        K0 = self._assemble_global_stiffness_matrix()
        t.send('Stiffness Matrix Assembly')

        F0 = self._assemble_global_force_vector()
        t.send('Force Vector Assembly')

        K, F = self._apply_boundary_conditions(K0, F0)
        t.send('Boundary Conditions')
        
        uh = self.tensor_space.function()

        try:
            # logger.setLevel('INFO')
            # TODO 目前 FEALPy 中的 cg 只能通过 logger 获取迭代步数，无法直接返回
            uh[:] = cg(K, F[:], x0=x0, maxiter=maxiter, atol=atol, rtol=rtol)
            t.send('Solving Phase')  # 记录求解阶段时间

            # 结束计时
            t.send(None)
        except Exception as e:
            raise RuntimeError(f"CG solver failed: {str(e)}")
        
        
        return IterativeSolverResult(displacement=uh)
    
    def solve_direct(self, solver_type: str = 'mumps') -> DirectSolverResult:
        """使用直接法求解"""
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