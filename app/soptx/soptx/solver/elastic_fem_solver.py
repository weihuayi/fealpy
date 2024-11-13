from dataclasses import dataclass
from typing import Optional

from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike
from fealpy.functionspace import TensorFunctionSpace

from fealpy.fem import LinearElasticIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import DirichletBC

from fealpy.sparse import CSRTensor
from fealpy.solver import cg, spsolve

from soptx.material import ElasticMaterialProperties

@dataclass
class IterativeSolverResult:
    """迭代求解器的结果"""
    displacement: TensorLike
    iterations: int
    converged: bool
    residual: float

@dataclass
class DirectSolverResult:
    """直接求解器的结果"""
    displacement: TensorLike

class ElasticFEMSolver:
    """专门用于求解线弹性问题的有限元求解器"""
    
    def __init__(self, 
                material_properties: ElasticMaterialProperties, 
                tensor_space: TensorFunctionSpace,
                pde):
        """
        初始化求解器
        
        Parameters
        ----------
        material_properties : ElasticMaterialProperties
            材料属性
        tensor_space : TensorFunctionSpace
            张量函数空间
        pde : object
            包含荷载和边界条件的PDE模型
        """
        self.material_properties = material_properties
        self.tensor_space = tensor_space
        self.pde = pde
        
        # 初始化积分器
        self.integrator = LinearElasticIntegrator(
                                material=material_properties,
                                q=tensor_space.p + 3
                            )
            
        # 缓存基础刚度矩阵(E=1)
        self._base_stiffness = None
        
        # 存储最近一次计算的系统矩阵和载荷向量
        self._current_stiffness = None
        self._current_force = None
        
    @property    
    def base_stiffness_matrix(self) -> TensorLike:
        """获取或计算基础刚度矩阵(E=1)"""
        if self._base_stiffness is None:
            base_material = self.material_properties.base_elastic_material
            base_integrator = LinearElasticIntegrator(
                                    material=base_material,
                                    q=self.tensor_space.p + 3
                                )
            self._base_stiffness = base_integrator.assembly(space=self.tensor_space)

        return self._base_stiffness
    
    def assemble_system(self) -> tuple[CSRTensor, TensorLike]:
        """组装整体刚度矩阵和载荷向量"""
        # 组装刚度矩阵
        bform = BilinearForm(self.tensor_space)
        bform.add_integrator(self.integrator)
        K = bform.assembly(format='csr')
        
        # 组装载荷向量
        force = self.pde.force
        F = self.tensor_space.interpolate(force)
        
        # 存储当前系统
        self._current_stiffness = K
        self._current_force = F
        
        return K, F
    
    def apply_boundary_conditions(self, 
                                K: CSRTensor, 
                                F: TensorLike) -> tuple[CSRTensor, TensorLike]:
        """应用边界条件"""
        dirichlet = self.pde.dirichlet
        threshold = self.pde.threshold()
        
        uh_bd = bm.zeros(self.tensor_space.number_of_global_dofs(),
                        dtype=bm.float64, device=bm.get_device(self.tensor_space))
                        
        isBdDof = self.tensor_space.is_boundary_dof(
            threshold=threshold, 
            method='interp'
        )
        
        F = F - K.matmul(uh_bd)
        F[isBdDof] = uh_bd[isBdDof]
        
        dbc = DirichletBC(
            space=self.tensor_space,
            gd=dirichlet,
            threshold=threshold,
            method='interp'
        )
        K = dbc.apply_matrix(matrix=K, check=True)
        
        return K, F
    
    def solve_cg(self, 
                maxiter: int = 5000,
                atol: float = 1e-14,
                rtol: float = 1e-14,
                x0: Optional[TensorLike] = None) -> IterativeSolverResult:
        """使用共轭梯度法求解"""
        # 组装并应用边界条件
        K, F = self.assemble_system()
        K, F = self.apply_boundary_conditions(K, F)
        
        # 求解
        uh = self.tensor_space.function()
        try:
            uh[:], info = cg(K, F[:], x0=x0, maxiter=maxiter, 
                           atol=atol, rtol=rtol, return_info=True)
        except Exception as e:
            raise RuntimeError(f"CG solver failed: {str(e)}")
            
        return IterativeSolverResult(
            displacement=uh,
            iterations=info['iterations'],
            converged=info['success'],
            residual=info['residual']
        )
    
    def solve_direct(self, solver_type: str = 'mumps') -> DirectSolverResult:
        """使用直接法求解"""
        # 组装并应用边界条件
        K, F = self.assemble_system()
        K, F = self.apply_boundary_conditions(K, F)
        
        # 求解
        uh = self.tensor_space.function()
        try:
            uh[:] = spsolve(K, F[:], solver=solver_type)
        except Exception as e:
            raise RuntimeError(f"Direct solver failed: {str(e)}")
            
        return DirectSolverResult(
            displacement=uh
        )
    
    @property
    def current_system(self) -> Optional[tuple[CSRTensor, TensorLike]]:
        """获取最近一次组装的系统矩阵和载荷向量"""
        if self._current_stiffness is None or self._current_force is None:
            return None
        return self._current_stiffness, self._current_force