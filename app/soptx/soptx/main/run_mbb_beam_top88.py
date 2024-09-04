from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import UniformMesh2d
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace

from ...soptx.cases.mbb_beam_cases import MBBBeamCase
from ...soptx.utilfunc.fem_solver import FEMSolver
from ...soptx.optalg.oc_optimizer import OCOptimizer
from ...soptx.cases.termination_criterias import TerminationCriteria
from ...soptx.cases.top_opt_porblem import ComplianceMinimization
from ...soptx.utilfunc.sensitivity_calculations import manual_objective_sensitivity



# 设置计算后端，如 'numpy', 'pytorch' 等
backend = 'numpy'
bm.set_backend(backend)

# 初始化 MBB 梁算例
mbb_case = MBBBeamCase(case_name="top88")

# 定义网格和有限元空间
nx, ny = mbb_case.nx, mbb_case.ny
extent = [0, nx, 0, ny]
h = mbb_case.h  # 单元尺寸
origin = [0, 0]
mesh = UniformMesh2d(extent, h, origin)

# 创建有限元空间
p_C = 1
space_C = LagrangeFESpace(mesh, p=p_C, ctype='C')
tensor_space = TensorFunctionSpace(space_C, shape=(-1, 2))

p_D = 0
space_D = LagrangeFESpace(mesh, p=p_D, ctype='D')        
rho = space_D.function()  # 初始化设计变量
rho_phys = space_D.function()  # 初始化物理密度
rho[:] = mbb_case.rho
rho_phys[:] = mbb_case.rho  # 初始时物理密度和设计变量相同

# 创建材料属性
material_properties = mbb_case.material_properties

boundary_conditions = mbb_case.boundary_conditions

# 创建 FEM 求解器
fem_solver = FEMSolver(material_properties=material_properties, 
                       tensor_space=tensor_space,
                       rho=rho_phys,
                       boundary_conditions=boundary_conditions)

# 创建优化终止条件
termination_criteria = mbb_case.termination_criterias

# 定义目标函数
objective_function = ComplianceMinimization()

# 创建滤波器参数（如果需要）
filter_properties = mbb_case.filter_properties

constraint_conditions = mbb_case.constraint_conditions

# 创建 OC 优化器
optimizer = OCOptimizer(displacement_solver=fem_solver, 
                        objective_function=objective_function,
                        sensitivity_function=manual_objective_sensitivity,
                        termination_criteria=termination_criteria,
                        constraint_conditions=constraint_conditions,
                        filter_parameters=filter_properties)


# 执行优化
optimized_rho = optimizer.optimize(rho=rho)

# 输出优化结果
print(f"Optimized Density Distribution:\n{optimized_rho}")
